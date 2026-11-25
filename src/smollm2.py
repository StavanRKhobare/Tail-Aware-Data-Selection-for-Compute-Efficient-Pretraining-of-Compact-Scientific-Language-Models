"""Prepare, train, and evaluate matched SmolLM2 continued-pretraining runs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import config
from data import TokenWindows
from export_tokens import encode
from train import TrainConfig, _autocast, load_checkpoint, resolve_amp, resolve_device, train


STRATEGIES = ("random", "tail_aware")
GROUPS = ("head", "mid", "tail")
BASE = config.ROOT / "runs" / "smollm2_base"
DATA = config.ROOT / "data" / "subsets_smollm2"


def fingerprint(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def base_metadata(base):
    return {
        "backend": "smollm2",
        "tokenizer_sha256": fingerprint(base / "tokenizer.json"),
        "weights_sha256": fingerprint(base / "model.safetensors"),
        "config_sha256": fingerprint(base / "config.json"),
    }


def prepare(base, output):
    model_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
    eos = model_config["eos_token_id"]
    vocab_size = tokenizer.get_vocab_size()
    if not 0 <= eos < vocab_size <= min(model_config["vocab_size"], 65536):
        raise ValueError("Base tokenizer, vocabulary, or EOS is incompatible")
    meta = {
        "base": base_metadata(base), "vocab_size": vocab_size, "eot_id": eos,
        "pad_id": model_config.get("pad_token_id") or eos, "files": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    train_docs = pd.read_parquet(config.PROCESSED_DIR / "train.parquet", columns=["id", "text"]).set_index("id")
    for subset in STRATEGIES:
        with gzip.open(config.MANIFEST_DIR / f"{subset}.txt.gz", "rt") as manifest:
            ids = np.random.default_rng(config.SEED).permutation(manifest.read().split())
        tokens, lengths = encode(tokenizer, train_docs.loc[ids, "text"].tolist(), eos, add_special_tokens=False)
        (output / subset).mkdir()
        tokens.tofile(output / subset / "train.bin")
        meta["files"][f"{subset}/train.bin"] = {"docs": len(lengths), "tokens": len(tokens)}
        print(f"{subset}: {len(lengths):,} documents, {len(tokens):,} tokens", flush=True)
    del train_docs

    topics = pd.read_parquet(config.SELECTION_DIR / "topics.parquet")
    for split in ("val", "test"):
        docs = pd.read_parquet(config.PROCESSED_DIR / f"{split}.parquet", columns=["id", "text"]).sort_values("id")
        docs = docs.merge(topics.loc[topics["split"] == split, ["id", "group"]], on="id", how="left", validate="one_to_one")
        if not docs["group"].isin(GROUPS).all():
            raise ValueError(f"Missing or unknown group labels in {split}")
        tokens, lengths = encode(tokenizer, docs.pop("text").tolist(), eos, add_special_tokens=False)
        ends = np.cumsum(lengths)
        docs.assign(start=ends - lengths, end=ends).to_parquet(output / f"{split}_docs.parquet", index=False)
        tokens.tofile(output / f"{split}.bin")
        meta["files"][f"{split}.bin"] = {"docs": len(docs), "tokens": len(tokens)}
        print(f"{split}: {len(docs):,} documents, {len(tokens):,} tokens", flush=True)
    (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class CausalLM(torch.nn.Module):
    """Adapt HF logits to the trainer's already-shifted targets."""

    def __init__(self, base):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, use_safetensors=True, dtype=torch.float32,
        )

    def forward(self, x, y=None):
        logits = self.model(input_ids=x, use_cache=False).logits
        loss = None if y is None else F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
        return logits, loss

    def num_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())


def check_data(args):
    meta = json.loads((args.data / "meta.json").read_text(encoding="utf-8"))
    if meta["base"] != base_metadata(args.base):
        raise ValueError("Local base files do not match the data preparation metadata")
    return {**meta["base"], "block_size": args.block_size}


def train_model(args):
    metadata = check_data(args)
    if not args.dry_run and not args.resume and args.out.exists():
        raise FileExistsError(f"Run directory already exists: {args.out}. Use --resume or a new path.")
    settings = TrainConfig(
        subset=args.subset, data_dir=args.data, out_dir=args.out, seed=args.seed,
        block_size=args.block_size, batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_tokens=args.tokens, max_steps=args.max_steps, lr=args.lr, device=args.device,
        eval_every=args.eval_every, save_every=args.eval_every, resume=args.resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(train(settings, model_factory=lambda: CausalLM(args.base), model_metadata=metadata), indent=2))


@torch.inference_mode()
def measure(model, windows, docs, batch_size, device):
    ends = docs["end"].to_numpy(dtype=np.int64)
    groups = docs["group"].map({name: index for index, name in enumerate(GROUPS)}).to_numpy(dtype=np.int64)
    sums = np.zeros(len(GROUPS), dtype=np.float64)
    counts = np.zeros(len(GROUPS), dtype=np.int64)
    block = windows.block_size
    amp = resolve_amp(device, True)
    model.eval()
    for start in range(0, windows.n_windows, batch_size):
        indices = np.arange(start, min(start + batch_size, windows.n_windows))
        x, y = windows.get_batch(indices)
        with _autocast(device, amp):
            logits, _ = model(x.to(device))
        losses = F.cross_entropy(logits.flatten(0, 1).float(), y.to(device).flatten(), reduction="none")
        offsets = (indices[:, None] * block + np.arange(1, block + 1)).ravel()
        labels = groups[np.searchsorted(ends, offsets, side="right")]
        sums += np.bincount(labels, weights=losses.cpu().numpy(), minlength=len(GROUPS))
        counts += np.bincount(labels, minlength=len(GROUPS))
    return {
        "overall": float(sums.sum() / counts.sum()),
        "loss_by_group": {name: float(total / count) if count else None for name, total, count in zip(GROUPS, sums, counts)},
        "tokens_by_group": dict(zip(GROUPS, counts.tolist())),
    }


def evaluate_models(args):
    metadata = check_data(args)
    device = resolve_device(args.device)
    docs = pd.read_parquet(args.data / f"{args.split}_docs.parquet").sort_values("start")
    windows = TokenWindows(args.data / f"{args.split}.bin", args.block_size)
    model = CausalLM(args.base).to(device)
    rows = {"base": measure(model, windows, docs, args.batch_size, device)}
    for name in STRATEGIES:
        path = getattr(args, name)
        checkpoint = load_checkpoint(path)
        if checkpoint["config"] != metadata:
            raise ValueError(f"Incompatible checkpoint: {path}")
        model.load_state_dict(checkpoint["model"])
        rows[name] = {
            **measure(model, windows, docs, args.batch_size, device),
            "checkpoint": str(path), "step": checkpoint["step"], "training_tokens": checkpoint["tokens_seen"],
        }
    if rows["random"]["training_tokens"] != rows["tail_aware"]["training_tokens"]:
        raise ValueError("Comparison requires matched training budgets")
    report = {"split": args.split, "model": metadata, "runs": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    fit = commands.add_parser("train")
    evaluate = commands.add_parser("evaluate")
    for command in (prep, fit, evaluate):
        command.add_argument("--base", type=Path, default=BASE)
        command.add_argument("--out", type=Path, required=command is not prep, default=DATA if command is prep else None)
    for command in (fit, evaluate):
        command.add_argument("--data", type=Path, default=DATA)
        command.add_argument("--block-size", type=int, default=512)
        command.add_argument("--batch-size", type=int, default=2)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    fit.add_argument("--subset", choices=STRATEGIES, required=True)
    fit.add_argument("--seed", type=int, default=config.SEED)
    fit.add_argument("--tokens", type=int, default=60_000_000)
    fit.add_argument("--grad-accum", type=int, default=16)
    fit.add_argument("--lr", type=float, default=5e-5)
    fit.add_argument("--eval-every", type=int, default=1000)
    fit.add_argument("--max-steps", type=int)
    fit.add_argument("--resume", action="store_true")
    fit.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    for subset in STRATEGIES:
        evaluate.add_argument("--" + subset.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.base, args.out)
    elif args.command == "train":
        if args.eval_every < 1:
            parser.error("--eval-every must be positive")
        train_model(args)
    else:
        evaluate_models(args)


if __name__ == "__main__":
    main()
