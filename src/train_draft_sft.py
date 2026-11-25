"""Run a bounded output-masked drafting fine-tuning pilot."""
import argparse, json, random, time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import GPT

ROOT = Path(__file__).resolve().parent.parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'data'/'draft_sft_v2')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'runs'/'random'/'seed42'/'ckpt_best.pt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--smollm2-base', type=Path, help='Local base files for a SmolLM2 continued-pretraining checkpoint')
    args = p.parse_args()
    if args.steps < 1 or args.eval_every < 1: p.error('steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.smollm2_base:
        from smollm2 import CausalLM, base_metadata
        from train import load_checkpoint
        ckpt = load_checkpoint(args.checkpoint)
        if any(ckpt['config'].get(key) != value for key, value in base_metadata(args.smollm2_base).items()):
            p.error('checkpoint does not match the local SmolLM2 base')
        model = CausalLM(args.smollm2_base)
        model.load_state_dict(ckpt['model'])
        tok = Tokenizer.from_file(str(args.smollm2_base/'tokenizer.json'))
        eot = model.model.config.eos_token_id
        context_size = model.model.config.max_position_embeddings
    else:
        model, ckpt = GPT.from_checkpoint(args.checkpoint)
        tok = Tokenizer.from_file(str(ROOT/'tokenizer'/'tokenizer.json'))
        eot = tok.token_to_id('<|endoftext|>')
        context_size = model.config.block_size
    checkpoint_config = ckpt['config']
    del ckpt
    model.to(device)
    def load(split):
        rows = [json.loads(line) for line in (args.data/f'{split}.jsonl').read_text(encoding='utf8').splitlines()]
        if not rows: raise ValueError(f'empty {split} set')
        return rows
    train, val = load('train'), load('val')
    def encode(row):
        return (
            tok.encode(row['prompt'], add_special_tokens=not args.smollm2_base).ids,
            tok.encode(row['target'], add_special_tokens=not args.smollm2_base).ids + [eot],
        )
    def loss(example):
        prefix, target = example
        ids = prefix + target
        if len(ids) > context_size: raise ValueError('example exceeds context')
        x = torch.tensor([ids[:-1]],device=device)
        y = torch.tensor([[(-100 if i < len(prefix) else ids[i]) for i in range(1,len(ids))]],device=device)
        return model(x,y)[1]
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    train_ids = {r['id'] for r in train}
    if train_ids & {r['id'] for r in val}:
        raise ValueError('Train and validation paper IDs overlap')
    train, val = [encode(row) for row in train], [encode(row) for row in val]
    val_tokens = sum(len(target) for _, target in val)
    best = float('inf')
    start = time.perf_counter()
    with (args.output/'metrics.jsonl').open('x', encoding='utf8') as log:
        for step in range(1,args.steps+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            l = loss(random.choice(train)); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                model.eval()
                with torch.no_grad(): v = sum(float(loss(example)) * len(example[1]) for example in val) / val_tokens
                if v < best:
                    best = v
                    torch.save({'model':model.state_dict(),'config':checkpoint_config,'step':step,'draft_sft':True,'val_loss':v},args.output/'ckpt_best.pt')
                record = {'step':step,'train_loss':float(l.detach()),'val_loss':v,'elapsed_s':time.perf_counter()-start}
                log.write(json.dumps(record)+'\n'); log.flush(); print(record,flush=True)
    torch.save({'model':model.state_dict(),'config':checkpoint_config,'step':args.steps,'draft_sft':True},args.output/'ckpt_last.pt')
    (args.output/'config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2),encoding='utf8')

if __name__ == '__main__': main()
