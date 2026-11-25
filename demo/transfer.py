"""Inference for a locally exported Phase 1 text-to-text drafting model."""

from fastapi import HTTPException


def generate_smollm2(base, checkpoint, prompt, max_new_tokens):
    import torch
    from smollm2 import CausalLM, base_metadata
    from tokenizers import Tokenizer
    from train import load_checkpoint

    state = load_checkpoint(checkpoint)
    if not state.get('draft_sft') or any(state['config'].get(key) != value for key, value in base_metadata(base).items()):
        raise ValueError('Expected a SmolLM2 drafting checkpoint paired with this base.')
    tokenizer = Tokenizer.from_file(str(base / 'tokenizer.json'))
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if len(ids) > 511:
        raise HTTPException(status_code=400, detail='Notes exceed the 512-token input limit. Shorten them before generating.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CausalLM(base)
    model.load_state_dict(state['model'])
    del state
    model.to(device).eval()
    eos = model.model.config.eos_token_id
    inputs = torch.tensor([ids], device=device)
    with torch.inference_mode():
        generated = model.model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, eos_token_id=eos, pad_token_id=eos,
        )[0, len(ids):].tolist()
    stop = 'eos' if generated and generated[-1] == eos else 'max_new_tokens'
    return tokenizer.decode(generated, skip_special_tokens=True), stop


def generate_transfer(checkpoint, prompt, max_new_tokens):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    inputs = tokenizer(prompt, return_tensors='pt', verbose=False)
    if inputs['input_ids'].shape[-1] > min(512, tokenizer.model_max_length):
        raise HTTPException(status_code=400, detail='Notes exceed the transfer model input limit. Shorten them before generating.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint, local_files_only=True, use_safetensors=True,
    ).to(device).eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs.to(device), max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        )[0]
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    stop_reason = 'eos' if output[-1].item() in eos_ids else 'max_new_tokens'
    return tokenizer.decode(output, skip_special_tokens=True), stop_reason
