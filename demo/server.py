import csv
import json
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
app = FastAPI(title='TailLab')

DRAFT_ENGINES = {
    'model': ('17M Transformer (V3)', 'runs/draft_sft_v3/ckpt_best.pt'),
    'model_17m_rewrite': ('17M Transformer (draft pilot)', 'runs/draft_pilot_17m/ckpt_best.pt'),
    'model_34m': ('34M Transformer (draft pilot)', 'runs/draft_rewrite_34m/ckpt_best.pt'),
    'smollm2_random': ('SmolLM2 random (draft pilot)', 'runs/draft_smollm2/random/ckpt_best.pt'),
    'smollm2_tail': ('SmolLM2 tail-aware (draft pilot)', 'runs/draft_smollm2/tail_aware/ckpt_best.pt'),
    'transfer': ('Transfer learning', 'runs/draft_transfer'),
}


def engine_status(engine):
    _, location = DRAFT_ENGINES[engine]
    path = ROOT / location
    if engine == 'transfer':
        has_configs = all((path / name).is_file() for name in ('config.json', 'tokenizer_config.json'))
        if not has_configs or not any(path.glob('model*.safetensors')):
            return 'Waiting for the fine-tuned model and tokenizer.'
        if find_spec('transformers') is None:
            return 'Install Transformers to use the transfer checkpoint.'
    elif not path.is_file():
        return 'Waiting for the drafting checkpoint.'
    if engine.startswith('smollm2_'):
        base = ROOT / 'runs' / 'smollm2_base'
        if not all((base / name).is_file() for name in ('config.json', 'tokenizer.json', 'model.safetensors')):
            return 'Local SmolLM2 base files are unavailable.'
        if find_spec('transformers') is None:
            return 'Install Transformers to use this checkpoint.'
    return ''


class DraftRequest(BaseModel):
    title: str = Field(default='', max_length=2000)
    problem: str = Field(default='', max_length=10000)
    method: str = Field(default='', max_length=10000)
    results: str = Field(default='', max_length=10000)
    limitations: str = Field(default='', max_length=10000)
    mode: Literal['auto', 'formatter', 'model', 'model_17m_rewrite', 'model_34m', 'transfer', 'smollm2_random', 'smollm2_tail'] = 'auto'
    title_mode: Literal['preserve', 'suggest'] = 'preserve'
    lock_results: bool = True
    polish: bool = False
    fidelity_mode: Literal['source', 'compose', 'review'] = 'source'
    max_tokens: int = Field(default=220, ge=32, le=400)


@app.get('/api/draft/status')
def draft_status():
    engines = {}
    for engine, (label, _) in DRAFT_ENGINES.items():
        reason = engine_status(engine)
        engines[engine] = {'label': label, 'available': not reason, 'reason': reason}
    return {'model_available': engines['model']['available'], 'checkpoint': 'draft_sft_v3',
            'engines': engines,
            'capability': 'Structured drafting with extractive training targets',
            'review': 'Automatic checks are incomplete. Review claims against your notes.'}


@app.post('/api/draft')
def draft(request: DraftRequest):
    from draft_format import structured_prompt
    from draft_review import compose_abstract, edit_abstract, has_repetition, mostly_copies_notes, review_notes
    notes = {key: getattr(request, key).strip() for key in ('problem', 'method', 'results', 'limitations')}
    text = ' '.join(value for value in notes.values() if value)
    if not text:
        raise HTTPException(status_code=400, detail='Enter at least one research note')
    title = request.title.strip()
    method = 'formatter'
    notice = 'Formatter selected.'
    raw = text
    stop_reason = None
    engine = 'model' if request.mode == 'auto' else request.mode
    using_model = False
    if engine in DRAFT_ENGINES and request.fidelity_mode == 'review':
        label, location = DRAFT_ENGINES[engine]
        unavailable = engine_status(engine)
        if unavailable and request.mode != 'auto':
            raise HTTPException(status_code=503, detail=f'{label}: {unavailable}')
        using_model = not unavailable
        if unavailable:
            notice = 'Drafting checkpoint unavailable. Formatter used.'
    elif request.fidelity_mode == 'compose':
        method = 'rule-based composition'
        notice = 'Composed an abstract from your notes using editing rules.'
    elif request.mode != 'formatter':
        notice = 'Abstract assembled from supplied sentences.'
    if using_model:
        try:
            prefix = structured_prompt(title if request.title_mode == 'preserve' else '', **notes)
            if engine == 'transfer':
                from demo.transfer import generate_transfer
                continuation, stop_reason = generate_transfer(ROOT / location, prefix, request.max_tokens)
            elif engine.startswith('smollm2_'):
                from demo.transfer import generate_smollm2
                continuation, stop_reason = generate_smollm2(ROOT / 'runs' / 'smollm2_base', ROOT / location, prefix, request.max_tokens)
            else:
                from generate import generate_from_checkpoint
                from tokenizers import Tokenizer
                tokenizer = Tokenizer.from_file(str(ROOT / 'tokenizer' / 'tokenizer.json'))
                if len(tokenizer.encode(prefix).ids) > 511:
                    raise HTTPException(status_code=400, detail='Notes exceed the 512-token context. Shorten them before generating.')
                result = generate_from_checkpoint(ROOT / location, prefix, mode='continue', max_new_tokens=request.max_tokens, decoding='greedy', return_result=True)
                continuation, stop_reason = result.continuation_text, result.stop_reason
            proposed_title, separator, raw = continuation.partition('Abstract:')
            if not separator or not proposed_title.strip() or not raw.strip():
                return JSONResponse(status_code=502, content={
                    'detail': 'Model produced an incomplete draft. Select source wording or formatter mode.',
                    'raw_output': continuation,
                    'stop_reason': stop_reason,
                })
            if request.title_mode == 'suggest' or not title:
                title = proposed_title.strip()
            method = 'adapted checkpoint'
            notice = f'Generated with {label}. Review every scientific claim.'
        except ImportError as exc:
            raise HTTPException(status_code=503, detail='Model dependencies are unavailable.') from exc
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=f'Model inference failed: {exc}') from exc
    model_raw = raw.strip()
    if request.title_mode == 'suggest' and (not using_model or engine == 'model'):
        from prepare_title_sft import title_prompt
        title_checkpoint = ROOT / 'runs' / 'title_sft_v1' / 'ckpt_best.pt'
        title = ''
        if title_checkpoint.is_file():
            from generate import generate_from_checkpoint
            candidate = generate_from_checkpoint(title_checkpoint, title_prompt(text), max_new_tokens=48, decoding='greedy', return_result=True)
            suggestion = candidate.continuation_text.strip().split('\n')[0].strip().rstrip('.')
            import re
            if candidate.prompt_truncated or candidate.stop_reason != 'eos' or has_repetition(suggestion) or len(suggestion.split()) < 3 or re.search(r'\b(?:and|or|the|a|an|of|for|with|that|are|is|to|in|on)$', suggestion, re.I):
                notice += ' Title suggestion was incomplete; please enter a title.'
            else:
                title = suggestion
                notice += ' Model title suggestion requires review.'
        else:
            notice += ' Title model unavailable; please enter a title.'
    if request.lock_results and notes['results'] and request.fidelity_mode == 'review' and method == 'adapted checkpoint':
        method = 'source wording (results locked, composed)'
        notice = f'Results locked. The editable abstract uses composition rules. The {label} output is shown separately.'
    composed = request.fidelity_mode == 'compose' or method == 'source wording (results locked, composed)'
    abstract = compose_abstract(notes) if composed else edit_abstract(raw, request.polish)
    warnings = review_notes(notes, abstract)
    if using_model and mostly_copies_notes(notes, model_raw):
        warnings.append('The model repeated most of the wording in your notes.')
    if has_repetition(model_raw) or has_repetition(title):
        warnings.append('Model output contains repetition. Inspect the original output or use source wording.')
    if stop_reason == 'max_new_tokens':
        warnings.append('Output reached the token limit and may be incomplete.')
    return {'title': title, 'abstract': abstract, 'raw_abstract': model_raw,
            'abstract_origin': 'Rule-based composition' if composed else label if using_model else 'Supplied sentences',
            'model_generated': using_model,
            'engine': engine if using_model else 'formatter',
            'method': method, 'notice': notice, 'warnings': warnings,
            'editing': 'Rule-based composition and sentence deduplication' if composed else 'Sentence deduplication' + (' and rule-based prose edits' if request.polish else ''),
            'word_count': len(abstract.split()), 'stop_reason': stop_reason}

class GenerationRequest(BaseModel):
    strategy: Literal['random', 'quality', 'diversity', 'tail_aware', 'tail_only']
    checkpoint_seed: int = Field(ge=42, le=46)
    mode: Literal['continue', 'document']
    prompt: str = Field(default='', max_length=10000)
    decoding: Literal['sample', 'greedy'] = 'sample'
    max_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=.8, gt=0, le=3)
    top_k: int = Field(default=50, ge=0, le=16384)
    top_p: float = Field(default=.95, gt=0, le=1)
    sample_seed: int = 42

@app.get('/')
def index():
    return FileResponse(ROOT / 'demo' / 'index.html')

@app.get('/api/results')
def results():
    files = {'runs': '5seed_experimental_results_runs.csv', 'summary': '5seed_strategy_summary.csv', 'groups': '5seed_experimental_results_groups.csv', 'selection': 'selection_audit_summary.csv'}
    try:
        data = {}
        for key, name in files.items():
            with (ROOT / 'reports' / name).open(encoding='utf-8', newline='') as f:
                data[key] = list(csv.DictReader(f))
        return data
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post('/api/generate')
def generate(request: GenerationRequest):
    checkpoint = ROOT / 'runs' / request.strategy / f'seed{request.checkpoint_seed}' / 'ckpt_best.pt'
    if not checkpoint.is_file():
        raise HTTPException(status_code=404, detail='Checkpoint is unavailable')
    if request.mode == 'continue' and not request.prompt.strip():
        raise HTTPException(status_code=400, detail='Enter a prompt to continue')
    try:
        from generate import generate_from_checkpoint
        result = generate_from_checkpoint(checkpoint, request.prompt if request.mode == 'continue' else '', mode=request.mode, max_new_tokens=request.max_tokens, temperature=request.temperature, decoding=request.decoding, top_k=request.top_k, top_p=request.top_p, seed=request.sample_seed, return_result=True)
        return result.to_dict()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/api/comparisons')
def comparisons():
    paths = {
        'scratch': 'reports/val_loss_by_group.json',
        'smollm2': 'reports/smollm2_validation_comparison.json',
        'drafts': 'reports/phase1_drafting_comparison.json',
        'scratch_34m': 'runs/phase1_34m/random/seed42/summary.json',
    }
    data = {}
    for key, path in paths.items():
        try:
            data[key] = json.loads((ROOT / path).read_text(encoding='utf-8'))
        except FileNotFoundError:
            data[key] = None
    data['scratch'] = [
        row for row in (data['scratch'] or [])
        if row['seed'] == 42 and row['model'] in ('random', 'tail_aware')
    ]
    if data['scratch_34m']:
        data['scratch_34m'] = {
            key: data['scratch_34m'][key] for key in ('tokens_seen', 'final_val_loss')
        }
    return data


@app.get('/draft.js')
def draft_script():
    return FileResponse(ROOT / 'demo' / 'draft.js', media_type='text/javascript')


@app.get('/compare.js')
def comparison_script():
    return FileResponse(ROOT / 'demo' / 'compare.js', media_type='text/javascript')
