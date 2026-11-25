import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from demo import server
from demo.server import app

ROOT = Path(__file__).resolve().parents[1]

def test_formatter_status_and_negation():
    response = TestClient(app).post('/api/draft', json={'mode':'formatter','results':'No improvement in 81.2% of cases.'})
    assert response.status_code == 200
    assert response.json()['abstract'] == 'No improvement in 81.2% of cases.'
    assert response.json()['notice'] == 'Formatter selected.'

def test_empty_notes_rejected():
    assert TestClient(app).post('/api/draft', json={'title':'Title only'}).status_code == 400


@pytest.mark.parametrize('engine', ['model_34m', 'transfer', 'smollm2_random', 'smollm2_tail'])
def test_experimental_engine_requires_its_own_checkpoint(engine, monkeypatch, tmp_path):
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    client = TestClient(app)
    status = client.get('/api/draft/status').json()['engines'][engine]
    assert not status['available']
    assert status['reason']
    request = {'mode': engine, 'problem': 'Compare two classifiers.'}
    assert client.post('/api/draft', json={**request, 'fidelity_mode': 'review'}).status_code == 503
    assert client.post('/api/draft', json=request).json()['engine'] == 'formatter'


@pytest.mark.parametrize('engine', ['model_34m', 'transfer'])
def test_experimental_engine_uses_selected_checkpoint(engine, monkeypatch):
    import generate
    from demo import transfer

    continuation = 'Comparison of Two Classifiers\nAbstract: We compared two classifiers.'
    calls = []

    def generate_local(path, prompt, *args, **kwargs):
        calls.append((path, prompt))
        if engine == 'transfer':
            return continuation, 'eos'
        return SimpleNamespace(continuation_text=continuation, stop_reason='eos')

    monkeypatch.setattr(server, 'engine_status', lambda engine: '')
    monkeypatch.setattr(generate, 'generate_from_checkpoint', generate_local)
    monkeypatch.setattr(transfer, 'generate_transfer', generate_local)
    response = TestClient(app).post('/api/draft', json={
        'mode': engine, 'fidelity_mode': 'review', 'title_mode': 'suggest',
        'problem': 'Compare two classifiers.',
    })
    assert response.status_code == 200
    [(checkpoint, prompt)] = calls
    assert checkpoint == ROOT / server.DRAFT_ENGINES[engine][1]
    assert 'Compare two classifiers.' in prompt
    assert response.json()['engine'] == engine
    assert response.json()['title'] == 'Comparison of Two Classifiers'
    assert response.json()['abstract'] == 'We compared two classifiers.'


@pytest.mark.parametrize('engine', ['smollm2_random', 'smollm2_tail'])
def test_smollm2_routes_to_selected_drafting_checkpoint(engine, monkeypatch):
    from demo import transfer

    calls = []

    def generate_local(base, checkpoint, prompt, max_new_tokens):
        calls.append((base, checkpoint))
        return 'Classifier comparison\nAbstract: We compared two classifiers.', 'eos'

    monkeypatch.setattr(server, 'engine_status', lambda engine: '')
    monkeypatch.setattr(transfer, 'generate_smollm2', generate_local)
    response = TestClient(app).post('/api/draft', json={
        'mode': engine, 'fidelity_mode': 'review', 'problem': 'Compare two classifiers.',
    })
    assert response.status_code == 200
    assert response.json()['engine'] == engine
    assert calls == [(ROOT / 'runs/smollm2_base', ROOT / server.DRAFT_ENGINES[engine][1])]


def test_comparison_endpoint_handles_pending_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    reports = tmp_path / 'reports'
    reports.mkdir()
    rows = [
        {'model': 'random', 'seed': 42}, {'model': 'tail_aware', 'seed': 42},
        {'model': 'random', 'seed': 43}, {'model': 'tail_only', 'seed': 42},
    ]
    (reports / 'val_loss_by_group.json').write_text(json.dumps(rows), encoding='utf-8')
    response = TestClient(app).get('/api/comparisons')
    assert response.status_code == 200
    assert response.json() == {'scratch': rows[:2], 'smollm2': None, 'drafts': None, 'scratch_34m': None}


def test_invalid_smollm2_format_exposes_raw_output_for_comparison(monkeypatch):
    from demo import transfer

    monkeypatch.setattr(server, 'engine_status', lambda engine: '')
    monkeypatch.setattr(transfer, 'generate_smollm2', lambda *args: ('An unfinished title', 'eos'))
    response = TestClient(app).post('/api/draft', json={
        'mode': 'smollm2_tail', 'fidelity_mode': 'review', 'problem': 'Evaluate two models.',
    })
    assert response.status_code == 502
    assert response.json()['raw_output'] == 'An unfinished title'


def test_prepared_validation_uses_original_split():
    folder = ROOT / 'data' / 'draft_sft_v2'
    if not all((folder / name).is_file() for name in ('train.jsonl', 'val.jsonl')):
        pytest.skip('Local generated V2 drafting data is unavailable')
    train = [json.loads(line) for line in (folder/'train.jsonl').read_text(encoding='utf8').splitlines()]
    val = [json.loads(line) for line in (folder/'val.jsonl').read_text(encoding='utf8').splitlines()]
    assert train and val
    assert {r['id'] for r in train}.isdisjoint(r['id'] for r in val)
    assert all(r['source_split'] == 'val' for r in val)
    assert any('Working title: \n' in r['prompt'] for r in train)
