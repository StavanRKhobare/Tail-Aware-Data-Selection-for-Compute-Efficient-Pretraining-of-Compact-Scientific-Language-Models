import pytest
from fastapi.testclient import TestClient
from demo.server import app


def test_title_preserved_even_when_repetitive():
    title = 'Study of study of study of study of methods'
    result = TestClient(app).post('/api/draft',json={'title':title,'title_mode':'preserve','problem':'Testing.','mode':'formatter'}).json()
    assert result['title'] == title


def test_title_suggestion_unavailable_does_not_return_fragment(monkeypatch):
    from pathlib import Path
    original = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda self: False if 'title_sft_v1' in str(self) else original(self))
    result = TestClient(app).post('/api/draft',json={'title_mode':'suggest','problem':'Small scientific language models may perform poorly on research topics that are underrepresented.','mode':'formatter'}).json()
    assert result['title'] == ''
    assert 'unavailable' in result['notice']


def test_results_locked_on_model_review(monkeypatch):
    import generate
    from types import SimpleNamespace
    from demo import server
    monkeypatch.setattr(server, 'engine_status', lambda engine: '')
    monkeypatch.setattr(generate, 'generate_from_checkpoint', lambda *a, **kw: SimpleNamespace(continuation_text='Test title\nAbstract: Random improved rare-topic loss.',stop_reason='eos'))
    result = TestClient(app).post('/api/draft',json={'title':'Study','mode':'model','fidelity_mode':'review','results':'Tail-aware improved rare-topic loss.','lock_results':True}).json()
    assert result['abstract'] == 'Tail-aware improved rare-topic loss.'
    assert 'source wording' in result['method']
