from fastapi.testclient import TestClient
from demo.server import app
from draft_review import edit_abstract, review_notes


def test_editing_preserves_negative_findings_and_numbers():
    assert edit_abstract('Use 800 examples. Results did not improve. Use 800 examples.', True) == 'The study uses 800 examples. Results did not improve.'


def test_numerical_review():
    warnings = review_notes({'results':'Accuracy was 81.2%.'}, 'Accuracy was 90%.')
    assert any('absent' in warning for warning in warnings)
    assert any('omitted' in warning for warning in warnings)


def test_options_and_script():
    client = TestClient(app)
    result = client.post('/api/draft',json={'mode':'formatter','title':'A study','method':'Compare two classifiers.','polish':True}).json()
    assert result['title'] == 'A study'
    assert result['abstract'] == 'The study compares two classifiers.'
    assert result['warnings']
    assert client.get('/draft.js').status_code == 200
    assert 'model_available' in client.get('/api/draft/status').json()


def test_invalid_mode():
    assert TestClient(app).post('/api/draft',json={'mode':'unknown','problem':'Study.'}).status_code == 422

def test_repetition_check():
    from draft_review import has_repetition
    assert has_repetition('five seeds model ' * 8)
    assert has_repetition('Rare categories receive little data. Rare categories receive little data.')
    assert not has_repetition('Results did not improve.')


def test_composition_combines_objective_and_method_without_changing_the_result():
    response = TestClient(app).post('/api/draft', json={
        'mode': 'model', 'fidelity_mode': 'compose',
        'problem': 'Evaluate a classifier.',
        'method': 'Use 800 training and 200 validation examples.',
        'results': 'Validation accuracy did not exceed 81.2%.',
    })
    assert response.status_code == 200
    draft = response.json()
    assert draft['abstract'] == (
        'To evaluate a classifier, the study uses 800 training and 200 validation examples. '
        'Validation accuracy did not exceed 81.2%.'
    )
    assert draft['abstract_origin'] == 'Rule-based composition'
    assert not draft['model_generated']


def test_model_echo_with_locked_results_keeps_raw_output_separate(monkeypatch):
    from demo import server, transfer

    raw = 'Evaluate a classifier. Use 800 training examples. Validation accuracy did not exceed 81.2%.'
    monkeypatch.setattr(server, 'engine_status', lambda engine: '')
    monkeypatch.setattr(transfer, 'generate_smollm2', lambda *args: ('Classifier evaluation\nAbstract: ' + raw, 'eos'))
    response = TestClient(app).post('/api/draft', json={
        'mode': 'smollm2_tail', 'fidelity_mode': 'review',
        'problem': 'Evaluate a classifier.', 'method': 'Use 800 training examples.',
        'results': 'Validation accuracy did not exceed 81.2%.',
    })
    assert response.status_code == 200
    draft = response.json()
    assert draft['abstract'] != raw
    assert draft['raw_abstract'] == raw
    assert draft['abstract_origin'] == 'Rule-based composition'
    assert draft['model_generated']
    assert any('repeated most' in warning for warning in draft['warnings'])
