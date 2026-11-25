from draft_format import structured_prompt
from prepare_draft_structured import make_example


def test_structured_prompt_keeps_missing_results_empty():
    text = structured_prompt('', problem='Problem.', method='Compare methods.')
    assert 'Results: \n' in text
    assert text.endswith('\nTitle:')


def test_missing_results_target_excludes_source_result():
    example = make_example('paper', 'Hidden answer title',
                           'Rare topics need study. We propose a sampler. Results show 81.2% accuracy.',
                           'missing_results')
    assert '81.2%' not in example['target']
    assert '81.2%' not in example['prompt']
    assert 'Hidden answer title' not in example['target']


def test_numbers_and_negative_result_preserved():
    example = make_example('paper', 'Comparison',
                           'We propose a model trained on 800 examples. Results did not improve on 200 validation examples.',
                           'working_title')
    assert '800' in example['prompt'] and '800' in example['target']
    assert 'did not improve on 200' in example['prompt']
    assert 'did not improve on 200' in example['target']
