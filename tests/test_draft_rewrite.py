from prepare_draft_rewrite import roughen, build_pair


def test_rough_notes_preserve_attribution_and_negation():
    sentence='Tail-aware selection did not improve accuracy by 3%.'
    assert roughen(sentence)=='Tail-aware selection did not improve accuracy by 3%'


def test_original_prose_is_target_not_note_concatenation():
    row=build_pair('id','Study title','The approach improves accuracy. Random selection did not improve loss.','train',42)
    assert 'The approach improves accuracy.' in row['target']
    assert 'The approach' not in row['prompt']
    assert 'Random selection did not improve loss' in row['prompt']
    assert row['synthetic']
