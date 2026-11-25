import json
from pathlib import Path
import pytest
from tokenizers import Tokenizer
from prepare_draft_curated import validate_record

ROOT=Path(__file__).resolve().parents[1]


def test_curated_rejects_unsupported_number():
    tokenizer=Tokenizer.from_file(str(ROOT/'tokenizer/tokenizer.json'))
    record={'id':'test','source_split':'train','title':'Study','notes':{'method':'compare classifiers'},'abstract':'Accuracy was 99%.','authorship':'test'}
    with pytest.raises(ValueError,match='Unsupported numerical'):
        validate_record(record,tokenizer)


def test_curated_targets_are_rewrites_and_splits_disjoint():
    annotations = ROOT / 'data/draft_curated_v1/annotations.json'
    if not annotations.is_file():
        pytest.skip('Local curated pilot annotations are unavailable')
    records=json.loads(annotations.read_text(encoding='utf-8'))
    train={r['id'] for r in records if r['source_split']=='train'}
    val={r['id'] for r in records if r['source_split']=='val'}
    assert train and val and train.isdisjoint(val)
    for record in records:
        notes=' '.join(record['notes'].values())
        assert record['abstract'] != notes
        assert record['authorship'].startswith('Assistant-authored')
