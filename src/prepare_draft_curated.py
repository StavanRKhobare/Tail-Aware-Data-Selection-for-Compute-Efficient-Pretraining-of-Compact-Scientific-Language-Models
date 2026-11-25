"""Validate and build assistant-curated rough-notes-to-abstract examples."""
import argparse
import json
import re
from pathlib import Path
import pandas as pd
from tokenizers import Tokenizer
from draft_format import structured_prompt

ROOT = Path(__file__).resolve().parent.parent


def validate_record(record, tokenizer):
    required = {'id','source_split','title','notes','abstract','authorship'}
    if required - record.keys() or record['source_split'] not in ('train','val'):
        raise ValueError('Missing fields or invalid split')
    if not record['notes'] or not record['abstract'].strip():
        raise ValueError('Empty notes or abstract')
    prefix = structured_prompt(record['title'], **record['notes'])
    target = ' ' + record['title'] + '\nAbstract: ' + record['abstract']
    if len(tokenizer.encode(prefix).ids) + len(tokenizer.encode(target).ids) + 1 > 512:
        raise ValueError(f"Context exceeded for {record['id']}")
    notes = ' '.join(record['notes'].values())
    # Check numbers in abstract only; the supplied working title is preserved.
    added = set(re.findall(r'\b\d+(?:\.\d+)?%?',record['abstract'])) - set(re.findall(r'\b\d+(?:\.\d+)?%?',notes))
    if added:
        raise ValueError(f'Unsupported numerical values: {added}')
    return prefix,target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,default=ROOT/'data/draft_curated_v1/annotations.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/draft_curated_v1/prepared')
    args=parser.parse_args()
    records=json.loads(args.annotations.read_text(encoding='utf8'))
    tokenizer=Tokenizer.from_file(str(ROOT/'tokenizer/tokenizer.json'))
    sources={split:pd.read_parquet(ROOT/'data/processed'/f'{split}.parquet',columns=['id','title','abstract']).set_index('id') for split in ('train','val')}
    ids=set()
    prepared={'train':[],'val':[]}
    for record in records:
        if record['id'] in ids:
            raise ValueError('Duplicate paper ID')
        ids.add(record['id'])
        source=sources[record['source_split']]
        if record['id'] not in source.index:
            raise ValueError('Paper is absent from declared source split')
        if record['title'] != source.loc[record['id'],'title']:
            raise ValueError('Working title differs from recorded paper title')
        prefix,target=validate_record(record,tokenizer)
        prepared[record['source_split']].append(dict(record,prompt=prefix,target=target,source_abstract=source.loc[record['id'],'abstract']))
    if not all(prepared.values()):
        raise ValueError('Both splits must contain examples')
    args.output.mkdir(parents=True,exist_ok=False)
    for split, rows in prepared.items():
        (args.output/f'{split}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf8')
    report={'train':len(prepared['train']),'val':len(prepared['val']),'limitation':'Tiny assistant-authored seed set. Source papers and wording require user review. Suitable for pipeline checks, not generalization claims.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(report)


if __name__=='__main__':
    main()
