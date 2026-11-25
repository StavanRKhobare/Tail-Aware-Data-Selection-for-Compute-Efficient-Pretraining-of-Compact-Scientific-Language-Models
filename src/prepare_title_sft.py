"""Prepare abstract-to-title examples from the existing train and validation splits."""
import argparse
import json
import re
from pathlib import Path
import pandas as pd
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent


def title_prompt(notes):
    return 'Suggest a concise research title supported by these notes.\nNotes: ' + notes.strip() + '\nTitle:'


def supported_title(title, abstract):
    # Conservative surface checks are a filter, not evidence of entailment.
    numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?', title))
    if not numbers.issubset(set(re.findall(r'\b\d+(?:\.\d+)?%?', abstract))):
        return False
    acronyms = set(re.findall(r'\b[A-Z][A-Z0-9]{2,}\b', title))
    if not acronyms.issubset(set(re.findall(r'\b[A-Z][A-Z0-9]{2,}\b', abstract))):
        return False
    return 3 <= len(title.split()) <= 20 and len(abstract.split()) >= 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data'/'title_sft_v1')
    parser.add_argument('--limit', type=int, default=20000)
    parser.add_argument('--val-limit', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1 or args.val_limit < 1:
        parser.error('limits must be positive')
    tok = Tokenizer.from_file(str(ROOT/'tokenizer'/'tokenizer.json'))
    frames = {split: pd.read_parquet(ROOT/'data'/'processed'/f'{split}.parquet', columns=['id','title','abstract']).sort_values('id').sample(frac=1,random_state=args.seed).head(limit) for split,limit in [('train',args.limit),('val',args.val_limit)]}
    if set(frames['train'].id) & set(frames['val'].id):
        raise ValueError('Source split overlap')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'seed':args.seed,'filtered':0,'too_long':0,'filter_limit':'Surface checks do not establish semantic support. Manual review required.'}
    for split, frame in frames.items():
        count=0
        with (args.output/f'{split}.jsonl').open('x',encoding='utf8') as out:
            for row in frame.itertuples(index=False):
                if not isinstance(row.title,str) or not isinstance(row.abstract,str) or not supported_title(row.title,row.abstract):
                    report['filtered'] += 1
                    continue
                prefix=title_prompt(row.abstract)
                target=' '+row.title.strip()
                if len(tok.encode(prefix).ids)+len(tok.encode(target).ids)+1 > 512:
                    report['too_long'] += 1
                    continue
                out.write(json.dumps({'id':str(row.id),'prompt':prefix,'target':target,'source_split':split},ensure_ascii=False)+'\n')
                count+=1
        report[split]=count
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
