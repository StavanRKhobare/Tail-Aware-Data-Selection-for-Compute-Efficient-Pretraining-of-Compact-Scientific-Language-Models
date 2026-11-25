"""Build notes-to-original-abstract supervision from the existing paper corpus.

Synthetic rough notes retain all sentences but remove articles and discourse
openings. Source abstracts supply natural-prose targets. Semantic review remains
necessary. This differs from the extractive V3 targets.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
import pandas as pd
from tokenizers import Tokenizer
from draft_format import structured_prompt

ROOT=Path(__file__).resolve().parent.parent


def roughen(sentence):
    sentence=re.sub(r'^(?:In this (?:paper|work|study),?\s*|This paper\s+|This work\s+)', '', sentence, flags=re.I)
    # Remove only articles, retaining subjects, predicates, negation and quantities.
    sentence=re.sub(r'\b(?:a|an|the)\s+', '', sentence, flags=re.I)
    return sentence.strip().rstrip('.')


def build_pair(paper_id, title, abstract, split, seed):
    sentences=re.split(r'(?<=[.!?])\s+(?=[A-Z])', abstract.strip())
    if len(sentences)<2:
        return None
    rng=random.Random(int(hashlib.sha256(f'{seed}:{paper_id}'.encode()).hexdigest(),16))
    facts=[roughen(s) for s in sentences]
    rng.shuffle(facts)
    # Keep all facts in one field, avoiding unreliable sentence-role labelling.
    notes={'problem':'', 'method':'Facts to organize:\n- '+'\n- '.join(facts), 'results':'', 'limitations':''}
    return {'id':str(paper_id),'source_split':split,'notes':notes,
            'prompt':structured_prompt(title,**notes),
            'target':' '+title.strip()+'\nAbstract: '+abstract.strip(),
            'source_abstract':abstract, 'synthetic':True,
            'construction':'Article/discourse removal and sentence shuffle; original prose target'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/draft_rewrite_v1')
    parser.add_argument('--limit',type=int,default=12000)
    parser.add_argument('--val-limit',type=int,default=300)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if min(args.limit,args.val_limit)<1:
        parser.error('Limits must be positive')
    tok=Tokenizer.from_file(str(ROOT/'tokenizer/tokenizer.json'))
    frames={split:pd.read_parquet(ROOT/'data/processed'/f'{split}.parquet',columns=['id','title','abstract']).sort_values('id').sample(frac=1,random_state=args.seed).head(limit) for split,limit in [('train',args.limit),('val',args.val_limit)]}
    if set(frames['train'].id)&set(frames['val'].id):
        raise ValueError('Paper split overlap')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'seed':args.seed,'too_long':0,'invalid':0,'duplicate_text':0,
            'construction':'Synthetic notes from existing papers; original abstracts as prose targets',
            'limitation':'Not human-authored notes. No generalization or factual-fidelity guarantee.'}
    # Process validation first, excluding exact duplicate training abstracts.
    seen=set()
    for split in ('val','train'):
        rows=[]
        for row in frames[split].itertuples(index=False):
            if not isinstance(row.title,str) or not isinstance(row.abstract,str):
                report['invalid']+=1
                continue
            key=' '.join(row.abstract.lower().split())
            if key in seen:
                report['duplicate_text']+=1
                continue
            example=build_pair(row.id,row.title,row.abstract,split,args.seed)
            if example is None:
                report['invalid']+=1
                continue
            tokens=len(tok.encode(example['prompt']).ids)+len(tok.encode(example['target']).ids)+1
            if tokens>512:
                report['too_long']+=1
                continue
            seen.add(key)
            example['tokens']=tokens
            rows.append(example)
        if not rows:
            raise ValueError('Empty output split')
        (args.output/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
        report[split]=len(rows)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
