"""Derive structured, extractive drafting supervision from the existing corpus.

Targets use selected source sentences only. This provides synthetic formatting
supervision and does not constitute human-authored notes-to-prose supervision.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

from draft_format import structured_prompt

ROOT = Path(__file__).resolve().parent.parent
RESULT = re.compile(r'\b(results?|outperforms?|accuracy|achieves?|improves?|experiments? (?:show|demonstrate)|evaluat(?:e|ion))\b', re.I)
METHOD = re.compile(r'\b(propos(?:e|es)|introduc(?:e|es)|develop|method|approach|framework|algorithm)\b', re.I)
LIMIT = re.compile(r'\b(limitation|limited to|future work)\b', re.I)


def make_example(paper_id, title, abstract, variant):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', abstract.strip())
    if len(sentences) < 2:
        return None
    notes = dict.fromkeys(('problem', 'method', 'results', 'limitations'), '')
    for sentence in sentences:
        field = 'limitations' if LIMIT.search(sentence) else 'results' if RESULT.search(sentence) else 'method' if METHOD.search(sentence) else 'problem'
        if not notes[field]:
            notes[field] = sentence
    # A source sentence is used once even when its role is classified imperfectly.
    if variant == 'missing_results':
        notes['results'] = ''
    selected = [notes[field] for field in notes if notes[field]]
    if not selected:
        return None
    working_title = title.strip() if variant == 'working_title' else ''
    # Without a title, use words from supplied facts rather than a hidden gold title.
    output_title = working_title or ' '.join(re.findall(r"[\w-]+", selected[0])[:12])
    return {'id': str(paper_id), 'variant': variant, 'notes': notes,
            'working_title': working_title,
            'prompt': structured_prompt(working_title, **notes),
            'target': ' ' + output_title + '\nAbstract: ' + ' '.join(selected),
            'synthetic': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data'/'draft_sft_v3')
    parser.add_argument('--limit', type=int, default=5000)
    parser.add_argument('--val-limit', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1 or args.val_limit < 1:
        parser.error('limits must be positive')
    tokenizer = Tokenizer.from_file(str(ROOT/'tokenizer'/'tokenizer.json'))
    frames = {split: pd.read_parquet(ROOT/'data'/'processed'/f'{split}.parquet', columns=['id','title','abstract']).sort_values('id').sample(frac=1, random_state=args.seed).head(limit)
              for split, limit in [('train', args.limit), ('val', args.val_limit)]}
    if set(frames['train'].id) & set(frames['val'].id):
        raise ValueError('Source paper IDs overlap')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'seed': args.seed, 'variants': {}, 'too_long': 0, 'invalid': 0,
              'limitation': 'Synthetic extractive supervision. Role classification is heuristic and titles without working titles are extractive.'}
    for split, frame in frames.items():
        count = 0
        with (args.output/f'{split}.jsonl').open('x', encoding='utf8') as handle:
            for row in frame.itertuples(index=False):
                if not isinstance(row.title, str) or not isinstance(row.abstract, str):
                    report['invalid'] += 1
                    continue
                for variant in ('working_title', 'no_title', 'missing_results'):
                    example = make_example(row.id, row.title, row.abstract, variant)
                    if example is None:
                        report['invalid'] += 1
                        continue
                    length = len(tokenizer.encode(example['prompt']).ids) + len(tokenizer.encode(example['target']).ids) + 1
                    if length > 512:
                        report['too_long'] += 1
                        continue
                    example.update(source_split=split, tokens=length)
                    handle.write(json.dumps(example, ensure_ascii=False)+'\n')
                    count += 1
                    report['variants'][variant] = report['variants'].get(variant, 0) + 1
        report[split] = count
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
