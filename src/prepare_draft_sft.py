"""Prepare conservative copy-based drafting examples from the existing train split."""
import argparse, hashlib, json, random, re
from pathlib import Path
import pandas as pd
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent

def prompt(title, notes):
    return f"Working title: {title}\nResearch notes: {notes}\nTitle:"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'data'/'draft_sft_v2')
    p.add_argument('--limit', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tok = Tokenizer.from_file(str(ROOT/'tokenizer'/'tokenizer.json'))
    frames = {split: pd.read_parquet(ROOT/'data'/'processed'/f'{split}.parquet', columns=['id','title','abstract']).sort_values('id').sample(frac=1, random_state=args.seed).head(args.limit if split == 'train' else min(args.limit, 200)) for split in ('train', 'val')}
    if set(frames['train'].id) & set(frames['val'].id):
        raise ValueError('Paper IDs overlap across source splits')
    counts = {'train': 0, 'val': 0, 'too_long': 0, 'invalid': 0}
    files = {s: (args.output/f'{s}.jsonl').open('x', encoding='utf8') for s in ('train','val')}
    lengths = []
    for split, df in frames.items():
      for row in df.itertuples(index=False):
        title, abstract = str(row.title).strip(), str(row.abstract).strip()
        sentences = re.split(r'(?<=[.!?])\s+', abstract)
        if not title or len(sentences) < 2:
            counts['invalid'] += 1; continue
        # Every target sentence and the title are present verbatim in the input.
        # This is a conservative reconstruction task, not natural notes.
        rng = random.Random(int(hashlib.sha256((str(args.seed)+str(row.id)).encode()).hexdigest(),16))
        rng.shuffle(sentences)
        # Synthetic notes preserve every sentence, including numbers and negation.
        # Lexical cues are limited to discourse openings to avoid deleting facts.
        notes = [re.sub(r'^(?:In this paper,? |In this work,? |Here,? )', '', sentence, flags=re.I) for sentence in sentences]
        working_title = title if rng.random() < .5 else ''
        if not working_title:
            # Supply title concepts explicitly so the target title remains supported.
            notes.append('Title concepts: ' + title)
        prefix = prompt(working_title, '\n- ' + '\n- '.join(notes))
        target = f' {title}\nAbstract: {abstract}'
        a, b = tok.encode(prefix).ids, tok.encode(target).ids
        if len(a)+len(b)+1 > 512:
            counts['too_long'] += 1; continue
        files[split].write(json.dumps({'id':str(row.id),'prompt':prefix,'target':target, 'source_split':split, 'synthetic':True},ensure_ascii=False)+'\n')
        counts[split] += 1; lengths.append(len(a)+len(b)+1)
    for f in files.values(): f.close()
    counts['max_tokens'] = max(lengths, default=0)
    counts['mean_tokens'] = sum(lengths)/len(lengths) if lengths else 0
    (args.output/'report.json').write_text(json.dumps(counts,indent=2), encoding='utf8')
    print(json.dumps(counts,indent=2))

if __name__ == '__main__': main()
