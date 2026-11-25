"""Produce reviewable title predictions on held-out abstract-title pairs."""
import argparse
import json
from pathlib import Path
from generate import generate_from_checkpoint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--data', type=Path, default=Path('data/title_sft_v1/val.jsonl'))
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--limit', type=int, default=30)
args = parser.parse_args()
rows = [json.loads(line) for line in args.data.read_text(encoding='utf8').splitlines()][:args.limit]
with args.output.open('x', encoding='utf8') as out:
    for row in rows:
        result = generate_from_checkpoint(args.checkpoint,row['prompt'],max_new_tokens=48,decoding='greedy',return_result=True)
        out.write(json.dumps({'id':row['id'],'input':row['prompt'],'reference':row['target'].strip(),'prediction':result.continuation_text,'stop':result.stop_reason,'review_required':True},ensure_ascii=False)+'\n')
print(f'Wrote {len(rows)} examples for human review to {args.output}')
