"""Generate outputs for authored drafting cases. Review factual fidelity manually."""
import argparse
import json
import sys
from pathlib import Path
from draft_format import structured_prompt
from generate import generate_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=Path('data/draft_sft_v2/evaluation_cases.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding='utf8'))
    with args.output.open('x', encoding='utf8') as handle:
        for case in cases:
            notes = {field: case.get(field, '') for field in ('problem','method','results','limitations')}
            result = generate_from_checkpoint(args.checkpoint, structured_prompt(case.get('title',''), **notes), max_new_tokens=220, decoding='greedy', return_result=True)
            title, separator, abstract = result.continuation_text.partition('Abstract:')
            row = {'id':case['id'], 'checkpoint':str(args.checkpoint), 'input':case,
                   'output':result.continuation_text, 'format_valid':bool(separator and title.strip() and abstract.strip()),
                   'stop_reason':result.stop_reason, 'prompt_truncated':result.prompt_truncated,
                   'review_required':True}
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
