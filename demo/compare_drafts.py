"""Exercise the draft API on the same held-out inputs for every engine."""
import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from demo.server import DRAFT_ENGINES, ROOT, app, engine_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((ROOT / 'data/draft_sft_v2/evaluation_cases.json').read_text(encoding='utf-8'))
    cases.append({
        'id': 'research_notes',
        'problem': 'Compact scientific language models perform poorly on underrepresented topics.',
        'method': 'Compare random and tail-aware data selection. Train a 17.1-million-parameter Transformer on 60 million tokens from scientific paper titles and abstracts across five seeds.',
        'results': 'Tail-aware selection improved rare-topic validation loss across all five seeds. Random selection achieved lower overall validation loss.',
        'limitations': 'One corpus, one model size, one training budget.',
        'checks': ['Preserve the opposing results, training quantities, and scope limitation.'],
    })
    engines = {'formatter': 'Formatter', **{
        key: label for key, (label, _) in DRAFT_ENGINES.items() if key != 'transfer' and not engine_status(key)
    }}
    rows = []
    with TestClient(app) as client:
        for case in cases:
            outputs = {}
            for engine in engines:
                payload = {**case, 'mode': engine, 'lock_results': False, 'fidelity_mode': 'review'}
                response = client.post('/api/draft', json=payload)
                outputs[engine] = {'status': response.status_code, **response.json()}
            rows.append({'input': case, 'outputs': outputs})
            print(f"Evaluated {case['id']} across {len(engines)} engines", flush=True)

    with args.output.open('x', encoding='utf-8') as handle:
        json.dump({'engines': engines, 'cases': rows}, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
