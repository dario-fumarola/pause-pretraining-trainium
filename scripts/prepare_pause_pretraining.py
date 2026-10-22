"""Prepare bounded qualification inputs; never download or load model weights."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from train.pause_pretraining import EXPERIMENT, PAUSE_ID, arithmetic_program, source_digest, task_window
from train.artifacts import atomic_json, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--whole-instance', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    if PAUSE_ID < len(tok) or tok.convert_ids_to_tokens(PAUSE_ID) is not None:
        raise ValueError('Pause row must be tokenizer-undefined in the fixed vocabulary')
    rows = []
    items = 768 if a.whole_instance else 24
    with a.source.open() as f:
        for index, line in enumerate(f):
            if index == items:
                break
            source = json.loads(line)
            ids = source['ids'][:513]
            row = {'index': index, 'source_document_id': source['document_id'],
                   'kind': 'text', 'ids': ids}
            if index % 4 == 0:
                program = arithmetic_program(2200, 'qualification', index, (4, 8, 12)[index//4 % 3])
                row['ids'], row['task'] = task_window(tok, program, ids)
                row['kind'] = 'task'
            if len(row['ids']) != 513:
                raise ValueError('Every qualification window requires 513 real tokens')
            rows.append(row)
    if len(rows) != items:
        raise ValueError('Require the complete fixed qualification windows')
    suffix = '-whole' if a.whole_instance else ''
    path = a.output/f'qualification{suffix}.json'
    atomic_json(path, rows)
    atomic_json(a.output/f'data-manifest{suffix}.json', {
        'experiment_id': EXPERIMENT, 'purpose': 'hardware_qualification_not_science',
        'source': str(a.source.resolve()), 'source_sha256': sha256_file(a.source),
        'tokenizer': str(a.tokenizer.resolve()),
        'tokenizer_sha256': sha256_file(a.tokenizer/'tokenizer.json'),
        'rows': len(rows), 'ordered_source_digest': source_digest(rows),
        'data_file_sha256': sha256_file(path), 'source_tokens': items*513,
        'predicted_real_tokens': items*512, 'synthetic_windows': items//4,
        'held_out_outcomes_inspected': False})
    print(json.dumps({'rows': len(rows), 'source_digest': source_digest(rows)}))


if __name__ == '__main__':
    main()
