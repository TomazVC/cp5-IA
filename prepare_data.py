"""Gera os conjuntos determinísticos, sem sobrescrever dados modificados."""
import json
from collections import Counter
from support import ROOT, file_hash, make_splits, write_json


def main():
    scenarios = json.loads((ROOT / 'data/scenarios.json').read_text(encoding='utf-8'))
    splits = make_splits(scenarios)
    for split, rows in splits.items():
        path = ROOT / f'data/{split}.jsonl'
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        if path.exists() and path.read_text(encoding='utf-8') != content:
            raise ValueError(f'{path.name} já existe com dados diferentes; revise antes de regenerar.')
        path.write_text(content, encoding='utf-8')
    write_json(ROOT / 'data/manifest.json', {
        'source': 'Sintético autoral para demonstração acadêmica; 80 cenários, 5 redações por cenário.',
        'license': 'CC0-1.0', 'split_rule': 'Famílias 0-9 treino, 10-12 validação, 13-15 teste por categoria.',
        'counts': {key: dict(Counter(row['category'] for row in rows)) for key, rows in splits.items()},
        'sha256': {split: file_hash(ROOT / f'data/{split}.jsonl') for split in splits},
        'human_review_ids': [row['id'] for row in splits['test'] if row['id'].endswith(('-0', '-1'))]
    })
    # Cinco IDs por categoria, fixados antes da geração de respostas.
    manifest_path = ROOT / 'data/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['human_review_ids'] = [row['id'] for row in splits['test']
                                    if row['id'].endswith('-0') or
                                    (row['id'].endswith('-1') and not row['family'].endswith('-15'))]
    write_json(manifest_path, manifest)
    print({key: len(rows) for key, rows in splits.items()})


if __name__ == '__main__':
    main()
