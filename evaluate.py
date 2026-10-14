"""Salva respostas brutas e métricas; nenhum juiz remoto é usado."""
import argparse
import json
import statistics

from model_runtime import CONFIG, Runtime
from support import ROOT, file_hash, load_splits, prompt_hash, score_rows, write_json


def evaluate(adapter=None, split='validation', temperature=0, name='original'):
    if split not in ('validation', 'test'):
        raise ValueError('Avalie somente validação ou teste.')
    rows = load_splits()[split]
    output = ROOT / f'reports/{name}/{split}-temp-{temperature:g}.json'
    identity = {'config': CONFIG, 'data_sha256': file_hash(ROOT / f'data/{split}.jsonl'),
                'adapter_sha256': file_hash(adapter / 'adapter_model.safetensors') if adapter else None,
                'prompt_sha256': prompt_hash()}
    if output.exists():
        saved = json.loads(output.read_text(encoding='utf-8'))
        if saved['identity'] != identity:
            raise ValueError(f'Relatório {output} pertence a outra configuração. Preserve e mova antes de repetir.')
        print(f'Reutilizando avaliação concluída: {output.relative_to(ROOT)}', flush=True)
        return saved
    predictions = []
    with Runtime(adapter) as engine:
        engine.generate('Esqueci a senha do portal.', temperature=0)  # aquecimento, fora das métricas
        for index, row in enumerate(rows):
            result = engine.generate(row['prompt'], temperature=temperature, seed=CONFIG['seed'])
            predictions.append(dict(row, **result))
            if (index + 1) % 15 == 0:
                print(f'{name} {split} temp={temperature:g}: {index + 1}/{len(rows)}', flush=True)
    metrics = score_rows(predictions)
    metrics.update(mean_seconds=statistics.mean(r['seconds'] for r in predictions),
                   median_seconds=statistics.median(r['seconds'] for r in predictions),
                   peak_vram_mb=max(r['peak_vram_mb'] for r in predictions),
                   mean_output_tokens=statistics.mean(r['output_tokens'] for r in predictions))
    report = {'name': name, 'split': split, 'temperature': temperature, 'identity': identity,
              'metrics': metrics, 'rows': predictions,
              'latency_note': 'Batch 1, aquecimento excluído, CUDA sincronizada. Tempo da geração; exclui seed, tokenização, decode, download e carregamento.'}
    write_json(output, report)
    print(f'{name}: accuracy={metrics["accuracy"]:.3f}, JSON={metrics["json_valid_rate"]:.3f}', flush=True)
    return report


if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--name', default='original')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--temperature', type=float, choices=CONFIG['temperatures'], default=0)
    options = parser.parse_args()
    evaluate(options.adapter, options.split, options.temperature, options.name)
