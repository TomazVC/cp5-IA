"""Executa e retoma a sequência acadêmica sem usar teste final na seleção."""
import csv
import json
from model_runtime import CONFIG, require_cuda
from evaluate import evaluate
from support import ROOT, file_hash, load_splits, write_json
from train import run_name, train


def check_validation(runs):
    if not runs or any(run.get('split', 'validation') != 'validation' for run in runs):
        raise ValueError('A seleção exige avaliações de validação, nunca teste final.')


def choose_training(runs):
    check_validation(runs)
    return max(runs, key=lambda r: (r['metrics']['accuracy'], r['metrics']['json_valid_rate'], -r['eval_loss']))


def choose_generation(runs):
    check_validation(runs)
    return max(runs, key=lambda r: (r['metrics']['accuracy'], r['metrics']['json_valid_rate'], -r['temperature']))


def human_review(original, trained):
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    selected_ids = set(manifest['human_review_ids'])
    path = ROOT / 'reports/human_review.csv'
    if path.exists():
        print('CSV de revisão humana já existe; preservado.', flush=True)
        return
    baseline = {row['id']: row for row in original['rows']}
    fields = ['id', 'prompt', 'category', 'reference_action', 'original_response', 'trained_response',
              'original_pertinencia_0_2', 'original_clareza_0_2', 'original_seguranca_0_2',
              'trained_pertinencia_0_2', 'trained_clareza_0_2', 'trained_seguranca_0_2', 'observacoes']
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in trained['rows']:
            if row['id'] in selected_ids:
                writer.writerow({'id': row['id'], 'prompt': row['prompt'], 'category': row['category'],
                                 'reference_action': row['action'], 'original_response': baseline[row['id']]['raw'],
                                 'trained_response': row['raw']})


def write_report(baseline, runs, generation, selection, original, trained):
    lines = ['# Resultados reais — CP5', '',
             f"GPU: {runs[0]['training']['gpu']}. Modelo: {CONFIG['model_id']}.",
             f"Revisão: `{CONFIG['revision']}`. Seed: {CONFIG['seed']}.", '',
             '## Validação — treinamento', '',
             '| Modelo | Learning rate | Acurácia | JSON válido | Loss validação | Tempo treino (s) | VRAM treino (MiB) |',
             '|---|---:|---:|---:|---:|---:|---:|',
             f"| Original | — | {baseline['metrics']['accuracy']:.1%} | {baseline['metrics']['json_valid_rate']:.1%} | — | — | — |"]
    for run in runs:
        stats = run['training']
        lines.append(f"| {run['name']} | {stats['learning_rate']:g} | {run['metrics']['accuracy']:.1%} | "
                     f"{run['metrics']['json_valid_rate']:.1%} | {run['eval_loss']:.4f} | "
                     f"{stats['duration_seconds']:.1f} | {stats['peak_vram_mb']:.0f} |")
    lines.extend(['', '## Validação — geração', '', '| Temperatura | Acurácia | JSON válido | Tempo médio (s) |',
                  '|---:|---:|---:|---:|'])
    for run in generation:
        lines.append(f"| {run['temperature']:g} | {run['metrics']['accuracy']:.1%} | "
                     f"{run['metrics']['json_valid_rate']:.1%} | {run['metrics']['mean_seconds']:.2f} |")
    lines.extend(['', f"Seleção somente na validação: **{selection['name']}**, temperatura **{selection['temperature']:g}**.",
                  '', '## Teste final reservado — 75 chamados', '',
                  '| Modelo | Acurácia | Macro-F1 | JSON válido | Latência média (s) | Pico VRAM (MiB) |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for report in (original, trained):
        m = report['metrics']
        lines.append(f"| {report['name']} | {m['accuracy']:.1%} | {m['macro_f1']:.3f} | "
                     f"{m['json_valid_rate']:.1%} | {m['mean_seconds']:.2f} | {m['peak_vram_mb']:.0f} |")
    lines.extend(['', 'JSON inválido conta como erro de classificação e permanece no denominador. '
                  'Macro-F1 inclui todas as cinco categorias. Matrizes de confusão e respostas brutas estão nos JSONs em reports/.',
                  '', '## Exemplos', ''])
    pairs = list(zip(original['rows'], trained['rows']))
    for title, condition in [
        ('Melhoria', lambda a, b: (not a['parsed'] or a['parsed']['categoria'] != a['category']) and
                                 b['parsed'] and b['parsed']['categoria'] == b['category']),
        ('Regressão', lambda a, b: a['parsed'] and a['parsed']['categoria'] == a['category'] and
                                  (not b['parsed'] or b['parsed']['categoria'] != b['category'])),
        ('Falha do treinado', lambda a, b: not b['parsed'] or b['parsed']['categoria'] != b['category'])]:
        pair = next(((a, b) for a, b in pairs if condition(a, b)), None)
        if pair:
            a, b = pair
            lines.extend([f'### {title}', '', f"Chamado ({b['id']}, esperado {b['category']}): {b['prompt']}",
                          '', 'Original:', '```text', a['raw'], '```', 'Treinado:', '```text', b['raw'], '```', ''])
        else:
            lines.extend([f'{title}: nenhum caso nesta execução segundo acerto de categoria.', ''])
    lines.extend(['## Limitações e revisão humana', '',
        'Dados sintéticos: 80 cenários com cinco redações cada; os conjuntos têm famílias distintas, '
        'mas apenas 15 cenários independentes em cada conjunto de avaliação. '
        'As 75 respostas não são 75 cenários independentes. Não extrapolar a acurácia para chamados reais.',
        '', 'Uma única seed de treinamento e de amostragem: a comparação de temperaturas não mede estabilidade '
        'entre múltiplas seeds. Latência varia com comprimento da resposta, temperatura da GPU e programas concorrentes.',
        '', 'Pendente: usuário revisar referências/rótulos e preencher reports/human_review.csv '
        '(25 linhas; pertinência, clareza e segurança de 0 a 2 para cada modelo). '
        '0 = inadequado; 1 = parcialmente adequado; 2 = adequado. Notas humanas não foram inventadas.',
        '', 'Pendente: print real no VSCode e publicação no GitHub no destino definido pelo usuário.', ''])
    (ROOT / 'docs/RESULTADOS.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    require_cuda()
    load_splits()
    baseline = evaluate()
    runs = []
    for lr in CONFIG['learning_rates']:
        name = run_name(lr)
        adapter = ROOT / f'artifacts/{name}'
        marker = adapter / 'training_summary.json'
        if not marker.exists():
            train(lr)
        stats = json.loads(marker.read_text(encoding='utf-8'))
        current_hashes = {split: file_hash(ROOT / f'data/{split}.jsonl') for split in ('train', 'validation', 'test')}
        if stats['config'] != CONFIG or stats['data_sha256'] != current_hashes or stats['smoke']:
            raise ValueError(f'{name} pertence a outros dados/parâmetros ou é smoke; preserve antes de repetir.')
        result = evaluate(adapter, name=name)
        runs.append(dict(result, eval_loss=stats['eval_loss'], training=stats))
    best = choose_training(runs)
    adapter = ROOT / f'artifacts/{best["name"]}'
    generation = [evaluate(adapter, temperature=temp, name=best['name']) for temp in CONFIG['temperatures']]
    selected_generation = choose_generation(generation)
    selection = {'name': best['name'], 'adapter': str(adapter.relative_to(ROOT)),
                 'temperature': selected_generation['temperature'], 'config': CONFIG,
                 'data_sha256': best['training']['data_sha256'],
                 'adapter_sha256': file_hash(adapter / 'adapter_model.safetensors'),
                 'rule': 'validation accuracy, JSON validity; training loss tie-break; greedy generation tie-break'}
    write_json(ROOT / 'reports/selection.json', selection)
    original = evaluate(split='test', temperature=selection['temperature'])
    trained = evaluate(adapter, split='test', temperature=selection['temperature'], name=best['name'])
    human_review(original, trained)
    write_report(baseline, runs, generation, selection, original, trained)
    print('Experimentos concluídos. Veja docs/RESULTADOS.md e reports/human_review.csv.', flush=True)


if __name__ == '__main__':
    main()
