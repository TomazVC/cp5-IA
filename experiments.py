"""Executa e retoma a sequência acadêmica sem usar teste final na seleção."""
import csv
import json
from model_runtime import CONFIG, require_cuda
from evaluate import evaluate
from support import ROOT, file_hash, load_splits, prompt_hash, write_json
from train import full_training_steps, run_name, train


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
    baseline = {row['id']: row for row in original['rows']}
    fields = ['id', 'prompt', 'category', 'reference_action', 'original_response', 'trained_response',
              'original_pertinencia_0_2', 'original_clareza_0_2', 'original_seguranca_0_2',
              'trained_pertinencia_0_2', 'trained_clareza_0_2', 'trained_seguranca_0_2', 'observacoes']
    records = [{'id': row['id'], 'prompt': row['prompt'], 'category': row['category'],
                'reference_action': row['action'], 'original_response': baseline[row['id']]['raw'],
                'trained_response': row['raw']} for row in trained['rows'] if row['id'] in selected_ids]
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as handle:
            existing = list(csv.DictReader(handle))
        identity_fields = fields[:6]
        if len(existing) != len(records) or any(
                any(old.get(key) != new[key] for key in identity_fields) for old, new in zip(existing, records)):
            raise ValueError('CSV humano pertence a outra avaliação. Arquive reports/human_review.csv antes de repetir.')
        print('CSV da mesma avaliação já existe; respostas e notas registradas preservadas.', flush=True)
        return
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def review_report_lines():
    path = ROOT / 'reports/human_review.csv'
    lines = ['## Avaliação qualitativa das respostas', '']
    if not path.exists():
        return lines + ['Nenhuma revisão qualitativa registrada em reports/human_review.csv.', '']
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    criteria = [('pertinencia', 'Pertinência'), ('clareza', 'Clareza'), ('seguranca', 'Segurança')]
    fields = [f'{variant}_{key}_0_2' for variant in ('original', 'trained') for key, _ in criteria]
    values = [(row.get(field) or '').strip() for row in rows for field in fields]
    if any(value not in ('', '0', '1', '2') for value in values):
        raise ValueError('CSV de revisão contém nota inválida; use 0, 1 ou 2.')
    filled = sum(value != '' for value in values)
    if not rows or filled != len(values):
        return lines + [f'Revisão registrada em [reports/human_review.csv](../reports/human_review.csv): '
                        f'{filled} de {len(values)} notas preenchidas. Médias disponíveis após completar as notas.', '']
    by_ai = all('revisão por ia' in (row.get('observacoes') or '').casefold() for row in rows)
    if by_ai:
        lines[0] = '## Revisão qualitativa por IA'
    lines.extend([f'O arquivo [reports/human_review.csv](../reports/human_review.csv) contém '
                  f'{len(rows)} casos e {filled} notas preenchidas de 0 a 2. '
                  'A autoria e as justificativas estão identificadas nas observações do CSV.', '',
                  '| Critério — média de 0 a 2 | Original | Treinado |', '|---|---:|---:|'])
    for key, label in criteria:
        means = [sum(int(row[f'{variant}_{key}_0_2']) for row in rows) / len(rows)
                 for variant in ('original', 'trained')]
        original, trained = [f'{mean:.2f}'.replace('.', ',') for mean in means]
        lines.append(f'| {label} | {original} | {trained} |')
    lines.extend(['', 'Rubrica: 0 inadequada, 1 parcialmente adequada, 2 adequada. '
                  'Pertinência mede a adequação da ação ao problema; clareza, sua compreensão; '
                  'segurança, as precauções e a ausência de orientações perigosas. '
                  'Acertar a categoria não garante uma ação adequada.', ''])
    if by_ai:
        lines.extend(['A avaliação foi realizada por IA e não foi cega, pois as variantes dos modelos '
                      'estavam identificadas. As notas são subjetivas e não constituem revisão humana independente.', ''])
    return lines


def write_report(baseline, runs, generation, selection, original, trained):
    lines = ['# Resultados reais — CP5', '',
             f"GPU: {runs[0]['training']['gpu']}. Modelo: {CONFIG['model_id']}.",
             f"Revisão: `{CONFIG['revision']}`. Seed: {CONFIG['seed']}.", '',
             f"Parâmetros totais (incluindo adaptadores): {runs[0]['training']['parameters']['total']:,}; "
             f"treináveis: {runs[0]['training']['parameters']['trainable']:,}. "
             'Todos os três treinos completam três épocas/96 atualizações; o checkpoint salvo usa menor loss de validação.', '',
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
    lines.extend(review_report_lines())
    lines.extend(['## Limitações', '',
        'Dados sintéticos: 80 cenários com cinco redações cada; os conjuntos têm famílias distintas, '
        'mas apenas 15 cenários independentes em cada conjunto de avaliação. '
        'As 75 respostas não são 75 cenários independentes. Não extrapolar a acurácia para chamados reais.',
        '', 'Uma única seed de treinamento e de amostragem: a comparação de temperaturas não mede estabilidade '
        'entre múltiplas seeds. Latência mede a chamada de geração com sincronização CUDA, excluindo '
        'tokenização, seed, decode e carregamento; varia com comprimento da resposta, temperatura da GPU e programas concorrentes.',
        '', 'As notas qualitativas complementam as métricas automáticas e não demonstram segurança para atendimento real.',
        '', '## Evidências e publicação', '',
        'Os prints estão em [evidence/](../evidence/) e incorporados ao [README](../README.md):', '',
        '- [GPU funcionando no VSCode](../evidence/gpu-vscode.png).',
        '- [Frontend com modelo original](../evidence/evidencia-front-original.png).',
        '- [Frontend com modelo treinado](../evidence/evidencia-front-treinado.png).',
        '- [Notebook de comprovação da GPU](../evidence/evidencia-cp5.png).', '',
        'Os prints do frontend usam o mesmo chamado e temperatura 0. '
        'O notebook executado está em [gpu-vscode.ipynb](../evidence/gpu-vscode.ipynb).', '',
        'Repositório de entrega: [TomazVC/cp5-IA](https://github.com/TomazVC/cp5-IA). '
        'O autor confirmou a verificação do acesso para entrega.', ''])
    (ROOT / 'docs/RESULTADOS.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    require_cuda()
    splits = load_splits()
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
        expected_steps = full_training_steps(len(splits['train']), CONFIG['batch_size'],
                                             CONFIG['gradient_accumulation_steps'], CONFIG['epochs'])
        if (stats['config'] != CONFIG or stats['data_sha256'] != current_hashes or stats['smoke']
                or stats.get('prompt_sha256') != prompt_hash() or stats['global_step'] != expected_steps):
            raise ValueError(f'{name} pertence a outros dados/parâmetros ou é smoke; preserve antes de repetir.')
        result = evaluate(adapter, name=name)
        runs.append(dict(result, eval_loss=stats['eval_loss'], training=stats))
    best = choose_training(runs)
    adapter = ROOT / f'artifacts/{best["name"]}'
    generation = [evaluate(adapter, temperature=temp, name=best['name']) for temp in CONFIG['temperatures']]
    selected_generation = choose_generation(generation)
    selection = {'name': best['name'], 'adapter': adapter.relative_to(ROOT).as_posix(),
                 'temperature': selected_generation['temperature'], 'config': CONFIG,
                 'data_sha256': best['training']['data_sha256'],
                 'adapter_sha256': file_hash(adapter / 'adapter_model.safetensors'),
                 'prompt_sha256': prompt_hash(),
                 'rule': 'validation accuracy, JSON validity; training loss tie-break; greedy generation tie-break'}
    write_json(ROOT / 'reports/selection.json', selection)
    original = evaluate(split='test', temperature=selection['temperature'])
    trained = evaluate(adapter, split='test', temperature=selection['temperature'], name=best['name'])
    human_review(original, trained)
    write_report(baseline, runs, generation, selection, original, trained)
    print('Experimentos concluídos. Veja docs/RESULTADOS.md e reports/human_review.csv.', flush=True)


if __name__ == '__main__':
    main()
