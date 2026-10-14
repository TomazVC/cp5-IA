"""Contrato de dados, validação e métricas, sem dependência de GPU."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATEGORIES = ('acesso', 'rede', 'hardware', 'software', 'outros')
SYSTEM_PROMPT = (
    'Você faz triagem de chamados de suporte em português. '
    'Responda somente um JSON com categoria e acao_inicial. '
    'Categorias: acesso (conta, senha, autenticação, permissão), '
    'rede (conectividade, Wi-Fi, DNS, VPN), hardware (defeito físico, periférico), '
    'software (instalação, atualização, falha de aplicativo), outros (ambíguo ou fora do escopo). '
    'Use o impedimento principal; se não for claro, use outros e peça esclarecimento. '
    'Sugira uma ação inicial curta, pertinente e não destrutiva. Não peça senhas.'
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_hash():
    return hashlib.sha256(SYSTEM_PROMPT.encode('utf-8')).hexdigest()


def validate_prompt(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Descreva o chamado antes de enviar.')
    if len(prompt.strip()) > 1200:
        raise ValueError('O chamado deve ter no máximo 1200 caracteres.')
    return prompt.strip()


def parse_response(text):
    try:
        result = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(result, dict) or set(result) != {'categoria', 'acao_inicial'}:
        return None
    if result['categoria'] not in CATEGORIES:
        return None
    if not isinstance(result['acao_inicial'], str) or not result['acao_inicial'].strip():
        return None
    return result


def score_rows(rows):
    if not rows:
        raise ValueError('Não há respostas para avaliar.')
    matrix = {category: dict.fromkeys((*CATEGORIES, 'invalido'), 0) for category in CATEGORIES}
    valid = correct = 0
    for row in rows:
        parsed = parse_response(row['raw'])
        predicted = parsed['categoria'] if parsed else 'invalido'
        matrix[row['category']][predicted] += 1
        valid += parsed is not None
        correct += predicted == row['category']
    f1_scores = []
    for category in CATEGORIES:
        tp = matrix[category][category]
        fp = sum(matrix[other][category] for other in CATEGORIES if other != category)
        fn = sum(matrix[category].values()) - tp
        f1_scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return {'n': len(rows), 'accuracy': correct / len(rows),
            'macro_f1': sum(f1_scores) / len(CATEGORIES),
            'json_valid_rate': valid / len(rows), 'confusion_matrix': matrix}


def validate_splits(splits):
    seen_ids, seen_prompts, families = set(), set(), {}
    for split, rows in splits.items():
        for row in rows:
            if set(row) != {'id', 'family', 'prompt', 'category', 'action'}:
                raise ValueError('Campos inesperados no dataset.')
            if any(not isinstance(value, str) or not value.strip() for value in row.values()):
                raise ValueError('Campos vazios ou não textuais no dataset.')
            if row['category'] not in CATEGORIES:
                raise ValueError('Categoria desconhecida.')
            validate_prompt(row['prompt'])
            normalized = ' '.join(row['prompt'].casefold().split())
            if row['id'] in seen_ids or normalized in seen_prompts:
                raise ValueError('ID ou prompt duplicado.')
            if row['family'] in families and families[row['family']] != split:
                raise ValueError('Vazamento de família entre conjuntos.')
            families[row['family']] = split
            seen_ids.add(row['id'])
            seen_prompts.add(normalized)


def make_splits(scenarios):
    splits = {'train': [], 'validation': [], 'test': []}
    # shortcut: variações de redação simples; ampliar com dados reais anonimizados para uso fora do CP5.
    templates = ['{issue}', 'Olá, preciso de ajuda: {issue}',
                 'Estou com este problema: {issue}', 'Chamado para o suporte: {issue}',
                 'Podem orientar o primeiro passo? {issue}']
    for category in CATEGORIES:
        if len(scenarios[category]) != 16:
            raise ValueError('Esperados 16 cenários por categoria.')
        for index, (issue, action) in enumerate(scenarios[category]):
            split = 'train' if index < 10 else 'validation' if index < 13 else 'test'
            family = f'{category}-{index:02d}'
            for variant, template in enumerate(templates):
                splits[split].append({'id': f'{family}-{variant}', 'family': family,
                                      'prompt': template.format(issue=issue),
                                      'category': category, 'action': action})
    validate_splits(splits)
    return splits


def load_splits():
    splits = {split: [json.loads(line) for line in (ROOT / f'data/{split}.jsonl')
                      .read_text(encoding='utf-8').splitlines() if line.strip()]
              for split in ('train', 'validation', 'test')}
    validate_splits(splits)
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    for split in splits:
        if file_hash(ROOT / f'data/{split}.jsonl') != manifest['sha256'][split]:
            raise ValueError(f'Dataset {split} mudou; confira os dados e regenere o manifesto conscientemente.')
    return splits
