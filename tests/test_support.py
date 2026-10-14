import json
import unittest

try:
    import support
except ImportError:
    support = None


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(support, 'Contrato de dados ainda não implementado')

    def test_invalid_output_counts_as_classification_error(self):
        rows = [
            {'category': 'acesso', 'raw': '{"categoria":"acesso","acao_inicial":"Recupere a senha."}'},
            {'category': 'rede', 'raw': 'Não sei'},
        ]
        result = support.score_rows(rows)
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['json_valid_rate'], 0.5)
        self.assertEqual(result['confusion_matrix']['rede']['invalido'], 1)

    def test_parser_rejects_unknown_category_empty_action_and_extra_fields(self):
        for text in ['{}', '[]', 'null', '{"categoria":"wifi","acao_inicial":"Teste"}',
                     '{"categoria":"rede","acao_inicial":" "}',
                     '{"categoria":"rede","acao_inicial":12}',
                     '{"categoria":"rede","acao_inicial":"Teste","extra":1}',
                     '```json\n{"categoria":"rede","acao_inicial":"Teste"}\n```']:
            self.assertIsNone(support.parse_response(text), text)

    def test_family_leakage_rejected_even_with_different_prompts(self):
        row = {'id': 'a', 'family': 'senha', 'prompt': 'Perdi a senha',
               'category': 'acesso', 'action': 'Recupere a senha.'}
        second = dict(row, id='b', prompt='Esqueci minha senha')
        with self.assertRaisesRegex(ValueError, 'família'):
            support.validate_splits({'train': [row], 'validation': [second], 'test': []})

    def test_dataset_has_disjoint_families_and_balanced_sizes(self):
        scenarios = json.loads((support.ROOT / 'data/scenarios.json').read_text(encoding='utf-8'))
        splits = support.make_splits(scenarios)
        support.validate_splits(splits)
        self.assertEqual({key: len(value) for key, value in splits.items()},
                         {'train': 250, 'validation': 75, 'test': 75})
        for key, count in [('train', 50), ('validation', 15), ('test', 15)]:
            for category in ['acesso', 'rede', 'hardware', 'software', 'outros']:
                self.assertEqual(sum(r['category'] == category for r in splits[key]), count)

    def test_empty_and_oversized_input_rejected(self):
        for prompt in ['', '   ', 'x' * 1201]:
            with self.assertRaises(ValueError):
                support.validate_prompt(prompt)
        self.assertEqual(support.validate_prompt('  Não consigo entrar  '), 'Não consigo entrar')


if __name__ == '__main__':
    unittest.main()
