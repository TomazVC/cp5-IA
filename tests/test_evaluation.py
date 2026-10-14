import unittest
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

try:
    import experiments
except ImportError:
    experiments = None


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(experiments, 'Seleção ainda não implementada')

    def test_selection_uses_accuracy_then_format_then_validation_loss(self):
        runs = [
            {'name': 'a', 'metrics': {'accuracy': .8, 'json_valid_rate': .9}, 'eval_loss': .2},
            {'name': 'b', 'metrics': {'accuracy': .9, 'json_valid_rate': .8}, 'eval_loss': .5},
            {'name': 'c', 'metrics': {'accuracy': .9, 'json_valid_rate': .9}, 'eval_loss': .7},
            {'name': 'd', 'metrics': {'accuracy': .9, 'json_valid_rate': .9}, 'eval_loss': .6},
        ]
        self.assertEqual(experiments.choose_training(runs)['name'], 'd')

    def test_generation_ties_prefer_deterministic_output(self):
        runs = [
            {'temperature': .7, 'metrics': {'accuracy': .8, 'json_valid_rate': 1}},
            {'temperature': 0, 'metrics': {'accuracy': .8, 'json_valid_rate': 1}},
        ]
        self.assertEqual(experiments.choose_generation(runs)['temperature'], 0)

    def test_test_set_cannot_be_used_for_selection(self):
        with self.assertRaisesRegex(ValueError, 'validação'):
            experiments.choose_training([{'name': 'bad', 'split': 'test',
                'metrics': {'accuracy': 1, 'json_valid_rate': 1}, 'eval_loss': .1}])

    def test_human_review_preserves_notes_but_rejects_changed_responses(self):
        import csv
        row = {'id': 'a', 'prompt': 'Perdi a senha', 'category': 'acesso', 'action': 'Recupere', 'raw': 'Resposta original'}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'reports').mkdir()
            (root / 'data/manifest.json').write_text(json.dumps({'human_review_ids': ['a']}), encoding='utf-8')
            with patch.object(experiments, 'ROOT', root):
                original = {'rows': [row]}
                trained = {'rows': [dict(row, raw='Resposta treinada')]}
                experiments.human_review(original, trained)
                path = root / 'reports/human_review.csv'
                records = list(csv.DictReader(path.open(encoding='utf-8-sig')))
                records[0]['observacoes'] = 'Minha nota humana'
                with path.open('w', newline='', encoding='utf-8-sig') as handle:
                    writer = csv.DictWriter(handle, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)
                experiments.human_review(original, trained)
                self.assertIn('Minha nota humana', path.read_text(encoding='utf-8-sig'))
                with self.assertRaisesRegex(ValueError, 'CSV'):
                    experiments.human_review(original, {'rows': [dict(row, raw='Outra resposta')]})

    def test_cached_evaluation_rejects_changed_system_prompt(self):
        import evaluate
        import support
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'reports/original').mkdir(parents=True)
            data_path = root / 'data/validation.jsonl'
            data_path.write_text('dados', encoding='utf-8')
            row = {'id': 'fixture', 'family': 'senha', 'prompt': 'Esqueci a senha.',
                   'category': 'acesso', 'action': 'Recupere a senha.'}
            with patch.object(evaluate, 'ROOT', root), patch.object(evaluate, 'load_splits', return_value={'validation': [row]}):
                evaluate.evaluate()
                with patch.object(support, 'SYSTEM_PROMPT', support.SYSTEM_PROMPT + ' Use linguagem simples.'):
                    with self.assertRaisesRegex(ValueError, 'configuração'):
                        evaluate.evaluate()


if __name__ == '__main__':
    unittest.main()
