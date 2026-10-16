import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import experiments


class ReportTests(unittest.TestCase):
    def test_completed_ai_review_is_reported_with_averages(self):
        self.assertTrue(hasattr(experiments, 'review_report_lines'), 'Resumo do CSV ausente no gerador')
        fields = ['observacoes'] + [f'{variant}_{criterion}_0_2'
            for variant in ('original', 'trained') for criterion in ('pertinencia', 'clareza', 'seguranca')]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'reports').mkdir()
            with (root / 'reports/human_review.csv').open('w', newline='', encoding='utf-8-sig') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, ['Revisão por IA.', '0', '1', '2', '2', '2', '2'])))
                writer.writerow(dict(zip(fields, ['Revisão por IA.', '1', '1', '0', '2', '1', '2'])))
            with patch.object(experiments, 'ROOT', root):
                text = '\n'.join(experiments.review_report_lines())
            self.assertIn('por IA', text)
            self.assertIn('12', text)
            self.assertIn('| Pertinência | 0,50 | 2,00 |', text)
            self.assertIn('| Clareza | 1,00 | 1,50 |', text)
            self.assertIn('| Segurança | 1,00 | 2,00 |', text)
            self.assertNotIn('Pendente:', text)

    def test_partial_review_does_not_present_incomplete_averages(self):
        self.assertTrue(hasattr(experiments, 'review_report_lines'), 'Resumo do CSV ausente no gerador')
        fields = ['observacoes'] + [f'{variant}_{criterion}_0_2'
            for variant in ('original', 'trained') for criterion in ('pertinencia', 'clareza', 'seguranca')]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'reports').mkdir()
            with (root / 'reports/human_review.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, ['Revisão por IA.', '0', '', '', '', '', ''])))
            with patch.object(experiments, 'ROOT', root):
                text = '\n'.join(experiments.review_report_lines())
            self.assertIn('1 de 6', text)
            self.assertNotIn('| Pertinência |', text)

    def test_invalid_score_cannot_be_reported_as_completed(self):
        self.assertTrue(hasattr(experiments, 'review_report_lines'), 'Resumo do CSV ausente no gerador')
        fields = ['observacoes'] + [f'{variant}_{criterion}_0_2'
            for variant in ('original', 'trained') for criterion in ('pertinencia', 'clareza', 'seguranca')]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'reports').mkdir()
            with (root / 'reports/human_review.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, ['Revisão por IA.', '3', '1', '2', '2', '2', '2'])))
            with patch.object(experiments, 'ROOT', root), self.assertRaisesRegex(ValueError, 'nota'):
                experiments.review_report_lines()
