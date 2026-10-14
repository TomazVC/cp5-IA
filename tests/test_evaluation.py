import unittest

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


if __name__ == '__main__':
    unittest.main()
