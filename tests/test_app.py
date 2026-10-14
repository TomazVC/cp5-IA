import unittest

try:
    import app
except ImportError:
    app = None


class AppTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(app, 'Interface ainda não implementada')

    def test_blank_prompt_rejected_before_loading_model(self):
        with self.assertRaises(ValueError):
            app.respond('  ', 'Original', 0)

    def test_oversized_prompt_rejected_before_loading_model(self):
        with self.assertRaises(ValueError):
            app.respond('x' * 1201, 'Original', 0)

    def test_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, 'modelo'):
            app.respond('Preciso recuperar minha conta.', 'Inexistente', 0)


if __name__ == '__main__':
    unittest.main()
