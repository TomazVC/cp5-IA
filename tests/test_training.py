import unittest

try:
    import model_runtime as runtime
except ImportError:
    runtime = None


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runtime, 'Runtime ainda não implementado')

    def test_prompt_and_padding_never_enter_training_loss(self):
        tokenizer = runtime.get_tokenizer()
        row = {'prompt': 'Esqueci a senha.', 'category': 'acesso', 'action': 'Recupere a senha.'}
        encoded = runtime.encode_example(tokenizer, row, 256)
        labels = encoded['labels']
        start = next(i for i, value in enumerate(labels) if value != -100)
        self.assertGreater(start, 20)
        self.assertTrue(all(value == -100 for value in labels[:start]))
        response = tokenizer.decode([value for value in labels if value != -100])
        self.assertIn('"categoria": "acesso"', response)
        self.assertNotIn('Esqueci', response)
        padded = runtime.collate_examples([encoded, runtime.encode_example(tokenizer,
            dict(row, action='Solicite ao suporte a recuperação da sua conta pelo canal oficial.'), 256)])
        self.assertTrue((padded['labels'][padded['attention_mask'] == 0] == -100).all().item())

    def test_long_training_record_is_rejected_instead_of_truncated(self):
        with self.assertRaisesRegex(ValueError, 'tokens'):
            runtime.encode_example(runtime.get_tokenizer(),
                {'prompt': 'rede ' * 500, 'category': 'rede', 'action': 'Confira a rede.'}, 256)

    def test_real_generation_uses_cuda_and_keeps_raw_output(self):
        with runtime.Runtime() as engine:
            self.assertTrue(all(p.device.type == 'cuda' for p in engine.model.parameters()))
            result = engine.generate('Esqueci a senha do portal.', temperature=0, seed=42)
            self.assertTrue(result['raw'])
            self.assertGreater(result['peak_vram_mb'], 0)
            self.assertGreater(result['seconds'], 0)
            with self.assertRaisesRegex(ValueError, 'tokens'):
                engine.generate('palavra ' * 150, temperature=0, seed=42)


if __name__ == '__main__':
    unittest.main()
