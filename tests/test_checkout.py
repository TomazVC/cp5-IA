import hashlib
import json
import subprocess
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from support import ROOT


class CheckoutTests(unittest.TestCase):
    def test_windows_checkout_does_not_change_dataset_hashes(self):
        cache = ROOT / '.cache'
        cache.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=cache) as directory:
            for autocrlf in ('true', 'false'):
                with self.subTest(autocrlf=autocrlf):
                    clone = Path(directory) / autocrlf
                    self.assertTrue(clone.resolve().is_relative_to(ROOT))
                    clone.mkdir()
                    subprocess.run(['git', '-c', f'core.autocrlf={autocrlf}', 'checkout-index', '--all',
                                    '--prefix', clone.as_posix() + '/'], cwd=ROOT, check=True)
                    manifest = json.loads((clone / 'data/manifest.json').read_text(encoding='utf-8'))
                    for split in ('train', 'validation', 'test'):
                        actual = hashlib.sha256((clone / f'data/{split}.jsonl').read_bytes()).hexdigest()
                        self.assertEqual(actual, manifest['sha256'][split], 'Checkout alterou bytes do dataset')


if __name__ == '__main__':
    unittest.main()
