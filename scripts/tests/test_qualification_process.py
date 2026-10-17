import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('qualify', Path(__file__).parents[1] / 'qualify-program-engine.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QualificationProcess(unittest.TestCase):
    def test_success(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            text = module.run_engine([sys.executable, '-c', 'print("evidence")'], os.environ, out, 5)
            self.assertEqual(text, 'evidence\n')
            report = json.loads((out/'process.json').read_text())
            self.assertTrue(report['completed'])
            self.assertFalse(report['timed_out'])
            self.assertEqual(report['returncode'], 0)

    def test_failure_retains_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                module.run_engine([sys.executable, '-c', 'print("partial"); raise SystemExit(7)'], os.environ, out, 5)
            self.assertEqual((out/'tokens.jsonl').read_text(), 'partial\n')
            self.assertEqual(json.loads((out/'process.json').read_text())['returncode'], 7)
            self.assertFalse((out/'results.json').exists())

    def test_timeout_retains_output_without_success(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                module.run_engine([sys.executable, '-c', 'import time; print("partial", flush=True); time.sleep(10)'], os.environ, out, 1)
            self.assertEqual((out/'tokens.jsonl').read_text(), 'partial\n')
            report = json.loads((out/'process.json').read_text())
            self.assertFalse(report['completed'])
            self.assertTrue(report['timed_out'])
            self.assertFalse((out/'results.json').exists())


if __name__ == '__main__':
    unittest.main()
