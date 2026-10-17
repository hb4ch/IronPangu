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
    def test_eos_scope_preserves_reference_and_requests_no_forced_continuation(self):
        cases = [dict(input_ids=[1], output_ids=[2, 99, 7, 8]),
                 dict(input_ids=[1], output_ids=[99, 3]),
                 dict(input_ids=[1], output_ids=[2, 3])]
        scoped = module.scoped_cases(cases, 99, True)
        self.assertEqual([c['output_ids'] for c in scoped], [[2, 99], [99], [2, 3]])
        self.assertEqual(cases[0]['output_ids'], [2, 99, 7, 8])
        self.assertEqual(module.scoped_cases(cases, 99, False), cases)
        for invalid in [dict(output_ids=[]), dict(output_ids=[99], qualification=dict(output_is_placeholder=True))]:
            for mode in [True, False]:
                with self.assertRaises(ValueError):
                    module.scoped_cases([invalid], 99, mode)

    def test_uncaptured_forward_audit_rejects_graphs_missing_counts_and_fragmentation(self):
        program = dict(uncaptured_prefill=True, prefill_tokens=4096, prefill_variants=[])
        audit = dict(graphs=[dict(name='token_decode', replays=2)],
                     uncaptured_prefill_forwards=18, uncaptured_prefill_tokens=8192+65536,
                     transient_invocations=0, peak_transient_invocations=8,
                     transient_invocation_preparations=144, retained_invocations=4)
        cases = [dict(input_ids=[0]*65536)]
        report = module.validate_uncaptured_prefill(program, audit, cases)
        self.assertEqual(report['preparation_forwards'], 2)
        self.assertEqual(report['request_forwards'], 16)
        self.assertEqual(report['request_tokens'], 65536)
        self.assertEqual(report['expected_request_segments'], [[[4096]*16]])
        for key, value in [('uncaptured_prefill_forwards', 19), ('uncaptured_prefill_tokens', 8192+65535),
                           ('graphs', [dict(name='packed_prefill', replays=16)]),
                           ('transient_invocations', 1), ('retained_invocations', None),
                           ('peak_transient_invocations', 145),
                           ('graphs', [dict(name='token_decode', replays=1),dict(name='packed_prefill', replays=0)])]:
            with self.assertRaises(RuntimeError):
                module.validate_uncaptured_prefill(program, dict(audit, **{key:value}), cases)
        missing = dict(audit); missing.pop('uncaptured_prefill_forwards')
        with self.assertRaises(RuntimeError):
            module.validate_uncaptured_prefill(program, missing, cases)
        tail = dict(audit, uncaptured_prefill_forwards=3, uncaptured_prefill_tokens=8192+4096)
        report = module.validate_uncaptured_prefill(program, tail, [dict(input_ids=[0]*4097)])
        self.assertEqual(report['request_forwards'], 1)
        self.assertEqual(report['expected_request_segments'], [[[4096, 1]]])
        copied = dict(audit, uncaptured_prefill_forwards=5, uncaptured_prefill_tokens=8192+3*4096)
        report = module.validate_uncaptured_prefill(program, copied,
            [dict(input_ids=[0]*8192)], copy_boundary=4096)
        self.assertEqual(report['expected_request_segments'], [[[4096], [4096], [4096]]])

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
