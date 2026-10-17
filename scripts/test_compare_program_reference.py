import importlib.util
from pathlib import Path
import unittest
import json
import subprocess
import sys
import tempfile

spec = importlib.util.spec_from_file_location('compare', Path(__file__).with_name('compare-program-reference.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.cases = [dict(case='a', input_ids=[10], output_ids=[4, 9, 5]),
                      dict(case='b', input_ids=[11, 12], output_ids=[6])]
        self.rows = [dict(request=0, step=0, token=4, elapsed_us=10000),
                     dict(request=1, step=0, token=6, elapsed_us=20000),
                     dict(request=0, step=1, token=9, elapsed_us=3000),
                     dict(request=0, step=2, token=5, elapsed_us=5000)]

    def test_diagnostic_intervention_cannot_become_an_oracle(self):
        self.cases[0]['reference_intervention'] = 'changed rotary tables'
        with self.assertRaisesRegex(ValueError, 'not an independent reference'):
            module.compare(self.cases, self.rows, 1, 9)

    def test_step_durations_are_not_cumulative_and_single_output_has_no_tpot(self):
        a, b = module.compare(self.cases, self.rows, 1, 9)
        self.assertTrue(a['exact_all_tokens'])
        self.assertEqual(a['ttft_ms_mean'], 10)
        self.assertEqual(a['tpot_ms_mean'], 4)
        self.assertIsNone(b['tpot_ms_mean'])

    def test_post_eos_difference_remains_a_full_output_failure(self):
        self.rows[-1]['token'] = 7
        a, _ = module.compare(self.cases, self.rows, 1, 9)
        self.assertTrue(a['exact_through_reference_eos'])
        self.assertFalse(a['exact_all_tokens'])

    def test_every_repetition_must_match(self):
        more = [dict(r, request=r['request'] + 2) for r in self.rows]
        more[0]['token'] = 8
        a, _ = module.compare(self.cases, self.rows + more, 2, 9)
        self.assertFalse(a['stable_repetitions'])
        self.assertFalse(a['exact_through_reference_eos'])

    def test_missing_extra_and_invalid_records_fail(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[-1]],
                     self.rows + [dict(request=99, step=0, token=4, elapsed_us=1)],
                     self.rows + [dict(unknown=True)],
                     [dict(self.rows[0], elapsed_us=float('nan'))] + self.rows[1:]):
            with self.assertRaises(ValueError):
                module.compare(self.cases, rows, 1, 9)

    def test_cli_retains_failure_evidence_and_exits_nonzero(self):
        self.rows[-1]['token'] = 7
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'reference.json').write_text(json.dumps(self.cases))
            (root/'tokens.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('compare-program-reference.py')),
                '--reference',str(root/'reference.json'),'--tokens',str(root/'tokens.jsonl'),
                '--output',str(root/'comparison.json'),'--repetitions','1','--eos','9'],capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stderr)
            report=json.loads((root/'comparison.json').read_text())
            self.assertFalse(report['cases'][0]['exact_all_tokens'])
            self.assertTrue(report['cases'][0]['exact_through_reference_eos'])


if __name__ == '__main__':
    unittest.main()
