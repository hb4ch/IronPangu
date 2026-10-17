import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('divergence', Path(__file__).parents[1] / 'analyze-token-divergence.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TokenDivergence(unittest.TestCase):
    def fixture(self):
        case = dict(case='sample', input_ids=[1, 2], output_ids=[3, 4, 5],
                    logprobs=[{'3': {'logprob': -0.1}}, {'4': {'logprob': -0.1}, '6': {'logprob': -2.1}}, {}])
        rows = [dict(step=1, token=3, top5=[dict(token=3, logprob=-0.2)]),
                dict(request=0, step=0, token=3),
                dict(step=2, token=6, top5=[dict(token=6, logprob=-0.3), dict(token=4, logprob=-1.3)]),
                dict(request=0, step=1, token=6)]
        return case, rows

    def test_compares_only_first_divergence_at_identical_history(self):
        case, rows = self.fixture()
        rows += [dict(step=3, token=7, top5=[dict(token=7, logprob=-0.1)]), dict(request=0, step=2, token=7)]
        result = module.analyze(case, rows, case['input_ids'])
        self.assertEqual(result['matching_prefix_tokens'], 1)
        first = result['first_difference']
        self.assertEqual(first['position'], 2)
        self.assertAlmostEqual(first['native_actual_minus_expected_logprob'], 1.0)
        self.assertAlmostEqual(first['reference_actual_minus_expected_logprob'], -2.0)

    def test_rejects_wrong_positions_missing_diagnostics_and_intervention(self):
        case, rows = self.fixture()
        bad = copy.deepcopy(rows); bad[2]['step'] += 1
        for variant in (bad, rows[:-1], rows[1:]):
            with self.assertRaises(ValueError):
                module.analyze(case, variant, case['input_ids'])
        with self.assertRaisesRegex(ValueError, 'prompt differs'):
            module.analyze(case, rows, [2, 1])
        case['reference_intervention'] = 'changed coefficients'
        with self.assertRaises(ValueError):
            module.analyze(case, rows, case['input_ids'])

    def test_partial_equal_prefix_is_not_full_qualification(self):
        case, rows = self.fixture()
        result = module.analyze(case, rows[:2], case['input_ids'])
        self.assertIsNone(result['first_difference'])
        self.assertEqual(result['inspected_outputs'], 1)
        self.assertEqual(result['reference_outputs'], 3)
        rows[2]['top5'][0]['logprob'] = float('nan')
        with self.assertRaises(ValueError):
            module.analyze(case, rows, case['input_ids'])


if __name__ == '__main__':
    unittest.main()
