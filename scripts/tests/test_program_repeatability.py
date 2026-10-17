import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('repeatability', Path(__file__).parents[1] / 'check-program-repeatability.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Repeatability(unittest.TestCase):
    def setUp(self):
        self.cases = [dict(case='sample', output_ids=[2, 99, 7])]
        self.rows = [dict(request=request, generation_step=step, token=token,
                          logit=10, top5=[dict(token=token, logprob=-0.1)])
                     for request in range(3) for step, token in enumerate([2, 99, 7])]

    def test_same_tokens_do_not_hide_logit_or_top_five_drift(self):
        self.assertTrue(module.compare(self.cases, self.rows, 3, 99)[0]['stable'])
        for field, value in [('logit', 11), ('top5', [dict(token=2, logprob=-0.2)])]:
            rows = copy.deepcopy(self.rows)
            rows[6][field] = value
            self.assertFalse(module.compare(self.cases, rows, 3, 99)[0]['stable'])

    def test_eos_suffix_ignored_but_missing_prefix_rejected(self):
        self.rows[-1]['logit'] = 999
        self.assertTrue(module.compare(self.cases, self.rows, 3, 99)[0]['stable'])
        self.assertFalse(module.compare(self.cases, self.rows, 3)[0]['stable'])
        with self.assertRaises(ValueError):
            module.compare(self.cases, self.rows[:-2], 3, 99)
        self.rows[4]['generation_step'] = 0
        with self.assertRaises(ValueError):
            module.compare(self.cases, self.rows, 3, 99)


if __name__ == '__main__':
    unittest.main()
