import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('qualify', Path(__file__).parents[1] / 'qualify-sequence-serving.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OutputComparison(unittest.TestCase):
    def test_first_divergence_is_separate_for_each_request(self):
        reference = {(0, 0): 10, (0, 1): 20, (1, 0): 30}
        actual = [dict(request=0, step=0, token=11), dict(request=0, step=1, token=21)]
        result = module.compare_output(actual, reference)
        self.assertFalse(result['passed'])
        self.assertEqual(result['first_difference_by_request'], [
            dict(request=0, step=0, expected_token=10, actual_token=11),
            dict(request=1, step=0, expected_token=30, actual_token=None)])

    def test_duplicate_and_extra_records_cannot_pass(self):
        row = dict(request=0, step=0, token=10)
        reference = {(0, 0): 10}
        self.assertTrue(module.compare_output([row], reference)['passed'])
        duplicate = module.compare_output([row, row], reference)
        self.assertFalse(duplicate['passed'])
        self.assertEqual(duplicate['duplicates'], [dict(request=0, step=0)])
        extra = module.compare_output([row, dict(request=0, step=1, token=20)], reference)
        self.assertFalse(extra['passed'])
        self.assertIsNone(extra['first_difference_by_request'][0]['expected_token'])


if __name__ == '__main__':
    unittest.main()
