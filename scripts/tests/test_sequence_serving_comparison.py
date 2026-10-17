import importlib.util
from pathlib import Path
import unittest
import json
import tempfile

spec = importlib.util.spec_from_file_location('qualify', Path(__file__).parents[1] / 'qualify-sequence-serving.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OutputComparison(unittest.TestCase):
    def test_stale_physical_page_names_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'device-program.json').write_text(json.dumps({'instructions': [
                {'name': 'cache_v2', 'op': 'cache_c8_norm_rope_append'},
                {'name': 'projection', 'op': 'linear'}]}))
            bindings = root / 'bindings.txt'
            for bad in ('0 cache_v1 0 1\n', '0 projection 0 1\n', '0 cache_v2 0\n0 cache_v2 1\n'):
                bindings.write_text(bad)
                with self.assertRaises(ValueError):
                    module.validate_binding_names(root, bindings)
            bindings.write_text('0 cache_v2 0 1\n1 cache_v2 1 0\n')
            module.validate_binding_names(root, bindings)

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
