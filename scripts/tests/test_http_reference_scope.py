import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('http_qualify',Path(__file__).parents[1]/'qualify-program-http.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class HttpReferenceScope(unittest.TestCase):
    def test_stop_mode_preserves_full_reference_and_counts_eos(self):
        case={'output_ids':[1,2,99,3,4]}
        self.assertEqual(module.expected_ids(case,99,True),[1,2,99])
        self.assertEqual(module.expected_ids(case,99,False),[1,2,99,3,4])
        self.assertEqual(case['output_ids'],[1,2,99,3,4])

    def test_no_eos_and_first_token_eos(self):
        self.assertEqual(module.expected_ids({'output_ids':[1,2]},99,True),[1,2])
        self.assertEqual(module.expected_ids({'output_ids':[99,99,2]},99,True),[99])

    def test_placeholders_and_empty_references_fail_in_both_modes(self):
        for mode in [False,True]:
            for case in [{'output_ids':[]},{'output_ids':[0],'qualification':{'output_is_placeholder':True}}]:
                with self.assertRaises(ValueError):module.expected_ids(case,99,mode)

if __name__=='__main__':unittest.main()
