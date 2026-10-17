import importlib.util
from pathlib import Path
import unittest
import copy

spec=importlib.util.spec_from_file_location('http_qualify',Path(__file__).parents[1]/'qualify-program-http.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class HttpReferenceScope(unittest.TestCase):
    def test_long_context_usage_cannot_hide_truncation_or_missing_sse_usage(self):
        case={'input_ids':[1]*65520,'output_ids':[2,3,99]+[4]*13}
        usage={'prompt_tokens':65520,'completion_tokens':3,'total_tokens':65523}
        module.validate_usage(case,{'usage':usage},99,True)
        for key in usage:
            truncated=dict(usage)
            truncated[key]-=1
            with self.assertRaises(ValueError):
                module.validate_usage(case,{'usage':truncated},99,True)
        for missing in [{},{'usage':None},{'usage':{}}]:
            with self.assertRaises(ValueError):module.validate_usage(case,missing,99,True)
        with self.assertRaises(ValueError):
            module.validate_usage(case,{'usage':usage},99,False)
        full={'prompt_tokens':65520,'completion_tokens':16,'total_tokens':65536}
        module.validate_usage(case,{'usage':full},99,False)

    def test_completed_history_must_account_for_every_declared_invocation(self):
        manifest={'resident_sequences':2,'variants':[{'numerical_class':'packed4','max_query_tokens':4,'max_tokens':4,'max_sequences':1,'max_context':65536}]}
        record={'format':'inferfabric.completed-arithmetic.v1','sequence':1,'resident':0,'history':{'version':1,'spans':[{'start':0,'end':64,'query_tokens':4,'numerical_class':'packed4'}]}}
        self.assertTrue(module.validate_histories([record],manifest,64)['all_scheduler_tokens_accounted'])
        for defect in range(7):
            bad=copy.deepcopy(record)
            if defect==0:bad['resident']=2
            elif defect==1:bad['history']['spans'][0]['start']=4
            elif defect==2:bad['history']['spans'][0]['end']=63
            elif defect==3:bad['history']['spans'][0]['numerical_class']='other'
            elif defect==4:bad['sequence']=2
            elif defect==5:bad['history']['spans']=[]
            else:bad['history']['version']=2
            with self.assertRaises(ValueError):module.validate_histories([bad],manifest,64)
        with self.assertRaises(ValueError):module.validate_histories([record,record],manifest,128)
        with self.assertRaises(ValueError):module.validate_histories([record],manifest,68)

    def test_stop_mode_preserves_full_reference_and_counts_eos(self):
        case={'output_ids':[1,2,99,3,4]}
        self.assertEqual(module.expected_ids(case,99,True),[1,2,99])
        self.assertEqual(module.expected_ids(case,99,False),[1,2,99,3,4])
        self.assertEqual(case['output_ids'],[1,2,99,3,4])

    def test_imported_prefix_is_not_new_scheduler_work(self):
        manifest={'resident_sequences':2,'variants':[{'numerical_class':'packed4','max_query_tokens':4,'max_tokens':4,'max_sequences':1,'max_context':65536}]}
        record={'format':'inferfabric.completed-arithmetic.v1','sequence':1,'resident':1,'imported_tokens':64,'history':{'version':1,'spans':[{'start':0,'end':80,'query_tokens':4,'numerical_class':'packed4'}]}}
        self.assertEqual(module.validate_histories([record],manifest,16)['completed_input_tokens'],16)
        for imported in [-1,63,81]:
            bad=copy.deepcopy(record);bad['imported_tokens']=imported
            with self.assertRaises(ValueError):module.validate_histories([bad],manifest,80-imported)

    def test_no_eos_and_first_token_eos(self):
        self.assertEqual(module.expected_ids({'output_ids':[1,2]},99,True),[1,2])
        self.assertEqual(module.expected_ids({'output_ids':[99,99,2]},99,True),[99])

    def test_placeholders_and_empty_references_fail_in_both_modes(self):
        for mode in [False,True]:
            for case in [{'output_ids':[]},{'output_ids':[0],'qualification':{'output_is_placeholder':True}}]:
                with self.assertRaises(ValueError):module.expected_ids(case,99,mode)

if __name__=='__main__':unittest.main()
