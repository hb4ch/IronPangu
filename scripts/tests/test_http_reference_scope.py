import importlib.util
from pathlib import Path
import unittest
import copy
import tempfile

spec=importlib.util.spec_from_file_location('http_qualify',Path(__file__).parents[1]/'qualify-program-http.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class HttpReferenceScope(unittest.TestCase):
    def test_allocator_snapshot_rejects_dead_server_and_stale_ack(self):
        class Server:
            def __init__(self,code):self.code=code
            def poll(self):return self.code
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stacks'
            Path(str(path)+'.ack').write_text('2 12345\n')
            self.assertEqual(module.allocator_snapshot(path,2,Server(None)),12345)
            with self.assertRaises(RuntimeError):module.allocator_snapshot(path,2,Server(1))
            with self.assertRaises(RuntimeError):module.allocator_snapshot(path,3,Server(None),timeout=0)

    def test_resource_gate_detects_transient_growth_not_only_final_sample(self):
        base=dict(rss_bytes=1024,peak_rss_bytes=4096,threads=8,file_descriptors=10)
        self.assertTrue(module.resource_growth([base,base],0)['passed'])
        for key in ('rss_bytes','threads','file_descriptors'):
            high={**base,key:base[key]+1}
            self.assertFalse(module.resource_growth([base,high,base],0)['passed'])
        self.assertTrue(module.resource_growth([base,{**base,'rss_bytes':2048}],1)['passed'])
        for limit in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):module.resource_growth([base,base],limit)

    def test_process_resources_requires_live_complete_linux_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);process=root/'123';process.mkdir();(process/'fd').mkdir()
            (process/'fd'/'0').touch()
            (process/'status').write_text('Name:\tserver\nVmRSS:\t12 kB\nVmHWM:\t30 kB\nThreads:\t8\n')
            self.assertEqual(module.process_resources(123,root),dict(rss_bytes=12288,peak_rss_bytes=30720,threads=8,file_descriptors=1))
            (process/'status').write_text('Threads:\t8\n')
            with self.assertRaises(ValueError):module.process_resources(123,root)

    def test_prefix_boundary_matches_full_chunk_checkpoint_contract(self):
        self.assertEqual(module.prefix_boundary(8176,64,4096),4096)
        self.assertEqual(module.prefix_boundary(65520,64,4096),61440)
        self.assertEqual(module.prefix_boundary(4096,64,4096),0)
        self.assertEqual(module.prefix_boundary(4097,64,2048),4096)
        self.assertEqual(module.prefix_boundary(8176,64),8128)
        for args in [(0,64,4096),(12,0,4096),(12,64,0)]:
            with self.assertRaises(ValueError):module.prefix_boundary(*args)

    def test_mixed_progress_uses_first_output_not_completion_order(self):
        long=dict(first_text_ms_since_run=20000)
        short=dict(request_started_ms_since_run=500, first_text_ms_since_run=7000, total_ms=30000)
        self.assertTrue(module.validate_mixed_progress(16368,4096,long,short))
        with self.assertRaises(ValueError):
            module.validate_mixed_progress(16368,4096,long,{**short,'first_text_ms_since_run':21000})
        with self.assertRaises(ValueError):
            module.validate_mixed_progress(16368,4096,long,{**short,'request_started_ms_since_run':21000})
        self.assertFalse(module.validate_mixed_progress(4088,4096,long,{**short,'first_text_ms_since_run':21000}))

    def test_batch_history_keeps_exact_identity_and_accepts_actual_tail(self):
        manifest={'resident_sequences':2,'variants':[{'numerical_class':'extend','max_query_tokens':4096,'max_tokens':4096,
            'max_sequences':1,'max_context':65536,'phase':'prefill','packed':True,'graph_capable':False}]}
        identity='extend:sequence-batch-v1:'+'a'*64+':member=1'
        record={'format':'inferfabric.completed-arithmetic.v1','sequence':1,'resident':0,
            'history':{'version':1,'spans':[{'start':0,'end':17,'query_tokens':17,'numerical_class':identity}]}}
        self.assertTrue(module.validate_histories([record],manifest,17)['all_scheduler_tokens_accounted'])
        for bad_class in [identity.replace('member=1','member=2'),identity.replace('a'*64,'x'*64),identity.replace('extend:','unknown:')]:
            bad=copy.deepcopy(record);bad['history']['spans'][0]['numerical_class']=bad_class
            with self.assertRaises(ValueError):module.validate_histories([bad],manifest,17)
        bad=copy.deepcopy(record);bad['history']['spans'][0].update(query_tokens=4097,end=4097)
        with self.assertRaises(ValueError):module.validate_histories([bad],manifest,4097)
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
