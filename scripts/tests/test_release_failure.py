import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('release_failure',Path(__file__).parents[1]/'qualify-program-release-failure.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class ReleaseFailure(unittest.TestCase):
    def test_requires_error_and_forbids_success_even_if_error_follows(self):
        error={'error':{'message':'release failed'}}
        module.validate_failure([{'choices':[{'text':'partial','finish_reason':None}]},error])
        for events in ([],[{'choices':[]}],[{'choices':[{'finish_reason':'stop'}]},error],
                       [error,{'choices':[{'finish_reason':'length'}]}]):
            with self.assertRaises(ValueError):module.validate_failure(events)

    def test_requires_specific_failure_and_drained_logical_pages(self):
        failure='program worker failed: Io("File exists (os error 17)")\n'
        drained='shared state pool admissions=2 remaining_occupied_bytes=0\n'
        module.validate_fault_log(failure+drained)
        for log in (failure, drained, failure+drained.replace('bytes=0','bytes=15458304'),
                    failure.replace('File exists (os error 17)','unrelated error')+drained):
            with self.assertRaises(ValueError):module.validate_fault_log(log)
