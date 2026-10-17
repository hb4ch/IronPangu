import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('allocations',Path(__file__).parents[1]/'summarize-host-allocations.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class AllocationSummary(unittest.TestCase):
    def test_live_growth_does_not_count_freed_sample(self):
        a,b=m.parse('snapshot 1 rate=1024 dropped=0\nallocation 1 64 lib+0x1\nend\n'
                    'snapshot 2 rate=1024 dropped=0\nallocation 2 128 lib+0x2\nend\n')
        report=m.compare(a,b)
        self.assertEqual(report['sampled_byte_growth'],64)
        self.assertEqual([r['byte_growth'] for r in report['stacks']],[128,-64])

    def test_incomplete_or_dropped_samples_are_not_evidence(self):
        for text in ('snapshot 1 rate=1 dropped=1\nend\n',
                     'snapshot 1 rate=1 dropped=0\nallocation 1 3 a\n',
                     'snapshot 1 rate=1 dropped=0\nallocation 1 3 a\nallocation 1 4 a\nend\n'):
            with self.assertRaises(ValueError):m.parse(text)
