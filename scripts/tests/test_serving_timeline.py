import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('timeline', Path(__file__).parents[1] / 'summarize-serving-timeline.py')
timeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timeline)


class TimelineTest(unittest.TestCase):
    def test_overlap_is_not_counted_as_additional_latency(self):
        self.assertEqual(timeline.union_us([(5, 10), (0, 6), (1, 2), (12, 14)]), 12)
        self.assertEqual(timeline.union_us([]), 0)
        self.assertEqual(timeline.union_us([(0, 1), (1, 2), (2, 2)]), 2)

    def test_invalid_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            timeline.union_us([(2, 1)])


if __name__ == '__main__':
    unittest.main()
