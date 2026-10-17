import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('batch_trace', Path(__file__).resolve().parents[1] / 'check-sequence-batch-trace.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BatchTrace(unittest.TestCase):
    def events(self):
        return [dict(event='sequence_batch', elapsed_us=100, tokens=n,
                     members=[dict(request=0, resident=0, position=p, tokens=n, sample=s)])
                for p, n, s in [(0, 4096, 0), (4096, 4080, 1), (8176, 1, 1)]]

    def test_decode_timings_are_grouped_by_request_context_and_batch_size(self):
        events = [dict(event='sequence_batch', elapsed_us=9999, tokens=8,
                       members=[dict(request=r, resident=r, position=0, tokens=4, sample=1) for r in range(2)])]
        for position, us in [(4, 10), (5, 30)]:
            events.append(dict(event='sequence_batch', elapsed_us=us, tokens=2,
                members=[dict(request=r, resident=r, position=position, tokens=1, sample=1) for r in range(2)]))
        events.append(dict(event='sequence_batch', elapsed_us=15, tokens=1,
                           members=[dict(request=1, resident=1, position=6, tokens=1, sample=1)]))
        result = module.validate(events, [(4, 3), (4, 4)])['decode_by_request']
        self.assertEqual(result[0]['by_batch_size']['2']['mean_us'], 20)
        self.assertEqual(result[0]['by_batch_size']['2']['p95_us'], 30)
        self.assertEqual(result[1]['by_batch_size']['1']['samples'], 1)
        self.assertEqual(result[1]['by_batch_size']['1']['mean_us'], 15)
        self.assertNotIn('1', result[0]['by_batch_size'])
        self.assertIsNone(module.duration_stats([]))

    def test_two_bulk_chunks_and_separate_decode_timing(self):
        result = module.validate(self.events(), [(8176, 2)])
        self.assertEqual(result['prefill_chunks'], [[4096, 4080]])
        self.assertEqual(result['pure_decode_batch_us'], [100])
        self.assertEqual(result['pure_decode_batches'], [dict(sequences=1, elapsed_us=100)])

    def test_64k_is_sixteen_forwards_and_mixed_tail_keeps_histories(self):
        events = [dict(event='sequence_batch', elapsed_us=100, tokens=4096,
                       members=[dict(request=0, resident=0, position=p, tokens=4096, sample=int(p == 61440))])
                  for p in range(0, 65536, 4096)]
        result = module.validate(events, [(65536, 1)])
        self.assertEqual(result['prefill_chunks'], [[4096] * 16])
        self.assertEqual(result['pure_decode_batch_us'], [])
        mixed = [dict(event='sequence_batch', elapsed_us=100, tokens=18,
                      members=[dict(request=r, resident=r, position=0, tokens=n, sample=1)
                               for r, n in enumerate([12, 6])])]
        self.assertEqual(module.validate(mixed, [(12, 1), (6, 1)])['prefill_chunks'], [[12], [6]])

    def test_reject_truncation_gaps_early_sample_and_fragmentation(self):
        original = self.events()
        variants = [original[:-1]]
        for key, value in [('position', 1), ('sample', 1), ('tokens', 1)]:
            bad = copy.deepcopy(original)
            bad[0]['members'][0][key] = value
            if key == 'tokens':
                bad[0]['tokens'] = value
            variants.append(bad)
        for events in variants:
            with self.assertRaises(ValueError):
                module.validate(events, [(8176, 2)])


if __name__ == '__main__':
    unittest.main()
