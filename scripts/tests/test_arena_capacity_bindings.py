import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('bind_arena', Path(__file__).resolve().parents[1] / 'bind-prepared-arena.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RequestEnvelopes(unittest.TestCase):
    def test_request_ceiling_excludes_workspace(self):
        arena = {'slots': [{'id': i, 'bytes': n} for i, n in enumerate([159221248, 27648, 8388608, 25165824])],
                 'instruction_slots': {'-1': [0, 3], '8': [0, 1, 2], '9': [2], '10': [0]}}
        result = module.request_envelopes(arena, 0)
        self.assertEqual(result['8'], dict(max_request_bytes=8388608, workspace=True))
        self.assertEqual(result['9'], dict(max_request_bytes=8388608, workspace=False))
        self.assertEqual(result['10'], dict(max_request_bytes=0, workspace=True))
        self.assertEqual(arena['instruction_slots']['8'], [0, 1, 2])
        # Request size fits although its free backing slot is larger than the ceiling.
        self.assertLessEqual(1083392, result['8']['max_request_bytes'])
        self.assertGreater(arena['slots'][3]['bytes'], result['8']['max_request_bytes'])

    def test_no_workspace_and_deterministic_order(self):
        slots = [{'id': 0, 'bytes': 64}, {'id': 1, 'bytes': 128}]
        arena = {'slots': slots, 'instruction_slots': {'2': [1]}}
        expected = {'2': dict(max_request_bytes=128, workspace=False)}
        self.assertEqual(module.request_envelopes(arena, None), expected)
        self.assertEqual(module.request_envelopes({**arena, 'slots': slots[::-1]}, None), expected)


if __name__ == '__main__':
    unittest.main()
