import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('bind_arena', Path(__file__).resolve().parents[1] / 'bind-prepared-arena.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CapacityBindings(unittest.TestCase):
    def test_live_sampled_slot_has_equal_capacity_alternative(self):
        arena = {'slots': [{'id': i, 'bytes': n} for i, n in enumerate([128, 64, 128, 128, 256])],
                 'instruction_slots': {'-1': [0, 1], '472': [0, 2], '471': [3], '9': [4]}}
        result = module.capacity_bindings(arena, 0)
        self.assertEqual(result['472'], [0, 2, 3])
        self.assertEqual(result['471'], [2, 3])  # Never borrow workspace.
        self.assertEqual(result['9'], [4])     # Never widen capacity.
        free = {1, 3, 4}  # sampled slot 2 is live
        eligible = free.intersection(result['472'])
        self.assertEqual(eligible, {3})
        self.assertEqual(arena['instruction_slots']['472'], [0, 2])
        self.assertEqual(module.capacity_bindings({**arena, 'instruction_slots': result}, 0), result)

    def test_no_workspace_and_deterministic_order(self):
        slots = [{'id': 0, 'bytes': 64}, {'id': 1, 'bytes': 64}]
        arena = {'slots': slots, 'instruction_slots': {'2': [1]}}
        self.assertEqual(module.capacity_bindings(arena, None), {'2': [0, 1]})
        self.assertEqual(module.capacity_bindings({**arena, 'slots': slots[::-1]}, None), {'2': [0, 1]})


if __name__ == '__main__':
    unittest.main()
