import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('fixture', Path(__file__).resolve().parents[1] / 'prepare-sequence-fixture.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SequenceFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = self.root / 'plan'
        self.plan.mkdir()
        (self.plan / 'device-program.json').write_text(json.dumps(dict(eos=99, max_context=32,
            instructions=[dict(logical_node=7, op='index_cache_append', name='index.cache')])))
        (self.plan / 'serving.json').write_text(json.dumps(dict(states=[dict(logical_node=7,
            ranks=[0, 1], capacity_tokens=256, storage=dict(kind='pages', page_tokens=64))])))
        self.refs = self.root / 'reference.json'
        self.refs.write_text(json.dumps([dict(case='needle', input_ids=[1, 2, 3], output_ids=[4, 99, 5])]))

    def test_rejects_diagnostic_intervention_as_oracle(self):
        cases = json.loads(self.refs.read_text())
        cases[0]['reference_intervention'] = 'changed rotary coefficients'
        self.refs.write_text(json.dumps(cases))
        with self.assertRaisesRegex(ValueError, 'not an independent reference'):
            module.prepare(self.plan, self.refs, self.root / 'bad-intervention')
        self.assertFalse((self.root / 'bad-intervention').exists())

    def test_deterministic_scope_and_independent_page_permutations(self):
        first, second = self.root / 'one', self.root / 'two'
        self.assertEqual(module.prepare(self.plan, self.refs, first)['expected_tokens'], 2)
        module.prepare(self.plan, self.refs, second)
        for path in first.iterdir():
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
        self.assertEqual((first / 'requests.txt').read_text(), '2 1 2 3\n')
        self.assertEqual([json.loads(l)['token'] for l in (first / 'reference.jsonl').read_text().splitlines()], [4, 99])
        for line in (first / 'page-bindings.txt').read_text().splitlines():
            self.assertEqual(sorted(map(int, line.split()[2:])), [0, 1, 2, 3])
        with self.assertRaises(FileExistsError):
            module.prepare(self.plan, self.refs, first)

    def test_named_peer_and_reject_placeholder_before_writing(self):
        peers = self.root / 'peer.json'
        peers.write_text(json.dumps([dict(case='short', input_ids=[6], output_ids=[7])]))
        out = self.root / 'mixed'
        result = module.prepare(self.plan, self.refs, out, peers, 'short')
        self.assertEqual((result['requests'], result['expected_tokens']), (2, 3))
        self.assertEqual((out / 'requests.txt').read_text(), '2 1 2 3\n1 6\n')
        self.refs.write_text(json.dumps([dict(case='bad', input_ids=[1], output_ids=[2], qualification=dict(output_is_placeholder=True))]))
        with self.assertRaises(ValueError):
            module.prepare(self.plan, self.refs, self.root / 'bad')
        self.assertFalse((self.root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
