#!/usr/bin/env python3
"""Portable regression tests for discovery -> workspace plan -> arena binding."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner, binder = load('plan-shared-workspace'), load('bind-prepared-arena')


class WorkspacePlanning(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'execution.bin').write_bytes(b'program')
        (self.source / 'rank-0.bin').write_bytes(b'initialization')
        (self.source / 'execution.html').write_text('<script type="application/json" id="plan-data">{}</script>')
        self.fingerprint = binder.fingerprint(b'program')
        self.audit = {'steps': [{'position': 0}], 'graphs': [{'replays': 1}],
            'workspace': {'planned': False, 'planned_bytes': 0, 'peak_requested_bytes': 1024,
                          'growth_allocations': 2, 'slot': 0, 'instruction_requests': {'2': 512, '7': 1024}},
            'arena': {'format': 'inferfabric.frozen-arena.v1', 'program_fingerprint': self.fingerprint,
                      'backend_fingerprint': '456', 'bytes': 1024,
                      'slots': [{'id': 0, 'offset': 0, 'bytes': 1024}],
                      'instruction_slots': {'-1': [0], '2': [0], '7': [0]}}}
        self.audits = self.root / 'audits'
        self.audits.mkdir()
        self.write()

    def write(self, value=None, folder=None):
        ((folder or self.audits) / 'rank-0.json').write_text(json.dumps(value or self.audit))

    def planned(self):
        output = self.root / 'planned'
        planner.plan(self.source, output, [self.audits])
        self.audit['workspace'].update(planned=True, planned_bytes=1024, growth_allocations=0)
        self.write()
        return output

    def test_maximum_across_discovery_runs_then_binding(self):
        other = self.root / 'other';other.mkdir()
        second = copy.deepcopy(self.audit)
        second['workspace']['peak_requested_bytes'] = 2048
        second['workspace']['instruction_requests']['7'] = 2048
        self.write(second, other)
        output = self.root / 'planned'
        planner.plan(self.source, output, [self.audits, other])
        self.assertTrue((output / 'workspace-rank-0.txt').read_text().endswith('2048\n'))
        self.audit['workspace'].update(planned=True, planned_bytes=2048, growth_allocations=0)
        self.audit['arena']['bytes'] = 2048
        self.audit['arena']['slots'][0]['bytes'] = 2048
        self.write()
        binder.bind(output, self.audits, self.root / 'bound')
        metadata = json.loads((self.root / 'bound/arena.json').read_text())
        self.assertEqual(metadata['ranks'][0]['workspace']['planned_bytes'], 2048)
        self.assertTrue((self.root / 'bound/arena-rank-0.txt').read_text().startswith('IFARENA2 '))
        self.assertEqual(metadata['ranks'][0]['instruction_envelopes']['7'],
                         dict(max_request_bytes=0, workspace=True))

    def test_rebinding_updates_existing_viewer_contract(self):
        source = self.planned()
        (source / 'execution.html').write_text('<script type="application/json" id="plan-data">{}</script>\n'
            '    function memoryDetails(id) { return "old eligibility"; }\n'
            '    function inspect(id) {}\n')
        binder.bind(source, self.audits, self.root / 'bound')
        html = (self.root / 'bound/execution.html').read_text()
        self.assertNotIn('old eligibility', html)
        self.assertIn('arena.instruction_envelopes', html)
        self.assertIn('Maximum request:', html)
        self.assertIn('Observed slot IDs:', html)

    def test_discovery_is_not_bindable(self):
        with self.assertRaises(ValueError):
            binder.bind(self.source, self.audits, self.root / 'bad')

    def test_discovery_requires_matching_program_backend_and_capture(self):
        for mutation in [lambda a:a['arena'].update(program_fingerprint='wrong'),
                         lambda a:a.update(graphs=[]),
                         lambda a:a['workspace'].update(peak_requested_bytes=1)]:
            bad=copy.deepcopy(self.audit);mutation(bad);self.write(bad)
            with self.assertRaises(ValueError):
                planner.plan(self.source, self.root / 'bad', [self.audits])
        self.write()
        other=self.root / 'other';other.mkdir()
        bad=copy.deepcopy(self.audit);bad['arena']['backend_fingerprint']='789';self.write(bad,other)
        with self.assertRaises(ValueError):
            planner.plan(self.source,self.root / 'bad',[self.audits,other])

    def test_binding_rejects_growth_overflow_slot_and_missing_dependency(self):
        source=self.planned()
        for mutation in [lambda a:a['workspace'].update(growth_allocations=1),
                         lambda a:a['workspace'].update(peak_requested_bytes=1025),
                         lambda a:a['workspace'].update(slot=1),
                         lambda a:a['arena']['instruction_slots'].update({'7': []})]:
            bad=copy.deepcopy(self.audit);mutation(bad);self.write(bad)
            with self.assertRaises(ValueError):
                binder.bind(source,self.audits,self.root / 'bad')


if __name__ == '__main__':
    unittest.main()
