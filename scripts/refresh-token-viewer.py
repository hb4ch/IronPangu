#!/usr/bin/env python3
"""Refresh an offline token viewer's presentation, preserving its embedded plan and arena."""
import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
match = re.search(r'<script[^>]*id="plan-data"[^>]*>(.*?)</script>', args.input.read_text(encoding='utf-8'), re.S)
if not match:
    raise ValueError('input has no embedded plan')
data = json.loads(match.group(1))
for key in ('graph', 'program', 'recipes', 'physical', 'parallel'):
    if key not in data:
        raise ValueError('missing plan metadata: ' + key)
payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
template = (Path(__file__).resolve().parents[1] / 'crates/inferfabric-compiler/src/token_view.html').read_text(encoding='utf-8')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(template.replace('__PLAN_DATA__', payload), encoding='utf-8')
