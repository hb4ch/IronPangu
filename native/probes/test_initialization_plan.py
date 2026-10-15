#!/usr/bin/env python3
"""Portable wire-format adversarial tests; executable performs no ACL initialization."""
import pathlib
import struct
import subprocess
import sys
import tempfile

def u32(n): return struct.pack('<I',n)
def u64(n): return struct.pack('<Q',n)
def text(s):
    b=s.encode()
    return u32(len(b))+b

def plan(dtype=2, source=2, shape=4, size=8, destination=0, opcode=1, shard='weights', offset=8, name='weight', parameter_id=0):
    return (b'IFINIT01'+u32(17)+u32(22)+text('a'*64)+u32(1)
            +u32(parameter_id)+text(name)+u32(dtype)+u32(1)+u64(shape)+u32(1)
            +u32(opcode)+u32(destination)+text(shard)+u64(offset)+u64(size)+u32(source))

cases=[('copy',plan(),True),('cast',plan(dtype=3),True),
       ('truncated',plan()[:-1],False),('trailing',plan()+b'x',False),
       ('old-format',b'IFLOAD01'+plan()[8:],False),('opcode',plan(opcode=99),False),
       ('destination',plan(destination=1),False),('parameter-id',plan(parameter_id=1),False),
       ('dtype',plan(dtype=99),False),('conversion',plan(dtype=1),False),
       ('shape',plan(shape=5),False),('zero',plan(shape=0),False),
       ('overflow',plan(shape=2**63),False),('offset-overflow',plan(offset=2**64-4),False),
       ('path',plan(shard='../weights'),False),('windows-path',plan(shard='C:weights'),False),
       ('nul',plan(name='a\0b'),False)]
binary=str(pathlib.Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory() as directory:
    for name,data,success in cases:
        path=pathlib.Path(directory)/(name+'.bin');path.write_bytes(data)
        result=subprocess.run([binary,'--validate',str(path)],capture_output=True,timeout=10)
        if (result.returncode==0)!=success:
            raise RuntimeError(f'{name}: {result.stderr!r}')
print(f'{len(cases)} initialization wire-format cases passed')
