#!/usr/bin/env python3
"""Malformed binary manifest tests. No model execution or tensor framework."""
import pathlib
import struct
import subprocess
import sys
import tempfile

binary = str(pathlib.Path(sys.argv[1]).resolve())

def string(value):
    b = value.encode()
    return struct.pack('<I', len(b)) + b

def manifest(name='weight', shard='model.safetensors', dtype='BF16',
             shape=4, size=8, total=8, prepared=8, offset=8, transform='rms_norm'):
    return (b'IFLOAD01' + struct.pack('<IIIQQ', 0, 0, 1, total, prepared)
            + string('a' * 64) + string(name) + string(shard) + string(dtype)
            + string(transform) + struct.pack('<QQIQ', offset, size, 1, shape))

cases = [
    ('valid', manifest(), True),
    ('cast', manifest(transform='cast_fp32', prepared=16), True),
    ('truncated', manifest()[:-1], False),
    ('trailing', manifest() + b'x', False),
    ('version', b'INVALID!' + manifest()[8:], False),
    ('path', manifest(shard='../model.safetensors'), False),
    ('nul', manifest(name='a\0b'), False),
    ('dtype', manifest(dtype='F64'), False),
    ('shape', manifest(shape=5), False),
    ('total', manifest(total=9), False),
    ('cast-total', manifest(transform='cast_fp32'), False),
    ('offset-overflow', manifest(offset=2**64-4), False),
    ('zero', manifest(shape=0), False),
]
with tempfile.TemporaryDirectory() as directory:
    for name, data, success in cases:
        path = pathlib.Path(directory) / (name + '.bin')
        path.write_bytes(data)
        result = subprocess.run([binary, '--validate', str(path)], capture_output=True, timeout=10)
        if (result.returncode == 0) != success:
            raise RuntimeError(f'{name}: unexpected status {result.returncode}: {result.stderr!r}')
print(f'{len(cases)} native load-plan validation cases passed')
