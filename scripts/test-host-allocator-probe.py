#!/usr/bin/env python3
"""Run inside the Linux container after compiling host_allocator_probe.cpp."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import signal
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library',type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        output=Path(directory)/'allocations.jsonl'
        env=dict(os.environ,LD_PRELOAD=str(args.library.resolve(strict=True)),
                 INFERFABRIC_HOST_ALLOCATOR_EXECUTABLE=str(Path(sys.executable).resolve()),
                 INFERFABRIC_HOST_ALLOCATOR_AUDIT=str(output))
        code=('import ctypes,time; c=ctypes.CDLL(None); '
              'c.malloc.restype=ctypes.c_void_p; c.free.argtypes=[ctypes.c_void_p]; '
              'time.sleep(2.3); p=c.malloc(64*1024*1024); assert p; '
              'time.sleep(2.3); c.free(p); time.sleep(2.3)')
        subprocess.run([sys.executable,'-c',code],env=env,check=True)
        rows=[json.loads(line) for line in output.read_text().splitlines()]
        values=[row['mmap_bytes'] for row in rows]
        if max(values)-values[0]<64*1024*1024 or values[-1]>=max(values)-60*1024*1024:
            raise RuntimeError('probe did not observe the known allocation and release')
        original=output.read_bytes()
        subprocess.run(['true'],env=env,check=True)
        result=subprocess.run([sys.executable,'-c','pass'],env=env,capture_output=True)
        if result.returncode!=125 or output.read_bytes()!=original:
            raise RuntimeError('probe did not reject an existing output without overwriting it')
        env['INFERFABRIC_HOST_ALLOCATOR_AUDIT']=str(Path(directory)/'signals.jsonl')
        ready=Path(directory)/'ready'
        code=('import signal,pathlib,sys,time; '
              'signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT}); '
              'pathlib.Path(sys.argv[1]).touch(); '
              'time.sleep(0.3); '
              'assert signal.sigwait({signal.SIGTERM})==signal.SIGTERM')
        child=subprocess.Popen([sys.executable,'-c',code,str(ready)],env=env)
        try:
            deadline=time.monotonic()+10
            while not ready.exists() and child.poll() is None and time.monotonic()<deadline:
                time.sleep(0.02)
            if not ready.exists():raise RuntimeError('signal test did not become ready')
            child.send_signal(signal.SIGTERM)
            if child.wait(timeout=10)!=0:raise RuntimeError('probe intercepted frontend sigwait shutdown')
        finally:
            if child.poll() is None:child.kill();child.wait()
        print('PASS: allocation/release, output ownership, helper isolation and sigwait shutdown')


if __name__=='__main__':
    main()
