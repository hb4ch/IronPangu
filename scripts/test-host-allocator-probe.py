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
    parser.add_argument('--cann',action='store_true',help='Also verify CANN 9.1 pool-client ownership with no device selection')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        output=Path(directory)/'allocations.jsonl'
        env=dict(os.environ,LD_PRELOAD=str(args.library.resolve(strict=True)),
                 INFERFABRIC_HOST_ALLOCATOR_EXECUTABLE=str(Path(sys.executable).resolve()),
                 INFERFABRIC_HOST_ALLOCATOR_AUDIT=str(output))
        stacks=Path(directory)/'stacks.txt'
        env.update(INFERFABRIC_HOST_ALLOCATION_STACKS=str(stacks),INFERFABRIC_HOST_ALLOCATION_RATE='1')
        code=('import ctypes,time; c=ctypes.CDLL(None); '
              'c.malloc.restype=ctypes.c_void_p; c.free.argtypes=[ctypes.c_void_p]; '
              'c.realloc.argtypes=[ctypes.c_void_p,ctypes.c_size_t]; c.realloc.restype=ctypes.c_void_p; '
              'time.sleep(2.3); p=c.malloc(64*1024*1024); assert p; '
              'assert not c.realloc(p,ctypes.c_size_t(-1).value); '
              'time.sleep(2.3); c.free(p); time.sleep(2.3)')
        subprocess.run([sys.executable,'-c',code],env=env,check=True)
        rows=[json.loads(line) for line in output.read_text().splitlines()]
        values=[row['mmap_bytes'] for row in rows]
        if max(values)-values[0]<64*1024*1024 or values[-1]>=max(values)-60*1024*1024:
            raise RuntimeError('probe did not observe the known allocation and release')
        snapshots=stacks.read_text().split('snapshot ')[1:]
        live=[any(line.startswith('allocation ') and line.split()[2]==str(64*1024*1024)
                  for line in snapshot.splitlines()) for snapshot in snapshots]
        if not any(live) or live[-1] or any('dropped=0\n' not in s for s in snapshots):
            raise RuntimeError('sampled stacks lost failed-realloc ownership or retained a freed allocation')
        env.pop('INFERFABRIC_HOST_ALLOCATION_STACKS')
        original=output.read_bytes()
        subprocess.run(['true'],env=env,check=True)
        result=subprocess.run([sys.executable,'-c','pass'],env=env,capture_output=True)
        if result.returncode!=125 or output.read_bytes()!=original:
            raise RuntimeError('probe did not reject an existing output without overwriting it')
        env['INFERFABRIC_HOST_ALLOCATOR_AUDIT']=str(Path(directory)/'signals.jsonl')
        env['INFERFABRIC_HOST_ALLOCATION_STACKS']=str(Path(directory)/'signal-stacks.txt')
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
            request=Path(env['INFERFABRIC_HOST_ALLOCATION_STACKS']+'.request')
            ack=Path(env['INFERFABRIC_HOST_ALLOCATION_STACKS']+'.ack')
            request.write_text('1\n')
            deadline=time.monotonic()+10
            while not ack.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(0.02)
            fields=ack.read_text().split()
            if fields[0]!='1' or f'snapshot {fields[1]} ' not in Path(env['INFERFABRIC_HOST_ALLOCATION_STACKS']).read_text():
                raise RuntimeError('snapshot acknowledgement does not identify a complete snapshot')
            child.send_signal(signal.SIGTERM)
            if child.wait(timeout=10)!=0:raise RuntimeError('probe intercepted frontend sigwait shutdown')
        finally:
            if child.poll() is None:child.kill();child.wait()
        if args.cann:
            stacks=Path(directory)/'cann-stacks.txt'
            env.update(INFERFABRIC_HOST_ALLOCATOR_AUDIT=str(Path(directory)/'cann.jsonl'),
                       INFERFABRIC_HOST_ALLOCATION_STACKS=str(stacks),
                       INFERFABRIC_HOST_ALLOCATION_START=str(Path(directory)/'cann-start'))
            code=('import ctypes,time,pathlib,sys; '
                  'acl=ctypes.CDLL("libascendcl.so",mode=ctypes.RTLD_GLOBAL); assert acl.aclInit(None)==0; '
                  'pool=ctypes.CDLL("libnnopbase.so",mode=ctypes.RTLD_GLOBAL); api=ctypes.CDLL(None); '
                  'alloc=api._ZN2op8internal8AllocateEm; alloc.argtypes=[ctypes.c_size_t]; alloc.restype=ctypes.c_void_p; '
                  'free=api._ZN2op8internal10DeAllocateEPv; free.argtypes=[ctypes.c_void_p]; '
                  'pathlib.Path(sys.argv[1]).touch(); '
                  'time.sleep(2.3); p=alloc(4093); assert p; time.sleep(2.3); free(p); time.sleep(2.3); '
                  'assert acl.aclFinalize()==0')
            subprocess.run([sys.executable,'-c',code,env['INFERFABRIC_HOST_ALLOCATION_START']],env=env,check=True)
            snapshots=stacks.read_text().split('snapshot ')[1:]
            live=[any(line.startswith('allocation ') and line.split()[2:4]==['4093','kind=cann-client']
                      for line in s.splitlines()) for s in snapshots]
            if not any(live) or live[-1] or any('dropped=0\n' not in s for s in snapshots):
                raise RuntimeError('CANN pool-client allocate/free ownership was not observed')
        print('PASS: allocation/release, output ownership, helper isolation and sigwait shutdown')


if __name__=='__main__':
    main()
