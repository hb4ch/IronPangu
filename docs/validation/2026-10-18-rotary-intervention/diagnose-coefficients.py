import ctypes,json,torch,torch_npu
lib=ctypes.CDLL('libm.so.6')
for name,nargs in [('powf',2),('cosf',1),('sinf',1)]:
 f=getattr(lib,name);f.argtypes=[ctypes.c_float]*nargs;f.restype=ctypes.c_float
positions=[0,4095,8191,32767,65407,65414,65535]
f32=lambda x:ctypes.c_float(x).value
inv=1.0/(6400000.0**(torch.arange(0,64,2,device='npu',dtype=torch.float32)/64))
freq=torch.einsum('i,j->ij',torch.tensor(positions,device='npu',dtype=torch.float32),inv)
reference=torch.cat((torch.cos(freq),torch.sin(freq)),dim=-1).to(torch.bfloat16).cpu().float()
rows=[]
for reciprocal in [False,True]:
 values=[]
 for pos in positions:
  angles=[]
  for i in range(32):
   power=lib.powf(6400000.,f32(2*i/64))
   angles.append(f32(f32(pos)*f32(1./power)) if reciprocal else f32(f32(pos)/power))
  values.append([lib.cosf(a) for a in angles]+[lib.sinf(a) for a in angles])
 native=torch.tensor(values,dtype=torch.float32).to(torch.bfloat16).float()
 diff=(reference-native).abs()
 rows.append(dict(mode='reciprocal_multiply' if reciprocal else 'native_divide',positions=[dict(position=p,differing=int((diff[i]!=0).sum()),max_abs=float(diff[i].max())) for i,p in enumerate(positions)]))
print(json.dumps(dict(observations=rows,reference_inv_freq=inv.cpu().tolist(),native_inv_freq=[f32(1/lib.powf(6400000.,f32(2*i/64))) for i in range(32)]),indent=2))
