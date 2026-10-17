"""Offline numerical intervention only; never used by native serving."""
import json
import os
from pathlib import Path


def install():
    directory = os.environ.get('INFERFABRIC_REFERENCE_ROTARY_OVERRIDE')
    if directory is None:
        return
    import torch
    from omni_npu.layers.rotary_embedding.deepseek_scaling_rope import NPUDeepseekScalingRotaryEmbedding
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    cache = {}
    original = NPUDeepseekScalingRotaryEmbedding._set_cos_sin_cache

    def set_cache(self, seq_len, device, dtype):
        original(self, seq_len, device, dtype)
        if (not self.is_neox_style or self.rotary_dim != manifest['dimension']
                or self.base != manifest['base'] or seq_len > manifest['positions']
                or self.scaling_factor != 1 or self.mscale != 1):
            raise ValueError('diagnostic rotary override does not match unscaled model')
        key = (str(device), dtype)
        if key not in cache:
            size = manifest['positions'] * manifest['dimension']
            values = []
            for name in ('cos', 'sin'):
                path = root / (name + '.bin')
                if path.stat().st_size != size * 2:
                    raise ValueError('diagnostic rotary file extent mismatch')
                values.append(torch.from_file(str(path), size=size, dtype=torch.int16)
                              .view(torch.bfloat16).reshape(manifest['positions'], manifest['dimension'])
                              .to(device=device, dtype=dtype))
            cache[key] = values
        cos, sin = cache[key]
        self.register_buffer('cos_cached', cos[:seq_len], persistent=False)
        self.register_buffer('sin_cached', sin[:seq_len], persistent=False)
        # Some callers use the base class cache rather than CachedCosSinMixin.
        half = self.rotary_dim // 2
        combined = torch.cat((cos[:seq_len, :half], sin[:seq_len, :half]), dim=-1)
        self.register_buffer('cos_sin_cache', combined.to(self.cos_sin_cache.dtype), persistent=False)
        with (root / ('applied-' + str(os.getpid()) + '.jsonl')).open('a') as out:
            out.write(json.dumps(dict(pid=os.getpid(), cls=type(self).__name__, positions=seq_len,
                                      dimension=self.rotary_dim, device=str(cos.device))) + '\n')

    NPUDeepseekScalingRotaryEmbedding._set_cos_sin_cache = set_cache
