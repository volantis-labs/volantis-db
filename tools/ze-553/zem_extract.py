"""ZE-553: copy one tower's tensors out of a .zem bundle.

Usage: zem_extract.py <bundle.zem> <document|query> <out.safetensors>
Descriptor layout follows tools/ze-model tensor_descriptor():
u32 name len, name, u8 dtype (0 f32, 1 f16), u8 ndim, 2 pad,
ndim x u32 dims, u64 offset, u64 length, u64 xxh3 digest.
"""
import re, struct, sys
import numpy as np
import torch
from safetensors.torch import save_file

path, role, out = sys.argv[1], sys.argv[2], sys.argv[3]
raw = np.memmap(path, dtype=np.uint8, mode="r")
(header_len,) = struct.unpack_from("<I", raw, 12)
head = bytes(raw[:header_len])
tensors = {}
for m in re.finditer(rb"(?s)(....)(" + role.encode() + rb"/[\x21-\x7e]+?)(?=[\x00\x01][\x01-\x04]\x00\x00)", head):
    (n,) = struct.unpack("<I", m.group(1))
    name = m.group(2)
    if n != len(name):
        continue
    at = m.end()
    dtype, ndim = head[at], head[at + 1]
    dims = struct.unpack_from(f"<{ndim}I", head, at + 4)
    offset, length, _ = struct.unpack_from("<QQQ", head, at + 4 + 4 * ndim)
    np_dtype = {0: np.float32, 1: np.float16}[dtype]
    array = np.frombuffer(raw[offset:offset + length], dtype=np_dtype).reshape(dims)
    tensors[name.decode()[len(role) + 1:]] = torch.from_numpy(array.astype(np.float32))
print(role, len(tensors), "tensors", sum(t.numel() for t in tensors.values()), "params")
save_file(tensors, out, metadata={"format": "pt"})
