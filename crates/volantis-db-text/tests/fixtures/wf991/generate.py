#!/usr/bin/env python3
"""Rebuild the WF-991 tiny FP16 BERT pair and its independent float64 oracle.

Tooling only: torch and xxhash are existing ze-model tools, not Rust dependencies.
The weights are invented integer arithmetic, rounded once to IEEE binary16. The
oracle is a float64 PyTorch CPU forward pass over those rounded weights. No
trained model, network access or accuracy claim is involved.
"""
from pathlib import Path
import hashlib
import math
import struct
import subprocess

import torch
import xxhash

ROOT = Path(__file__).resolve().parent
U16 = lambda n: struct.pack('<H', n)
U32 = lambda n: struct.pack('<I', n)
U64 = lambda n: struct.pack('<Q', n)
F32 = lambda n: struct.pack('<f', n)
BYTES = lambda b: U32(len(b)) + b
TEXT = lambda s: BYTES(s.encode())
torch.set_num_threads(1)

HEADER = 16384
HIDDEN, HEADS, INTERMEDIATE, POSITIONS, TYPES = 32, 2, 64, 16, 2
VOCAB = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', 'copper', 'kite', '991', 'violet', 'spool']
VOCAB += [f'w{i}' for i in range(32 - len(VOCAB))]
# Two ragged rows: "copper kite 991" and "violet spool", padded with [PAD].
IDS = [[2, 4, 5, 6, 3], [2, 7, 8, 3, 0]]
MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
TOWERS = [
    # role, tensor prefix, model id, prompt prefix, layers, seed shift
    (0, 'document/', 'FAKE-WF991-D', '', 2, 11),
    (1, 'query/', 'FAKE-WF991-Q', 'query: ', 1, 5),
]


def weights(layers, shift):
    tensors = {}

    def add(name, shape, norm=False):
        # Integer arithmetic, then one rounding to binary16. Most values are
        # not exact in binary16, so the runtime must widen real halves.
        seed = sum((i + 1) * ord(c) for i, c in enumerate(name)) + shift
        values = [((seed + i * 7 + i * i * 3) % 97 - 48) / 61 for i in range(math.prod(shape))]
        if norm:
            values = [1 + v / 4 for v in values]
        tensors[name] = torch.tensor(values, dtype=torch.float64).reshape(shape).to(torch.float16)

    add('embeddings.word_embeddings.weight', (len(VOCAB), HIDDEN))
    add('embeddings.position_embeddings.weight', (POSITIONS, HIDDEN))
    add('embeddings.token_type_embeddings.weight', (TYPES, HIDDEN))
    add('embeddings.LayerNorm.weight', (HIDDEN,), True)
    add('embeddings.LayerNorm.bias', (HIDDEN,))
    for layer in range(layers):
        base = f'encoder.layer.{layer}.'
        for part in ['attention.self.query', 'attention.self.key', 'attention.self.value',
                     'attention.output.dense']:
            add(base + part + '.weight', (HIDDEN, HIDDEN))
            add(base + part + '.bias', (HIDDEN,))
        for part in ['attention.output.LayerNorm', 'output.LayerNorm']:
            add(base + part + '.weight', (HIDDEN,), True)
            add(base + part + '.bias', (HIDDEN,))
        add(base + 'intermediate.dense.weight', (INTERMEDIATE, HIDDEN))
        add(base + 'intermediate.dense.bias', (INTERMEDIATE,))
        add(base + 'output.dense.weight', (HIDDEN, INTERMEDIATE))
        add(base + 'output.dense.bias', (HIDDEN,))
    return tensors


def forward(tensors, layers):
    """float64 PyTorch BERT over the binary16 weights; raw masked-mean rows."""
    t = {name: value.to(torch.float64) for name, value in tensors.items()}
    ids = torch.tensor(IDS)
    mask = torch.tensor(MASK, dtype=torch.float64)
    seq = ids.shape[1]

    def linear(x, name):
        return torch.nn.functional.linear(x, t[name + '.weight'], t[name + '.bias'])

    def norm(x, name):
        return torch.nn.functional.layer_norm(
            x, (HIDDEN,), t[name + '.weight'], t[name + '.bias'], 1e-12)

    x = t['embeddings.word_embeddings.weight'][ids]
    x = x + t['embeddings.position_embeddings.weight'][:seq]
    x = x + t['embeddings.token_type_embeddings.weight'][0]
    x = norm(x, 'embeddings.LayerNorm')
    bias = ((mask - 1) * 1e9)[:, None, None, :]
    for layer in range(layers):
        base = f'encoder.layer.{layer}.'

        def heads(value):
            return value.reshape(2, seq, HEADS, HIDDEN // HEADS).transpose(1, 2)

        q, k, v = [heads(linear(x, base + 'attention.self.' + n)) for n in ['query', 'key', 'value']]
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        attended = attended.transpose(1, 2).reshape(2, seq, HIDDEN)
        x = norm(x + linear(attended, base + 'attention.output.dense'),
                 base + 'attention.output.LayerNorm')
        inner = torch.nn.functional.gelu(linear(x, base + 'intermediate.dense'))
        x = norm(x + linear(inner, base + 'output.dense'), base + 'output.LayerNorm')
    return ((x * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]).tolist()


def tower_record(role, model_id, prompt, layers, digest):
    body = bytes([role]) + TEXT(model_id) + TEXT('invented-v1') + BYTES(digest)
    # dims, L2 normalization, prompt, max tokens, MLX runtime id, CPU+GPU units.
    body += U32(HIDDEN) + U16(1) + TEXT(prompt) + U32(POSITIONS) + U16(2) + U16(2) + bytes([0])
    # Mean pooling, BERT, then the architecture configuration.
    body += bytes([1]) + U16(1) + U16(layers) + U32(HIDDEN) + U16(HEADS) + U32(INTERMEDIATE)
    body += U32(TYPES) + U32(POSITIONS) + F32(1e-12) + F32(10000) + U32(0)
    return body


def tokenizer_record():
    # WordPiece, lowercase; pad 0, unk 1, cls 2, sep 3; no charsmap.
    body = bytes([1, 1, 0, 0]) + U32(0) + U32(1) + U32(2) + U32(3) + U32(len(VOCAB))
    for token in VOCAB:
        body += TEXT(token) + struct.pack('<d', 0)
    return body + BYTES(b'')


def main():
    raw, records, references = {}, [], []
    for role, prefix, model_id, prompt, layers, shift in TOWERS:
        tensors = weights(layers, shift)
        data = {prefix + name: (list(value.shape), value.numpy().astype('<f2').tobytes())
                for name, value in tensors.items()}
        raw.update(data)
        digest = xxhash.xxh3_128_intdigest(b''.join(d for _, d in data.values())).to_bytes(16, 'little')
        records.append(tower_record(role, model_id, prompt, layers, digest))
        references.append((model_id, forward(tensors, layers)))
    body = b'ZEMB0001' + U32(3) + U32(HEADER) + bytes([len(TOWERS)]) + bytes(7)
    body += b''.join(records) + tokenizer_record() + tokenizer_record()
    body += struct.pack('<d', .5) + U32(len(raw))
    offset = HEADER
    for name, (shape, data) in raw.items():
        body += TEXT(name) + bytes([1, len(shape), 0, 0]) + b''.join(U32(n) for n in shape)
        body += U64(offset) + U64(len(data)) + U64(xxhash.xxh3_64_intdigest(data))
        offset += len(data)
    body += BYTES(b'')
    assert len(body) < HEADER
    body += bytes(HEADER - len(body)) + b''.join(d for _, d in raw.values())
    body += xxhash.xxh3_128_intdigest(body).to_bytes(16, 'little')
    (ROOT / 'pair-f16.zem').write_bytes(body)

    lines = [
        '// Generated by generate.py: float64 PyTorch CPU over the binary16 weights.',
        '// Never copied from Rust or MLX output.',
        f'pub const BUNDLE_SHA256: &str = "{hashlib.sha256(body).hexdigest()}";',
        f'pub const IDS: [[i32; 5]; 2] = {IDS};',
        f'pub const MASK: [[f32; 5]; 2] = [{", ".join("[" + ", ".join(f"{m}.0" for m in row) + "]" for row in MASK)}];',
    ]
    for (model_id, rows), constant in zip(references, ['DOCUMENT_RAW', 'QUERY_RAW']):
        lines.append(f'// {model_id}: raw masked-mean rows before L2.')
        lines.append(f'pub const {constant}: [[f32; {HIDDEN}]; 2] = [')
        lines.extend('    [' + ', '.join(f'{v:.10f}' for v in row) + '],' for row in rows)
        lines.append('];')
    (ROOT / 'reference.rs').write_text('\n'.join(lines) + '\n')
    subprocess.run(['rustfmt', '--edition', '2024', str(ROOT / 'reference.rs')], check=True)
    print('torch', torch.__version__, 'xxhash', xxhash.VERSION)
    for name in ['pair-f16.zem', 'reference.rs']:
        data = (ROOT / name).read_bytes()
        print(name, len(data), hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
