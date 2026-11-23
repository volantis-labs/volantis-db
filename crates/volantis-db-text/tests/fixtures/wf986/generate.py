#!/usr/bin/env python3
"""Rebuild the WF-986 two-tower BERT + Unigram fixture and its independent oracles.

Tooling only: torch, safetensors, tokenizers 0.19.1 and xxhash come from the
ze-model tool environment, not from the Rust workspace. The weights are invented
integer arithmetic. No trained model, network access or accuracy claim is involved.

`write_pack(root)` writes a tiny offline pack in the PAIR.json layout:
two Sentence-Transformers BERT folders with XLM-R style tokenizers
(Sequence[Precompiled, Replace(" {2,}", " ")] + Metaspace). The bundle is made by
the production `ze-model bake --pair` command. Token IDs come from the reference
loader, `SentenceTransformer.tokenize` (Python strip, then HF tokenizers 0.19.1).
Vectors come from a float64 PyTorch CPU forward pass with mean pooling and L2.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parents[4]
BAKER = ENGINE / "tools" / "ze-model" / "ze-model"

SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]
# Shared pieces. Each role adds its own prompt piece and some role-only pieces.
WORDS = [
    ("▁", -2.0), (":", -2.5), ("▁Fake", -4.0), ("▁tulip", -4.1), ("▁ZQ", -4.5),
    ("986", -5.0), ("▁Fiktive", -5.1), ("▁Glüh", -5.5), ("birne", -5.6),
    ("▁Bal", -6.0), ("ise", -6.1), ("▁fictive", -6.2), ("▁Ver", -6.3),
    ("zonnen", -6.4), ("▁fiets", -6.5), ("▁Gl", -7.0), ("üh", -7.1),
]
LETTERS = "abcdefghiklmnoprstuvzABFGQVZéü9864"
CHARSMAP = {
    "▁": " ", "\t": " ", "\n": " ", "　": " ", "ü": "ü",
    "é": "é", "\u0001": "", "Ｚ": "Z",
}
ROLES = {
    "query": {"name": "FAKE-WF986-Q", "prefix": "query: ", "max_tokens": 12,
              "layers": 1, "heads": 1, "extra_rows": 0},
    "document": {"name": "FAKE-WF986-D", "prefix": "passage: ", "max_tokens": 16,
                 "layers": 2, "heads": 2, "extra_rows": 3},
}
HIDDEN, INTERMEDIATE, POSITIONS = 4, 6, 16
CASES = [
    ("en", "Fake tulip ZQ986"),
    ("de", "Fiktive Glühbirne ZQ986"),
    ("fr", "Balise fictive ZQ986"),
    ("nl", "Verzonnen fiets ZQ986"),
    ("empty", ""),
    ("repeated_spaces", "Fake   tulip  ZQ986"),
    ("trailing_space", "Fake tulip "),
    ("leading_spaces", "  Fake tulip"),
    ("literal_metaspace", "Fake▁tulip ▁ZQ986"),
    ("special_inside", "Fake<mask>tulip"),
    ("special_spaced", "Fake <mask> tulip"),
    ("adjacent_specials", "<s></s><pad><unk>Fake"),
    ("combining_marks", "Glühbirne fée"),
    ("whitespace_controls", "Fake\ttulip\n\nZQ　986 a\u0001b"),
    ("unknown", "ZQ🙂986 xylo"),
    ("python_space_edges", "\u001c\u3000 Fake tulip \t\u001f"),
    ("full_width", "ＺQ986"),
    ("query_exact", "Fake tulip Fake tulip Fake tulip Fake tulip"),
    ("query_over", "Fake tulip Fake tulip Fake tulip Fake tulip Fake"),
    ("document_exact", "Fake tulip " * 5 + "Fake tulip"),
    ("document_over", "Fake tulip " * 6 + "Fake"),
]


def vocabulary(role: str) -> list[tuple[str, float]]:
    prompt = "▁query" if role == "query" else "▁passage"
    words = [(prompt, -3.0)] + WORDS
    if role == "document":
        # One whole-word piece changes segmentation only in the document role.
        words.append(("▁Glühbirne", -5.2))
        # Swap the IDs of two shared pieces between the vocabularies.
        a, b = words.index(("▁Fake", -4.0)), words.index(("▁tulip", -4.1))
        words[a], words[b] = words[b], words[a]
    pieces = [(s, 0.0) for s in SPECIALS] + words
    pieces += [(c, -10.0 - i / 64) for i, c in enumerate(LETTERS)]
    return pieces + [("<mask>", 0.0)]


def darts(mapping: dict[str, str]) -> bytes:
    """Encode a SentencePiece precompiled charsmap: Darts double array + NUL strings."""
    strings = bytearray()
    tree: dict = {}
    for key, value in sorted(mapping.items(), key=lambda item: item[0].encode()):
        node = tree
        for byte in key.encode():
            node = node.setdefault(byte, {})
        node[None] = len(strings)
        strings += value.encode() + b"\0"
    units: dict[int, int] = {}
    bases: set[int] = set()

    def place(node: dict, index: int, label: int) -> None:
        labels = sorted(0 if key is None else key for key in node)
        base = 1
        # A unique base per node makes every label check unambiguous.
        while base in bases or any((base ^ c) in units or base ^ c == 0 for c in labels):
            base += 1
        bases.add(base)
        has_leaf = None in node
        units[index] = label | (256 if has_leaf else 0) | ((index ^ base) << 10)
        for c in labels:
            units[base ^ c] = 0
        if has_leaf:
            units[base] = 0x8000_0000 | node[None]
        for c in labels:
            if c:
                place(node[c], base ^ c, c)

    place(tree, 0, 0)
    size = max(max(units), max(base | 0xFF for base in bases)) + 1
    assert all(units[i] >> 10 < 1 << 21 for i in units if not units[i] & 0x8000_0000)
    array = [units.get(i, 0) for i in range(size)]
    return struct.pack("<I", 4 * size) + struct.pack(f"<{size}I", *array) + bytes(strings)


def tokenizer_json(role: str) -> dict:
    pieces = vocabulary(role)
    added = [{"id": i, "content": piece, "single_word": False, "lstrip": False,
              "rstrip": False, "normalized": False, "special": True}
             for i, (piece, _) in enumerate(pieces) if piece in SPECIALS + ["<mask>"]]
    if role == "query":
        pre = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}
    else:
        pre = {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True}
    template = [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}}]
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Precompiled",
             "precompiled_charsmap": base64.b64encode(darts(CHARSMAP)).decode()},
            {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}]},
        "pre_tokenizer": pre,
        "post_processor": {
            "type": "TemplateProcessing", "single": template,
            "pair": template + [{"SpecialToken": {"id": "</s>", "type_id": 0}},
                                {"Sequence": {"id": "B", "type_id": 0}},
                                {"SpecialToken": {"id": "</s>", "type_id": 0}}],
            "special_tokens": {t: {"id": t, "ids": [SPECIALS.index(t)], "tokens": [t]}
                               for t in ["</s>", "<s>"]}},
        "decoder": pre,
        "model": {"type": "Unigram", "unk_id": 3, "vocab": [list(p) for p in pieces],
                  **({"byte_fallback": False} if role == "query" else {})},
    }


def weights(role: str) -> dict:
    import torch

    spec = ROLES[role]
    rows = len(vocabulary(role)) + spec["extra_rows"]
    shift = 5 if role == "query" else 11
    tensors = {}

    def add(name, shape, norm=False):
        # Integer arithmetic gives exactly reproducible, role-distinct weights.
        seed = sum((i + 1) * ord(c) for i, c in enumerate(name)) + shift
        values = [((seed + i * 7 + i * i * 3) % 29 - 14) / 32 for i in range(math.prod(shape))]
        if norm:
            values = [1 + v / 4 for v in values]
        tensors[name] = torch.tensor(values, dtype=torch.float64).reshape(shape)

    add("embeddings.word_embeddings.weight", (rows, HIDDEN))
    add("embeddings.position_embeddings.weight", (POSITIONS, HIDDEN))
    add("embeddings.token_type_embeddings.weight", (2, HIDDEN))
    add("embeddings.LayerNorm.weight", (HIDDEN,), True)
    add("embeddings.LayerNorm.bias", (HIDDEN,))
    for layer in range(spec["layers"]):
        base = f"encoder.layer.{layer}."
        for part in ["attention.self.query", "attention.self.key",
                     "attention.self.value", "attention.output.dense"]:
            add(base + part + ".weight", (HIDDEN, HIDDEN))
            add(base + part + ".bias", (HIDDEN,))
        for part in ["attention.output.LayerNorm", "output.LayerNorm"]:
            add(base + part + ".weight", (HIDDEN,), True)
            add(base + part + ".bias", (HIDDEN,))
        add(base + "intermediate.dense.weight", (INTERMEDIATE, HIDDEN))
        add(base + "intermediate.dense.bias", (INTERMEDIATE,))
        add(base + "output.dense.weight", (HIDDEN, INTERMEDIATE))
        add(base + "output.dense.bias", (HIDDEN,))
    # Distinct non-zero type rows: a non-zero type ID would change every output.
    assert not torch.equal(tensors["embeddings.token_type_embeddings.weight"][0],
                           tensors["embeddings.token_type_embeddings.weight"][1])
    return tensors


def write_pack(root: Path) -> Path:
    """Write the tiny offline pack and return its PAIR.json path."""
    from safetensors.torch import save_file
    import torch

    root.mkdir(parents=True, exist_ok=True)
    pair = {"bundle": "fake-wf986-pair", "dimension": HIDDEN, "normalization": "L2",
            "similarity": "dot_product"}
    for role, spec in ROLES.items():
        tower = root / role
        (tower / "1_Pooling").mkdir(parents=True, exist_ok=True)
        tensors = {k: v.to(torch.float32).contiguous() for k, v in weights(role).items()}
        save_file(tensors, str(tower / "model.safetensors"), metadata={"format": "pt"})
        rows = tensors["embeddings.word_embeddings.weight"].shape[0]
        files = {
            "config.json": {
                "architectures": ["BertModel"], "model_type": "bert", "hidden_act": "gelu",
                "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE,
                "num_hidden_layers": spec["layers"], "num_attention_heads": spec["heads"],
                "max_position_embeddings": POSITIONS, "type_vocab_size": 2,
                "layer_norm_eps": 1e-12, "position_embedding_type": "absolute",
                "pad_token_id": 1, "vocab_size": rows},
            "1_Pooling/config.json": {
                "word_embedding_dimension": HIDDEN, "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False},
            "modules.json": [
                {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}],
            "sentence_bert_config.json": {"max_seq_length": spec["max_tokens"], "do_lower_case": False},
            "tokenizer.json": tokenizer_json(role),
            "tokenizer_config.json": {
                "bos_token": "<s>", "cls_token": "<s>", "eos_token": "</s>",
                "mask_token": "<mask>", "pad_token": "<pad>", "sep_token": "</s>",
                "unk_token": "<unk>", "model_max_length": spec["max_tokens"],
                "tokenizer_class": "PreTrainedTokenizerFast"},
        }
        for name, value in files.items():
            (tower / name).write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
        digest = lambda name: hashlib.sha256((tower / name).read_bytes()).hexdigest()
        pair[role] = {"name": spec["name"], "path": role, "prefix": spec["prefix"],
                      "max_tokens": spec["max_tokens"], "pooling": "mean then L2",
                      "weights_sha256": digest("model.safetensors"),
                      "tokenizer_sha256": digest("tokenizer.json")}
    path = root / "PAIR.json"
    path.write_text(json.dumps(pair, ensure_ascii=False, indent=2) + "\n")
    return path


def reference_vectors(role: str, rows: list[list[int]]) -> list[list[float]]:
    """Independent float64 BERT, attention-mask mean pooling incl. prompt, then L2."""
    import torch

    spec, tensors = ROLES[role], weights(role)
    width = max(len(row) for row in rows)
    ids = torch.tensor([row + [1] * (width - len(row)) for row in rows])
    mask = torch.tensor([[1.0] * len(row) + [0.0] * (width - len(row)) for row in rows],
                        dtype=torch.float64)
    linear = lambda x, n: torch.nn.functional.linear(x, tensors[n + ".weight"], tensors[n + ".bias"])
    norm = lambda x, n: torch.nn.functional.layer_norm(
        x, (HIDDEN,), tensors[n + ".weight"], tensors[n + ".bias"], 1e-12)
    x = (tensors["embeddings.word_embeddings.weight"][ids]
         + tensors["embeddings.position_embeddings.weight"][:width]
         + tensors["embeddings.token_type_embeddings.weight"][0])
    x = norm(x, "embeddings.LayerNorm")
    heads = spec["heads"]
    split = lambda t: t.reshape(len(rows), width, heads, HIDDEN // heads).transpose(1, 2)
    for layer in range(spec["layers"]):
        base = f"encoder.layer.{layer}."
        q, k, v = (split(linear(x, base + "attention.self." + n)) for n in ["query", "key", "value"])
        attended = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=(mask[:, None, None] - 1) * 1e9)
        attended = attended.transpose(1, 2).reshape(len(rows), width, HIDDEN)
        x = norm(x + linear(attended, base + "attention.output.dense"),
                 base + "attention.output.LayerNorm")
        hidden = torch.nn.functional.gelu(linear(x, base + "intermediate.dense"))
        x = norm(x + linear(hidden, base + "output.dense"), base + "output.LayerNorm")
    pooled = (x * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
    return torch.nn.functional.normalize(pooled, p=2, dim=1).tolist()


def rust_string(text: str) -> str:
    literal = json.dumps(text, ensure_ascii=False)
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: "\\u{" + m[1] + "}", literal)


def main() -> None:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    import tokenizers
    import torch
    import xxhash
    from sentence_transformers import SentenceTransformer

    if tokenizers.__version__ != "0.19.1":
        raise SystemExit("the token oracle requires HF tokenizers 0.19.1")
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="wf986-pack-") as directory:
        pair = write_pack(Path(directory) / "pack")
        bundle = ROOT / "pair.zem"
        subprocess.run([sys.executable, str(BAKER), "bake", "--pair", str(pair),
                        "--output", str(bundle)], check=True, stdout=subprocess.DEVNULL)
        lines = ["// Generated by generate.py: SentenceTransformer.tokenize IDs (HF tokenizers",
                 "// 0.19.1) and float64 PyTorch CPU vectors. Never copied from Rust or MLX output."]
        for role, spec in ROLES.items():
            source = Path(directory) / "pack" / role / "tokenizer.json"
            loader = SentenceTransformer(str(source.parent), device="cpu")
            loader.max_seq_length = spec["max_tokens"]
            rows = []
            for _, text in CASES:
                encoded = loader.tokenize([spec["prefix"] + text])
                assert encoded["attention_mask"].min() == 1
                rows.append(encoded["input_ids"][0].tolist())
            lengths = {name: len(ids) for (name, _), ids in zip(CASES, rows)}
            limit = spec["max_tokens"]
            if role == "query":
                assert lengths["query_exact"] == limit and lengths["query_over"] == limit
            else:
                assert lengths["document_exact"] == limit and lengths["document_over"] == limit
            upper = role.upper()
            lines += [f'pub const {upper}_NAME: &str = "{spec["name"]}";',
                      f"pub const {upper}_PREFIX: &str = {rust_string(spec['prefix'])};",
                      f"pub const {upper}_MAX_TOKENS: usize = {limit};",
                      f'pub const {upper}_TOKENIZER_SHA256: &str = "{hashlib.sha256(source.read_bytes()).hexdigest()}";',
                      f"pub const {upper}_CASES: &[(&str, &str, &[i32])] = &["]
            lines += [f"    ({rust_string(name)}, {rust_string(text)}, &{ids})," for (name, text), ids in zip(CASES, rows)]
            lines.append("];")
            vectors = reference_vectors(role, rows)
            lines.append(f"pub const {upper}_VECTORS: [[f32; {HIDDEN}]; {len(rows)}] = [")
            lines += ["    [" + ", ".join(f"{value:.10f}" for value in row) + "]," for row in vectors]
            lines.append("];")
        data = bundle.read_bytes()
        lines.insert(2, f'pub const BUNDLE_SHA256: &str = "{hashlib.sha256(data).hexdigest()}";')
        assert xxhash.xxh3_128_intdigest(data[:-16]).to_bytes(16, "little") == data[-16:]
    output = ROOT / "reference.rs"
    output.write_text("\n".join(lines) + "\n")
    subprocess.run(["rustfmt", "--edition", "2024", str(output)], check=True)
    print("torch", torch.__version__, "tokenizers", tokenizers.__version__, "xxhash", xxhash.VERSION)
    for name in ["pair.zem", "reference.rs"]:
        data = (ROOT / name).read_bytes()
        print(name, len(data), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
