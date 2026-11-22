"""ZE-553: export both towers to ONNX fp32, then int8 variants.

Usage: export_onnx.py <model dirs> <out dir>
<model dirs> holds document/ and query/ built by zem_extract.py.
Output: doc.onnx, query.onnx (CLS vector, unnormalized, 768-d),
doc.int8.onnx, query.int8.onnx (dynamic int8 MatMul + Gather).
"""
import os, sys
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoModel

pack, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)


class Doc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = AutoModel.from_pretrained(
            pack / "document", trust_remote_code=True, local_files_only=True,
            attn_implementation="eager", use_memory_efficient_attention=False,
            unpad_inputs=False).eval().float()

    def forward(self, ids, mask):
        # Same math as GteModel.forward, but RoPE and the bias come from
        # tensor shapes, so the legacy tracer keeps the length dynamic.
        e = self.m.embeddings
        x = e.word_embeddings(ids) + e.token_type_embeddings.weight[0]
        x = e.LayerNorm(x)
        pos = torch.cumsum(torch.ones_like(ids[0]), 0).to(torch.float32) - 1.0
        freqs = pos[:, None] * e.rotary_emb.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        rope = (emb.cos()[None, :, None, :], emb.sin()[None, :, None, :])
        bias = (1.0 - mask[:, None, None, :].to(x.dtype)) * torch.finfo(x.dtype).min
        for layer in self.m.encoder.layer:
            x = layer(x, bias, rope)[0]
        return x[:, 0]

    def reference(self, ids, mask):
        return self.m(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]


class Query(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = AutoModel.from_pretrained(
            pack / "query", trust_remote_code=True, local_files_only=True,
            attention_backend="explicit").eval().float()

    def forward(self, ids, mask):
        x = self.m(ids, mask, project_output=False).last_hidden_state[:, 0]
        return self.m.output_projection(x)


import numpy as np
import onnxruntime as ort
towers = [("query", Query().eval(), 24), ("doc", Doc().eval(), 37)]
for name, module, length in towers:
    path = out / f"{name}.onnx"
    if path.exists():
        module.eval()
        continue
    ids = torch.randint(5, 1000, (2, length))
    mask = torch.ones_like(ids)
    mask[1, length // 2:] = 0
    with torch.inference_mode():
        torch.onnx.export(module, (ids, mask), str(path), dynamo=False,
            input_names=["input_ids", "attention_mask"], output_names=["cls"],
            dynamic_axes={"input_ids": {0: "b", 1: "s"},
                          "attention_mask": {0: "b", 1: "s"}, "cls": {0: "b"}},
            opset_version=17)
    module.eval()  # export restores the pre-export training flag
    print("exported", path, path.stat().st_size)

# Parity: ORT fp32 against eager torch at shapes unlike the trace shape.
for name, module, _ in towers:
    sess = ort.InferenceSession(str(out / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    for rows, length in [(1, 5), (3, 90), (2, 300)]:
        if name == "query" and length > 96:
            continue
        ids = torch.randint(5, 90000, (rows, length))
        mask = torch.ones_like(ids)
        mask[-1, length // 3:] = 0
        with torch.inference_mode():
            ref = module(ids, mask).numpy()
            if hasattr(module, "reference"):
                hf = module.reference(ids, mask).numpy()
                print(f"WRAPPER {name} len={length} max_abs_vs_hf={np.abs(ref - hf).max():.2e}")
        got = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
        cos = (ref * got).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(got, axis=1)
        print(f"PARITY {name} rows={rows} len={length} min_cos={cos.min():.8f} "
              f"max_abs={np.abs(ref - got).max():.2e}")

from onnxruntime.quantization import QuantType, quantize_dynamic
for name in ("query", "doc"):
    src, dst = out / f"{name}.onnx", out / f"{name}.int8.onnx"
    if not dst.exists():
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gather"])
    print("int8", dst, dst.stat().st_size)
