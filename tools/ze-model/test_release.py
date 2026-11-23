"""Offline release assembly on the WF-986 tiny pair pack.

The bake is real. Only the Core ML compiler boundary is replaced by fixed bytes.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
EXPORT = runpy.run_path(str(HERE / "ze-model"))
FIXTURE = runpy.run_path(str(HERE.parents[1] / "crates" / "volantis-db-text" / "tests"
                             / "fixtures" / "wf986" / "generate.py"))
QUERY_NAME = "FAKE-WF986-Q"
PROVENANCE = "# Provenance\n\nFAKE-987 provenance: invented weights, no training data.\n"
LICENCE = "FAKE-987 licence\n"
STATUS = "FAKE-987 status: test pack only"
# (task, subset, this query tower, e5 as query tower). Only "nl" loses more than 0.05.
TASKS = [("FakeBelebele", "nld_Latn", 0.764, 0.929), ("FakeBelebele", "deu_Latn", 0.914, 0.925),
         ("FakeMiracl", "en", 0.493, 0.493)]


def release_pack(root: Path) -> Path:
    """The tiny pair pack plus the card inputs that a release pack carries."""
    pair_path = FIXTURE["write_pack"](root)
    pair = json.loads(pair_path.read_text())
    pair["query"].update(parameters=1234, parameters_note="FAKE-987 excludes the pooler")
    pair["document"].update(parameters=5678)
    groups = lambda index: {"fake_group": {"count": len(TASKS), "ndcg_at_10": index / 10,
                                           "recall_at_100": index / 10 + 0.05}}
    pair.update(target_languages=["en", "de"], measured_retrieval=groups(6),
                ceiling_retrieval_e5_as_query_tower=groups(7), release_status=STATUS)
    pair_path.write_text(json.dumps(pair, indent=2) + "\n")
    (root / "PROVENANCE.md").write_text(PROVENANCE)
    (root / "licenses").mkdir()
    (root / "licenses" / "FAKE-LICENSE.txt").write_text(LICENCE)
    (root / "benchmarks").mkdir()
    for name, column in [(f"retrieval-{QUERY_NAME}.json", 2), ("retrieval-e5-ceiling.json", 3)]:
        metrics = [{"task": task[0], "hf_subset": task[1], "split": "test",
                    "ndcg_at_10": task[column], "recall_at_100": 0.9} for task in TASKS]
        (root / "benchmarks" / name).write_text(json.dumps({"metrics": metrics}))
    return pair_path


def tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class Release(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.pair = release_pack(self.root / "pack")

    def tearDown(self):
        self.directory.cleanup()

    def release(self, out: Path, *flags: str, compiler=None, pack: Path | None = None) -> None:
        args = EXPORT["parser"]().parse_args([
            "release", "--pack", str(pack or self.pair.parent), "--out", str(out), "--name", "tiny",
            "--card-license", "mit", *flags])
        stubs = {} if compiler is None else {"coreml_export": compiler}
        with patch.dict(EXPORT["release"].__globals__, stubs), \
                contextlib.redirect_stdout(io.StringIO()):
            EXPORT["release"](args)

    def pair_bake(self, precision: str) -> bytes:
        output = self.root / f"pair-{precision}.zem"
        args = EXPORT["parser"]().parse_args([
            "bake", "--pair", str(self.pair), "--document-precision", precision,
            "--output", str(output)])
        with contextlib.redirect_stdout(io.StringIO()):
            EXPORT["bake"](args)
        return output.read_bytes()

    def check_sums(self, out: Path) -> None:
        listed = {}
        for line in (out / "SHA256SUMS").read_text().splitlines():
            digest, path = line.split("  ")
            listed[path] = digest
        self.assertEqual(sorted(listed), sorted(set(tree(out)) - {"SHA256SUMS"}))
        for path, digest in listed.items():
            self.assertEqual(digest, hashlib.sha256((out / path).read_bytes()).hexdigest(), path)

    def test_fp16_tensor_bytes_and_descriptor(self):
        root = self.root / "fp16"
        root.mkdir()
        save_file({"weight": torch.tensor([1.25, -0.5])}, root / "model.safetensors")
        tensor, = EXPORT["load_tensors"](root, "document", 0, "fp16")
        self.assertEqual(tensor.data, struct.pack("<ee", 1.25, -0.5))
        descriptor = EXPORT["tensor_descriptor"](tensor, 4096)
        self.assertEqual(descriptor[4 + len(tensor.name)], 1)

    def test_release_pair_mlx_only(self):
        out = self.root / "release"
        self.release(out, "--document-precision", "fp16", "--query-coreml", "none")
        files = tree(out)
        self.assertEqual(sorted(files), ["README.md", "SHA256SUMS", "licenses/FAKE-LICENSE.txt",
                                         "tiny.zem", "volantis-model.json"])
        # The bundle is the pair bake: PAIR.json names, prefixes and limits per tower.
        bundle = files["tiny.zem"]
        self.assertEqual(bundle, self.pair_bake("fp16"))
        manifest_bytes = files["volantis-model.json"]
        self.assertEqual(json.loads(manifest_bytes), {
            "format": 1, "name": "tiny", "bundle": "tiny.zem", "query_coreml": None,
            "files": [{"path": "tiny.zem", "size": len(bundle),
                       "sha256": hashlib.sha256(bundle).hexdigest()}]})
        self.assertEqual(files["licenses/FAKE-LICENSE.txt"].decode(), LICENCE)
        self.check_sums(out)
        card = files["README.md"].decode()
        self.assertTrue(card.startswith("---\nlicense: mit\nlanguage:\n- en\n- de\n"), card[:80])
        pair = json.loads(self.pair.read_text())
        for expected in [
            pair["query"]["name"], pair["document"]["name"], "`query: `", "`passage: `",
            "| 12 |", "| 16 |", "1,234", "5,678", "FAKE-987 excludes the pooler",
            "mean then L2", "dot product", "| 4 |", "FP16",
            "en, de", "FakeBelebele nld_Latn: 0.764 with this query tower, 0.929 with e5",
            "| fake_group | 3 | 0.600 | 0.700 |",
            "MLX only", "`query_coreml` is null",
            "FAKE-987 provenance: invented weights", STATUS,
            "`licenses/FAKE-LICENSE.txt`", f"{len(bundle):,}",
            hashlib.sha256(manifest_bytes).hexdigest(),
            hashlib.sha256(self.pair.read_bytes()).hexdigest(),
        ]:
            self.assertIn(expected, card)
        self.assertNotIn("deu_Latn: 0.914 with this", card)
        # Same pack and flags give the same bytes; an output is never overwritten.
        again = self.root / "again"
        self.release(again, "--document-precision", "fp16", "--query-coreml", "none")
        self.assertEqual(tree(again), files)
        with self.assertRaises(FileExistsError):
            self.release(out, "--document-precision", "fp16", "--query-coreml", "none")

    def test_release_coreml_and_query_mlx_precision(self):
        for precision, override, coreml in [("fp16", None, "fp16"), ("fp32", None, "int8"),
                                            ("fp16", "fp32", "int8"), ("fp32", "fp16", "fp16")]:
            with self.subTest(precision=precision, override=override, coreml=coreml):
                calls = []

                def compile_query(args):
                    calls.append((Path(args.query), args.tokens, args.precision))
                    compiled = Path(args.output).with_suffix(".mlmodelc")
                    (compiled / "weights").mkdir(parents=True)
                    (compiled / "weights/weight.bin").write_bytes(b"compiled query")

                out = self.root / f"{precision}-{override}-{coreml}"
                self.release(out, "--document-precision", precision, "--query-coreml", coreml,
                             *(["--query-mlx-precision", override] if override else []),
                             compiler=compile_query)
                # The Core ML query length is the PAIR.json query limit, not a fixed 96.
                self.assertEqual(calls, [((self.pair.parent / "query").resolve(), 12, coreml)])
                bundle = (out / "tiny.zem").read_bytes()
                name = b"query/embeddings.word_embeddings.weight"
                descriptor = bundle.index(struct.pack("<I", len(name)) + name)
                self.assertEqual(bundle[descriptor + 4 + len(name)],
                                 {"fp32": 0, "fp16": 1}[override or precision])
                manifest = json.loads((out / "volantis-model.json").read_text())
                self.assertEqual(manifest["query_coreml"], "tiny.mlmodelc")
                self.assertEqual([f["path"] for f in manifest["files"]],
                                 ["tiny.mlmodelc/weights/weight.bin", "tiny.zem"])
                for entry in manifest["files"]:
                    data = (out / entry["path"]).read_bytes()
                    self.assertEqual((entry["size"], entry["sha256"]),
                                     (len(data), hashlib.sha256(data).hexdigest()))
                self.check_sums(out)
                card = (out / "README.md").read_text()
                self.assertIn(f"Core ML query model `tiny.mlmodelc` ({coreml.upper()}, 12 tokens)",
                              card)
                self.assertNotIn("MLX only", card)

    def test_release_rejects_unclear_inputs(self):
        cases = iter(range(100))

        def fails(message, *flags, edit=None):
            index = next(cases)
            pack = self.pair.parent
            if edit is not None:
                pack = release_pack(self.root / f"pack-{index}").parent
                edit(pack)
            out = self.root / f"fail-{index}"
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.release(out, "--document-precision", "fp16", *flags, pack=pack)
                self.assertFalse(out.exists())

        fails("replaced by --query-coreml", "--query-coreml", "none",
              "--query-coreml-precision", "int8")
        fails("replaced by --query-coreml", "--query-coreml-precision", "fp16")
        fails("requires --query-coreml")

        def drop_pair_key(key):
            def edit(pack):
                pair = json.loads((pack / "PAIR.json").read_text())
                del pair[key]
                (pack / "PAIR.json").write_text(json.dumps(pair))
            return edit

        none = ("--query-coreml", "none")
        fails("PROVENANCE.md", *none, edit=lambda pack: (pack / "PROVENANCE.md").unlink())
        fails("licenses", *none, edit=lambda pack: (pack / "licenses/FAKE-LICENSE.txt").unlink())
        fails(f"retrieval-{QUERY_NAME}.json", *none,
              edit=lambda pack: (pack / f"benchmarks/retrieval-{QUERY_NAME}.json").unlink())
        fails("target_languages", *none, edit=drop_pair_key("target_languages"))
        fails("release_status", *none, edit=drop_pair_key("release_status"))
        fails("ceiling_retrieval_e5_as_query_tower", *none,
              edit=drop_pair_key("ceiling_retrieval_e5_as_query_tower"))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            EXPORT["parser"]().parse_args(["release", "--pack", "p", "--out", "o", "--name", "n",
                                           "--document-precision", "fp16", "--query-coreml", "none"])


if __name__ == "__main__":
    unittest.main()
