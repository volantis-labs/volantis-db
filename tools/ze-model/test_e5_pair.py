"""WF-986 pair bake contract: tiny invented pack, offline, no model execution."""
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
BAKER = HERE / "ze-model"
EXPORT = runpy.run_path(str(BAKER))
FIXTURE_DIR = HERE.parents[1] / "crates" / "volantis-db-text" / "tests" / "fixtures" / "wf986"
FIXTURE = runpy.run_path(str(FIXTURE_DIR / "generate.py"))


class Reader:
    def __init__(self, body: bytes, position: int = 0):
        self.body, self.position = body, position

    def take(self, fmt: str):
        values = struct.unpack_from("<" + fmt, self.body, self.position)
        self.position += struct.calcsize("<" + fmt)
        return values[0] if len(values) == 1 else values

    def field(self) -> bytes:
        length = self.take("I")
        self.position += length
        return self.body[self.position - length:self.position]

    def string(self) -> str:
        return self.field().decode("utf-8")


def parse(body: bytes) -> dict:
    """Independent reader of the layout-3 header fields that this ticket writes."""
    reader = Reader(body)
    assert body[:8] == b"ZEMB0001"
    version, header_len, count = reader.take("8xIIB7x")
    towers = []
    for _ in range(count):
        tower = {"role": reader.take("B"), "model_id": reader.string(), "version": reader.string(),
                 "weights_digest": reader.field(), "dims": reader.take("I"),
                 "normalization": reader.take("H"), "prefix": reader.string(),
                 "max_tokens": reader.take("I")}
        tower["runtime"], tower["compute"], tower["os_build"], tower["pooling"] = reader.take("HHBB")
        (tower["architecture"], tower["layers"], tower["hidden"], tower["heads"],
         tower["intermediate"], tower["type_vocab"], tower["positions"], tower["eps"],
         tower["rope"], tower["dense"]) = reader.take("HHIHIIIffI")
        towers.append(tower)
    tokenizers = []
    for _ in range(count):
        kind, flags, reserved, pad, unk, cls, sep, size = reader.take("BBHIIIII")
        pieces = [(reader.string(), reader.take("d")) for _ in range(size)]
        tokenizers.append({"kind": kind, "flags": flags, "reserved": reserved,
                           "specials": (pad, unk, cls, sep), "pieces": pieces,
                           "charsmap": reader.field()})
    alpha, tensors = reader.take("dI")
    for _ in range(tensors):
        reader.string()
        _, rank, _ = reader.take("BBH")
        reader.take(f"{rank}I")
        reader.take("QQQ")
    return {"version": version, "header_len": header_len, "towers": towers,
            "tokenizers": tokenizers, "alpha": alpha, "alignment": reader.field()}


class PairBake(unittest.TestCase):
    def setUp(self):
        self.directory = None
        self.fresh()

    def fresh(self):
        """Start from a new, unedited tiny pack."""
        self.tearDown()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.pair = FIXTURE["write_pack"](self.root / "pack")
        self.output = self.root / "out" / "pair.zem"

    def tearDown(self):
        if self.directory is not None:
            self.directory.cleanup()

    def edit_json(self, relative: str, change) -> None:
        path = self.pair.parent / relative
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value, ensure_ascii=False))
        # Keep the manifest digests current, so only the intended contract fails.
        pair = json.loads(self.pair.read_text())
        for role in ["query", "document"]:
            tower = self.pair.parent / pair[role]["path"]
            for key, name in [("weights_sha256", "model.safetensors"),
                              ("tokenizer_sha256", "tokenizer.json")]:
                if key in pair[role]:
                    pair[role][key] = hashlib.sha256((tower / name).read_bytes()).hexdigest()
        self.pair.write_text(json.dumps(pair, ensure_ascii=False))

    def bake(self, *flags: str) -> None:
        args = EXPORT["parser"]().parse_args(
            ["bake", "--pair", str(self.pair), "--output", str(self.output), *flags])
        with contextlib.redirect_stdout(io.StringIO()):
            EXPORT["bake"](args)

    def test_bake_pair(self):
        result = subprocess.run([sys.executable, str(BAKER), "bake", "--pair", str(self.pair),
                                 "--output", str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = self.output.read_bytes()
        self.assertEqual(EXPORT["xxhash"].xxh3_128_intdigest(body[:-16]).to_bytes(16, "little"),
                         body[-16:])
        header = parse(body)
        self.assertEqual(header["version"], 3)
        self.assertEqual(header["alignment"], hashlib.sha256(self.pair.read_bytes()).digest())
        self.assertEqual(header["alpha"], 0.5)
        manifest = json.loads(self.pair.read_text())
        expected = {"document": (0, "passage: ", 16, 2, 2), "query": (1, "query: ", 12, 1, 1)}
        for tower, tokenizer, role in zip(header["towers"], header["tokenizers"],
                                          ["document", "query"]):
            with self.subTest(role=role):
                number, prefix, limit, layers, heads = expected[role]
                self.assertEqual((tower["role"], tower["model_id"], tower["prefix"],
                                  tower["max_tokens"], tower["layers"], tower["heads"]),
                                 (number, manifest[role]["name"], prefix, limit, layers, heads))
                self.assertEqual((tower["dims"], tower["normalization"], tower["pooling"],
                                  tower["architecture"], tower["hidden"], tower["type_vocab"],
                                  tower["positions"], tower["dense"]), (4, 1, 1, 1, 4, 2, 16, 0))
                source = json.loads((self.pair.parent / role / "tokenizer.json").read_text())
                charsmap = source["normalizer"]["normalizers"][0]["precompiled_charsmap"]
                self.assertEqual((tokenizer["kind"], tokenizer["flags"], tokenizer["reserved"]),
                                 (2, 4, 0))
                self.assertEqual(tokenizer["specials"], (1, 3, 0, 2))
                self.assertEqual(tokenizer["pieces"],
                                 [(piece, score) for piece, score in source["model"]["vocab"]])
                self.assertEqual(tokenizer["charsmap"], base64.b64decode(charsmap))
        query, document = header["tokenizers"][1]["pieces"], header["tokenizers"][0]["pieces"]
        self.assertNotEqual([p for p, _ in query].index("▁Fake"),
                            [p for p, _ in document].index("▁Fake"))
        # The committed Rust fixture is this exact deterministic bake.
        self.assertEqual(body, (FIXTURE_DIR / "pair.zem").read_bytes())

    def test_pair_rejects_unrepresentable_tokenizers(self):
        def normalizers(change):
            return lambda tokenizer: change(tokenizer["normalizer"]["normalizers"])

        cases = [
            ("reversed normalizer order", "tokenizer.json", normalizers(list.reverse), "normalizer"),
            ("unsupported replacement pattern", "tokenizer.json",
             normalizers(lambda n: n[1].update(pattern={"Regex": " +"})), "normalizer"),
            ("unsupported replacement content", "tokenizer.json",
             normalizers(lambda n: n[1].update(content="▁")), "normalizer"),
            ("bare precompiled", "tokenizer.json",
             lambda t: t.update(normalizer=t["normalizer"]["normalizers"][0]), "normalizer"),
            ("metaspace without split", "tokenizer.json",
             lambda t: t["pre_tokenizer"].update(split=False), "Metaspace"),
            ("metaspace first-only prefix", "tokenizer.json",
             lambda t: t["pre_tokenizer"].update(prepend_scheme="first"), "Metaspace"),
            ("byte fallback", "tokenizer.json",
             lambda t: t["model"].update(byte_fallback=True), "Unigram"),
            ("left-stripped mask", "tokenizer.json",
             lambda t: t["added_tokens"][-1].update(lstrip=True), "added token"),
            ("roberta processor", "tokenizer.json",
             lambda t: t.update(post_processor={"type": "RobertaProcessing"}), "template"),
            ("lower casing", "tokenizer_config.json",
             lambda c: c.update(do_lower_case=True), "tokenizer_config"),
            ("left truncation", "tokenizer_config.json",
             lambda c: c.update(truncation_side="left"), "right"),
            ("CLS pooling", "1_Pooling/config.json",
             lambda p: p.update(pooling_mode_cls_token=True, pooling_mode_mean_tokens=False),
             "mean pooling"),
            ("prompt excluded from pooling", "1_Pooling/config.json",
             lambda p: p.update(include_prompt=False), "mean pooling"),
            ("no L2 module", "modules.json", lambda m: m.pop(), "Normalize"),
            ("hidden width", "config.json", lambda c: c.update(hidden_size=8), "dimension"),
            ("position bound", "config.json",
             lambda c: c.update(max_position_embeddings=8), "max_tokens"),
            ("architecture", "config.json", lambda c: c.update(model_type="gte"), "bert"),
        ]
        for name, relative, change, message in cases:
            for role in ["query", "document"]:
                with self.subTest(case=name, role=role):
                    self.fresh()
                    self.edit_json(f"{role}/{relative}", change)
                    with self.assertRaisesRegex(ValueError, message):
                        self.bake()
                    self.assertFalse(self.output.exists())

    def test_pair_rejects_manifest_and_cli_conflicts(self):
        for flags, message in [
            (["--query-max-tokens", "11"], "--query-max-tokens"),
            (["--document-max-tokens", "17"], "--document-max-tokens"),
            (["--query-prefix", "query: "], "--query-prefix"),
            (["--document", str(self.pair.parent / "document")], "--document"),
            (["--query-id", "other"], "--query-id"),
            (["--alignment-digest", "01"], "--alignment-digest"),
        ]:
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, message):
                self.bake(*flags)
        self.bake("--query-max-tokens", "12", "--document-max-tokens", "16")
        self.assertEqual([t["max_tokens"] for t in parse(self.output.read_bytes())["towers"]],
                         [16, 12])
        manifest_cases = [
            ("normalization", lambda p: p.update(normalization="none"), "L2"),
            ("similarity", lambda p: p.update(similarity="cosine"), "dot_product"),
            ("pooling", lambda p: p["query"].update(pooling="cls"), "pooling"),
            ("escaping path", lambda p: p["document"].update(path="../document"), "path"),
            ("weights digest", lambda p: p["document"].update(weights_sha256="0" * 64), "sha256"),
            ("tokenizer digest", lambda p: p["query"].update(tokenizer_sha256="0" * 64), "sha256"),
            ("prompt declaration", lambda p: p["query"].update(prefix="search: "), "prompt"),
        ]
        for name, change, message in manifest_cases:
            with self.subTest(manifest=name):
                self.fresh()
                pair = json.loads(self.pair.read_text())
                change(pair)
                self.pair.write_text(json.dumps(pair))
                if name == "prompt declaration":
                    (self.pair.parent / "query" / "config_sentence_transformers.json").write_text(
                        json.dumps({"prompts": {"query": "query: "}}))
                with self.assertRaisesRegex(ValueError, message):
                    self.bake()

    def test_release_pair_contract_is_exact(self):
        pair = json.loads(self.pair.read_text())
        pair["query"]["name"] = "v98-8-r1"
        self.pair.write_text(json.dumps(pair))
        with self.assertRaisesRegex(ValueError, "release pair contract"):
            self.bake()
        pair["document"]["name"] = "intfloat/multilingual-e5-small"
        pair["document"]["revision"] = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
        self.pair.write_text(json.dumps(pair))
        with self.assertRaisesRegex(ValueError, "release pair contract"):
            self.bake()
        contract = EXPORT["RELEASE_PAIRS"][("v98-8-r1", "intfloat/multilingual-e5-small")]
        self.assertEqual((contract["query"]["prefix"], contract["query"]["max_tokens"],
                          contract["query"]["layers"], contract["query"]["pieces"],
                          contract["query"]["rows"]), ("query: ", 96, 8, 98_304, 98_304))
        self.assertEqual((contract["document"]["prefix"], contract["document"]["max_tokens"],
                          contract["document"]["layers"], contract["document"]["pieces"],
                          contract["document"]["rows"]), ("passage: ", 512, 12, 250_002, 250_037))
        self.assertEqual(contract["dimension"], 384)

    def test_legacy_bake_still_requires_explicit_towers(self):
        for flags, message in [([], "--document"),
                               (["--document", "d"], "--document-id"),
                               (["--document", "d", "--document-id", "d"], "--query-prefix")]:
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, message):
                EXPORT["bake"](EXPORT["parser"]().parse_args(
                    ["bake", "--output", str(self.output), *flags]))


if __name__ == "__main__":
    unittest.main()
