#![allow(clippy::expect_used)]

use volantis_db_text::bundle::Bundle;
use volantis_db_text::{ModelOpenOptions, ModelRef, resolve_model};

// The e5 pair oracle: SentenceTransformer 3.0.1 tokenize on both tokenizer.json files.
#[path = "goldens/e5_pair.rs"]
#[allow(
    dead_code,
    reason = "shared with e5_pair.rs; the pack texts are not used here"
)]
mod golden;

#[test]
#[ignore = "requires the pinned HF model in ZE_MODEL_CACHE or the default cache; no MLX"]
fn release_tokenizer_ids_match_hf_0_19_1_for_both_towers() {
    let path = resolve_model(&ModelRef::default_model(), &ModelOpenOptions::default())
        .expect("resolve published release model");
    let bundle = Bundle::open(path).expect("open published release tokenizers");
    // The pair alignment identity is SHA-256 of the PAIR.json the goldens came from.
    let alignment: String = bundle
        .alignment_digest()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(alignment, golden::PAIR_SHA256);
    for (role, digest, prefix, maximum, cases) in [
        (
            "query",
            golden::QUERY_SHA256,
            golden::QUERY_PREFIX,
            golden::QUERY_MAX_TOKENS,
            golden::QUERY_CASES,
        ),
        (
            "document",
            golden::DOCUMENT_SHA256,
            golden::DOCUMENT_PREFIX,
            golden::DOCUMENT_MAX_TOKENS,
            golden::DOCUMENT_CASES,
        ),
    ] {
        let tower = if role == "query" {
            bundle.query_tower()
        } else {
            bundle.document_tower()
        };
        assert_eq!(tower.embedding.prompt_prefix, prefix);
        assert_eq!(tower.embedding.max_tokens as usize, maximum);
        for (name, text, repeat, expected) in cases {
            let input = text.repeat(*repeat);
            let batch = if role == "query" {
                bundle.tokenize_query(&input)
            } else {
                bundle.tokenize_document_chunks(&[input])
            }
            .expect("tokenize release case");
            assert_eq!(&batch.token_ids, expected, "{role}/{name}, source {digest}");
        }
    }
}
