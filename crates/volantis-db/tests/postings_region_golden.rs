#![allow(clippy::expect_used)]

use volantis_db::format::golden::decode_hex;
use volantis_db::fts::index::{Document, SegmentIndex};
use volantis_db::fts::sealed::SealedSegment;
use volantis_db::fts::tokenizer::{Analyzer, TokenizerConfig};

#[test]
fn whole_postings_region_v1_is_byte_exact() {
    let analyzer = Analyzer::new(TokenizerConfig::text_default()).expect("frozen analyzer");
    let mut active = SegmentIndex::new();
    active
        .push_document(&analyzer, &Document::with_text("bronze zeppelin"))
        .expect("first lexical row");
    active
        .push_document(&analyzer, &Document::with_text("silver zeppelin"))
        .expect("second lexical row");
    let sealed = SealedSegment::seal(&active).expect("seal lexical rows");
    let encoded = sealed.encode_region().expect("encode lexical region");
    let expected = decode_hex(include_str!(
        "fixtures/format/postings_segment_region_v1.hex"
    ))
    .expect("postings region fixture");
    assert_eq!(encoded, expected);
    let decoded = SealedSegment::decode_region(&encoded).expect("decode lexical region");
    assert_eq!(decoded.row_count(), 2);
    assert_eq!(
        decoded.field_lengths(volantis_db::fts::index::DEFAULT_FIELD),
        Some(&[2, 2][..])
    );
    let mut reserved = encoded;
    reserved[20] = 1;
    assert!(matches!(
        SealedSegment::decode_region(&reserved),
        Err(volantis_db::fts::sealed::SealedSegmentError::Geometry(
            "header reserved field is nonzero"
        ))
    ));
}
