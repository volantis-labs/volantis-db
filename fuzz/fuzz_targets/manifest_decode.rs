#![no_main]

use libfuzzer_sys::fuzz_target;
use volantis_db::manifest::decode_manifest;

fuzz_target!(|data: &[u8]| {
    let _ = decode_manifest("fuzz-manifest", data);
});
