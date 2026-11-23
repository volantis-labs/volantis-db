#![no_main]

use libfuzzer_sys::fuzz_target;
use volantis_db::wal::replay::replay;

fuzz_target!(|data: &[u8]| {
    let _ = replay(data);
});
