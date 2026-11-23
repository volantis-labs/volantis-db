use std::process::ExitCode;

use volantis_db::lifecycle::{OpenOptions, Store};

fn main() -> ExitCode {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: volantis-db-size-consumer <store-directory>");
        return ExitCode::from(2);
    };
    let store = match Store::open(path, OpenOptions::default()) {
        Ok(store) => store,
        Err(error) => {
            eprintln!("open failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    match store.close() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("close failed: {error}");
            ExitCode::FAILURE
        }
    }
}
