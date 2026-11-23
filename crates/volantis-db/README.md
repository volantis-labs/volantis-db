# Volantis DB

Volantis DB is the fastest and most accurate in-process search engine built
for macOS and Apple silicon. It stores vectors, text, metadata, and search
indexes in one persistent directory and supports vector, lexical, and hybrid
retrieval through a native Rust API.

```sh
cargo add volantis-db
```

Applications provide their own document and query vectors. The engine owns the
persistent store, append-only ingestion, point-in-time snapshots, filtering,
and retrieval.

Run the bundled example after cloning the repository:

```sh
cargo run --release -p volantis-db --example five_vectors_search
```

## List namespaces

```rust
use std::path::Path;
use volantis_db::lifecycle::{StoreError, list_namespaces};

fn main() -> Result<(), StoreError> {
    for name in list_namespaces(Path::new("my-database"))? {
        println!("{}", name.to_string_lossy());
    }
    Ok(())
}
```

`list_namespaces` returns the native names (`OsString`) of direct child
directories whose `manifest.ze` resolves to a regular file. Results are sorted
by OS-encoded bytes and are empty when no children qualify. Listing uses
filesystem metadata and works without opening a store.

Empty paths and paths containing NUL yield `StoreErrorKind::InvalidArgument`.
A missing or unreadable root yields `StoreErrorKind::Io`.

See the [project README](https://github.com/volantis-labs/volantis-db) for benchmark
results, storage details, and the complete native API quickstart.

Licensed under [GPL-3.0-only](https://www.gnu.org/licenses/gpl-3.0.html).
