use std::ffi::OsString;
use std::path::Path;

use super::{StoreError, allocation_bytes};

/// Lists direct child directories of `root` whose `manifest.ze` resolves to a
/// regular file. Names are returned in ascending OS-encoded byte order.
///
/// An empty or NUL-containing root yields
/// [`StoreErrorKind::InvalidArgument`](super::StoreErrorKind::InvalidArgument).
/// Filesystem failures yield [`StoreErrorKind::Io`](super::StoreErrorKind::Io).
/// Children without a manifest file are skipped. Listing does not open stores.
pub fn list_namespaces(root: &Path) -> Result<Vec<OsString>, StoreError> {
    if root.as_os_str().is_empty() {
        return Err(StoreError::InvalidNamespaceRoot {
            detail: "namespace root must not be empty",
        });
    }
    if root.as_os_str().as_encoded_bytes().contains(&0) {
        return Err(StoreError::InvalidNamespaceRoot {
            detail: "path contains an interior NUL byte",
        });
    }

    let io_error = |path, source| StoreError::Io { path, source };
    let directory = std::fs::read_dir(root).map_err(|source| io_error(root.to_owned(), source))?;
    let mut names = Vec::new();
    for entry in directory {
        let entry = entry.map_err(|source| io_error(root.to_owned(), source))?;
        let file_type = entry
            .file_type()
            .map_err(|source| io_error(entry.path(), source))?;
        if !file_type.is_dir() {
            continue;
        }
        let manifest = entry.path().join(crate::manifest::io::MANIFEST_FILE);
        let metadata = match std::fs::metadata(&manifest) {
            Ok(metadata) => metadata,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
            Err(source) => return Err(io_error(manifest, source)),
        };
        if !metadata.is_file() {
            continue;
        }
        names
            .try_reserve(1)
            .map_err(|_| StoreError::AllocationFailed {
                needed: allocation_bytes::<OsString>(names.len().saturating_add(1)),
                component: "namespace list",
            })?;
        names.push(entry.file_name());
    }
    names.sort_unstable_by(|left, right| left.as_encoded_bytes().cmp(right.as_encoded_bytes()));
    Ok(names)
}
