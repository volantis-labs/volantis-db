//! Dependency-free stdin fuzz entrypoint for M3 checkpoint bytes.

use std::io::Read;

fn main() {
    let mut bytes = Vec::new();
    if std::io::stdin().read_to_end(&mut bytes).is_ok() {
        let _ = volantis_db::graph::build::validate_graph_build_checkpoint(&bytes);
        let _ = volantis_db::graph::refine::validate_refinement_checkpoint(&bytes);
    }
}
