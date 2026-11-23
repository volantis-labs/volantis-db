//! Exact feature-campaign oracle boundary.
//!
//! Family-specific DTOs, expected-value algorithms, and comparators live in
//! the independent `volantis-db-adversarial-oracle` package. This harness
//! module only re-exports the stable evidence record; it deliberately has no
//! reusable comparison or planted-observation path.

pub use volantis_db_adversarial_oracle::{OracleFirstDifference, OracleRecord};
