//! Prepared-cache provenance metadata for the adaptive-probe benchmark.
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

use crate::distance::Distance;

pub(crate) struct Provenance {
    source: PathBuf,
    source_bytes: u64,
    rows: usize,
    original_dim: usize,
    metric: Distance,
    training_dim: usize,
}

impl Provenance {
    pub(crate) fn new(
        source: &Path,
        rows: usize,
        original_dim: usize,
        metric: Distance,
        training_dim: usize,
    ) -> Result<Self> {
        let source = source
            .canonicalize()
            .with_context(|| format!("cannot canonicalize {}", source.display()))?;
        let source_bytes = source
            .metadata()
            .with_context(|| format!("cannot inspect {}", source.display()))?
            .len();
        Ok(Self {
            source,
            source_bytes,
            rows,
            original_dim,
            metric,
            training_dim,
        })
    }

    fn as_json(&self) -> Value {
        json!({
            "source": self.source,
            "source_bytes": self.source_bytes,
            "rows": self.rows,
            "original_dim": self.original_dim,
            "metric": format!("{:?}", self.metric),
            "training_dim": self.training_dim,
        })
    }

    pub(crate) fn write(&self, prepared: &Path) -> Result<()> {
        let path = metadata_path(prepared);
        let contents = serde_json::to_string_pretty(&self.as_json())?;
        fs::write(&path, contents)
            .with_context(|| format!("cannot write prepared metadata {}", path.display()))
    }

    pub(crate) fn validate(&self, prepared: &Path) -> Result<()> {
        let path = metadata_path(prepared);
        let contents = fs::read_to_string(&path).with_context(|| {
            format!(
                "cannot read prepared metadata {}; recreate the cache with this benchmark",
                path.display()
            )
        })?;
        let actual: Value = serde_json::from_str(&contents)
            .with_context(|| format!("invalid prepared metadata {}", path.display()))?;
        let expected = self.as_json();
        ensure!(
            actual == expected,
            "prepared cache provenance mismatch for {}; expected {expected}, found {actual}",
            prepared.display()
        );
        Ok(())
    }
}

pub(crate) fn is_native_source(
    source: &Path,
    prepared: &Path,
    metric: Distance,
    original_dim: usize,
    training_dim: usize,
) -> Result<bool> {
    if metric != Distance::SquaredEuclidean || original_dim != training_dim {
        return Ok(false);
    }
    let source = source
        .canonicalize()
        .with_context(|| format!("cannot canonicalize {}", source.display()))?;
    let prepared = prepared
        .canonicalize()
        .with_context(|| format!("cannot canonicalize {}", prepared.display()))?;
    Ok(source == prepared)
}

fn metadata_path(prepared: &Path) -> PathBuf {
    let mut path = prepared.as_os_str().to_os_string();
    path.push(".json");
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use anyhow::Result;

    use super::{Provenance, is_native_source, metadata_path};
    use crate::distance::Distance;

    #[test]
    fn metadata_detects_missing_and_mismatched_provenance() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let source = directory.path().join("source.fvecs");
        let other = directory.path().join("other.fvecs");
        let prepared = directory.path().join("prepared.fvecs");
        fs::write(&source, b"abc")?;
        fs::write(&other, b"abc")?;
        let provenance = Provenance::new(&source, 2, 3, Distance::SquaredEuclidean, 128)?;
        provenance.write(&prepared)?;
        provenance.validate(&prepared)?;

        let other_source = Provenance::new(&other, 2, 3, Distance::SquaredEuclidean, 128)?;
        assert!(other_source.validate(&prepared).is_err());
        let other_metric = Provenance::new(&source, 2, 3, Distance::Cosine, 128)?;
        assert!(other_metric.validate(&prepared).is_err());
        let other_rows = Provenance::new(&source, 4, 3, Distance::SquaredEuclidean, 128)?;
        assert!(other_rows.validate(&prepared).is_err());

        fs::remove_file(metadata_path(&prepared))?;
        assert!(provenance.validate(&prepared).is_err());
        Ok(())
    }

    #[test]
    fn native_shortcut_requires_the_same_l2_128d_source() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let source = directory.path().join("source.fvecs");
        let other = directory.path().join("other.fvecs");
        fs::write(&source, b"abc")?;
        fs::write(&other, b"abc")?;
        assert!(is_native_source(
            &source,
            &source,
            Distance::SquaredEuclidean,
            128,
            128
        )?);
        assert!(!is_native_source(
            &source,
            &other,
            Distance::SquaredEuclidean,
            128,
            128
        )?);
        assert!(!is_native_source(
            &source,
            &source,
            Distance::Cosine,
            128,
            128
        )?);
        assert!(!is_native_source(
            &source,
            &source,
            Distance::SquaredEuclidean,
            96,
            128
        )?);
        Ok(())
    }
}
