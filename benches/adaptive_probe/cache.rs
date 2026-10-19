//! Prepared-cache provenance metadata for the adaptive-probe benchmark.
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

use crate::distance::Distance;

pub(crate) struct Provenance {
    source: PathBuf,
    source_stamp: Value,
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
        let source_stamp = file_stamp(&source)?;
        Ok(Self {
            source,
            source_stamp,
            rows,
            original_dim,
            metric,
            training_dim,
        })
    }

    fn as_json(&self, prepared: &Path) -> Result<Value> {
        Ok(json!({
            "version": 1,
            "source": self.source,
            "source_stamp": self.source_stamp,
            "prepared_stamp": file_stamp(prepared)?,
            "rows": self.rows,
            "original_dim": self.original_dim,
            "metric": format!("{:?}", self.metric),
            "training_dim": self.training_dim,
        }))
    }

    pub(crate) fn write(&self, prepared: &Path) -> Result<()> {
        let path = metadata_path(prepared);
        let contents = serde_json::to_string_pretty(&self.as_json(prepared)?)?;
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
        let expected = self.as_json(prepared)?;
        ensure!(
            actual == expected,
            "prepared cache provenance mismatch for {}; expected {expected}, found {actual}",
            prepared.display()
        );
        Ok(())
    }
}

/// Cheap filesystem identity checks, not a content checksum. Detect ordinary same-size
/// replacements of either input without hashing the complete corpus on every reuse.
fn file_stamp(path: &Path) -> Result<Value> {
    let metadata = path
        .metadata()
        .with_context(|| format!("cannot inspect {}", path.display()))?;
    let modified = metadata.modified()?.duration_since(UNIX_EPOCH)?;
    Ok(json!({
        "bytes": metadata.len(),
        "modified_s": modified.as_secs(),
        "modified_ns": modified.subsec_nanos(),
    }))
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
        fs::write(&prepared, b"projected")?;
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
    fn metadata_rejects_same_size_source_and_prepared_replacements() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let source = directory.path().join("source.fvecs");
        let prepared = directory.path().join("prepared.fvecs");
        fs::write(&source, b"abc")?;
        fs::write(&prepared, b"xyz")?;
        let provenance = Provenance::new(&source, 2, 3, Distance::SquaredEuclidean, 128)?;
        provenance.write(&prepared)?;
        provenance.validate(&prepared)?;

        // Set explicit modification times so this regression needs no sleeps and
        // does not depend on the filesystem clock's timestamp resolution.
        for path in [&source, &prepared] {
            let old_modified = path.metadata()?.modified()?;
            fs::write(path, b"def")?;
            let modified = old_modified + std::time::Duration::from_secs(2);
            fs::File::options()
                .write(true)
                .open(path)?
                .set_modified(modified)?;
            let current = Provenance::new(&source, 2, 3, Distance::SquaredEuclidean, 128)?;
            assert!(current.validate(&prepared).is_err());
            current.write(&prepared)?;
            current.validate(&prepared)?;
        }
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
