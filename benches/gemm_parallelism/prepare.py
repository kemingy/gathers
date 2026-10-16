"""Prepare an isolated benchmark checkout; never exposes hooks in the production API.

Requires Python 3.12+ for safe tar extraction. Destination must be empty.
"""

import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
if len(sys.argv) != 2:
    raise SystemExit("usage: prepare.py EMPTY_DESTINATION")
out = Path(sys.argv[1]).resolve()
if out.exists() and any(out.iterdir()):
    raise SystemExit("destination must be empty")
out.mkdir(parents=True, exist_ok=True)
data = subprocess.check_output(["git", "archive", "HEAD"], cwd=repo)
with tarfile.open(fileobj=io.BytesIO(data)) as t:
    t.extractall(out, filter="data")
s = (repo / "src/kmeans/matrix.rs").read_text()
s = s.replace(
    "pub(crate) struct MatrixAssignmentWorkspace",
    "#[allow(missing_docs)]\npub struct MatrixAssignmentWorkspace",
).replace("    pub(crate) fn", "    #[allow(missing_docs)]\n    pub fn")
needle = (
    "    // Size tiles for parallel batches, rounding up to amortize matrix packing."
)
assert s.count(needle) == 1, "matrix helper changed; update benchmark instrumentation"
s = s.replace(
    needle,
    """    // Benchmark-only alternatives; policy 1 continues through the production helper unchanged.
    let policy = BENCH_BLOCK.load(std::sync::atomic::Ordering::Relaxed);
    if policy != 1 {
        return Some(PREFERRED_MATMUL_BLOCK_SIZE.min(1 << max_block_size.ilog2()).min(policy));
    }
"""
    + needle,
)
s += """\nstatic BENCH_BLOCK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
#[allow(missing_docs)]
pub fn bench_set_matrix_block(block: usize) { BENCH_BLOCK.store(block, std::sync::atomic::Ordering::Relaxed); }
"""
needle = '    } else if cfg!(all(target_os = "macos", target_arch = "aarch64"))'
assert s.count(needle) == 1, "crossover helper changed; update benchmark instrumentation"
s = s.replace(
    needle,
    "    } else if BENCH_CROSSOVER.load(std::sync::atomic::Ordering::Relaxed)\n"
    '        && cfg!(all(target_os = "macos", target_arch = "aarch64"))',
)
s += """\nstatic BENCH_CROSSOVER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[allow(missing_docs)]
pub fn bench_use_crossover(enabled: bool) { BENCH_CROSSOVER.store(enabled, std::sync::atomic::Ordering::Relaxed); }
"""
(out / "src/kmeans/matrix.rs").write_text(s)
# Copy the shared backend and its declaration even when preparing uncommitted changes.
shutil.copy(repo / "src/lib.rs", out / "src/lib.rs")
shutil.copy(repo / "src/reduction/pca.rs", out / "src/reduction/pca.rs")
linalg = (repo / "src/linalg.rs").read_text()
needle = '    if accelerate::matmul(&dst, accum, lhs, rhs, alpha).is_some() {'
assert linalg.count(needle) == 1, "backend helper changed; update benchmark instrumentation"
linalg = linalg.replace(
    needle,
    "    if BENCH_ACCELERATE.load(std::sync::atomic::Ordering::Relaxed)\n"
    "        && accelerate::matmul(&dst, accum, lhs, rhs, alpha).is_some() {",
)
linalg += """\nstatic BENCH_ACCELERATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[allow(missing_docs)]
pub fn bench_use_accelerate(enabled: bool) { BENCH_ACCELERATE.store(enabled, std::sync::atomic::Ordering::Relaxed); }
"""
(out / "src/linalg.rs").write_text(linalg)
p = out / "src/kmeans.rs"
p.write_text(
    p.read_text()
    + """\n#[allow(missing_docs)]
pub use matrix::MatrixAssignmentWorkspace as BenchMatrix;
#[allow(missing_docs)]
pub use matrix::{bench_set_matrix_block, bench_use_crossover};
#[allow(missing_docs)]
pub use crate::linalg::bench_use_accelerate;
"""
)
examples = out / "crates/cli/examples"
examples.mkdir(exist_ok=True)
shutil.copy(Path(__file__).parent / "query_compare.rs", examples / "query_compare.rs")
print(
    f"cargo build --offline --profile perf -p gathers-cli --example query_compare --manifest-path {out / 'Cargo.toml'}"
)
