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
    let policy = REVIEW_BLOCK.load(std::sync::atomic::Ordering::Relaxed);
    if policy != 1 {
        let requested = if policy == 2 {
            1 << num_vectors.div_ceil(num_threads).max(MIN_MATMUL_BLOCK_SIZE).ilog2()
        } else { policy };
        return Some(PREFERRED_MATMUL_BLOCK_SIZE.min(1 << max_block_size.ilog2()).min(requested));
    }
"""
    + needle,
)
s += """\nstatic REVIEW_BLOCK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
#[allow(missing_docs)]
pub fn review_set_matrix_block(block: usize) { REVIEW_BLOCK.store(block, std::sync::atomic::Ordering::Relaxed); }
"""
(out / "src/kmeans/matrix.rs").write_text(s)
p = out / "src/kmeans.rs"
p.write_text(
    p.read_text()
    + """\n#[allow(missing_docs)]
pub use matrix::MatrixAssignmentWorkspace as ReviewMatrix;
#[allow(missing_docs)]
pub use matrix::review_set_matrix_block;
"""
)
examples = out / "crates/cli/examples"
examples.mkdir(exist_ok=True)
for name in ["query_ablate.rs", "final_assign.rs"]:
    shutil.copy(Path(__file__).parent / name, examples / name)
print(
    f"cargo build --offline --profile perf -p gathers-cli --examples --manifest-path {out / 'Cargo.toml'}"
)
