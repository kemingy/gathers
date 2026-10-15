# Bounded-memory PCA with Accelerate

PCA covariance uses checked Accelerate SYRK on macOS; projection and inverse projection use
the shared checked SGEMM wrapper. Other platforms, unsupported strides, and dimensions outside the legacy BLAS
integer range fall back to `faer`. The eigensolver remains `faer`; no new dependency, Cargo
feature, or system BLAS installation is required. Assignment keeps its existing numerical guards
and outer Rayon scheduling. Neither backend changes global or caller-thread BLAS settings.

PCA centers at most 16 MiB of rows at a time, or one complete row for exceptionally wide inputs.
It reuses the buffer for both covariance formation and transformation, rather than retaining a
full centered corpus. Block covariance contributions are combined in `f64` and then converted to
`f32`, avoiding extra `f32` summation error across thousands of blocks. Only the lower covariance
triangle is formed and accumulated, since the eigensolver explicitly reads `Side::Lower`. The
portable fallback may form both triangles, but the upper entries are not accumulated. The existing `f64` mean,
finite-result checks, and overflow errors remain. This changes accumulation order and can change
principal directions and cluster labels slightly; bitwise reproduction of the previous PCA is
not promised. Original-space centroid reconstruction still uses the existing accumulation code.
Its improvement below comes from the changed memory behavior, not a new centroid-update kernel.

The CLI memory preflight accounts for bounded scratch and covariance/eigensolver storage. A
selected fitting sample is retained only when PCA fits fewer than all supplied rows. Full-source
PCA fitting does not reserve another full-source centered copy. The preflight remains a
conservative estimate, not an enforced RSS cap.

## Full-data training

Measured 2026-10-15/16 on an Apple M3 Max, 16 physical cores, 64 GiB, macOS 26.7.1,
aarch64/NEON, rustc 1.99.0. Release builds use thin LTO and one codegen unit; 16 Rayon workers,
default Accelerate thread policy, no `target-cpu=native`, and no library `perf` feature.

Both datasets are the full local real inputs: GIST 1,000,000 × 960 and Cohere 10,000,000 × 768.
PCA fits every row and projects to 128D. K=3,943 / 24,881, seed=42, ten rounds, tolerance=1e-12.
The temporary benchmark overrides the large-input RaBitQ heuristic to force exact flat assignment
during training, as requested; it does not change that heuristic in production. File loading,
caller input preparation, centroid serialization and final quality evaluation are outside the
training timer. GIST uses medians of three runs; Cohere is one full run before and after.

| Training including PCA and reconstruction | Previous | Improved | Speedup |
| --- | ---: | ---: | ---: |
| Full GIST | 10.310 s | 7.260 s | 1.42× |
| Full Cohere | 671.422 s | 364.303 s | 1.84× |

| Cohere stage | Previous seconds | Improved seconds |
| --- | ---: | ---: |
| Fit PCA on all 10M rows | 103.224 | 10.414 |
| Project all rows | 97.431 | 7.045 |
| Exact flat K-means in 128D | 341.210 | 338.316 |
| Reconstruct original 768D centroids | 126.597 | 4.616 |

Final full-data assignment uses the same guarded original-space flat path for both models,
streaming 65,536 rows per batch. It includes norm/workspace setup and excludes reading and SSE
computation. Cohere assignment: 136.673 → 137.982 s.
Training plus final assignment: 808.095 →
502.285 s
(1.61×).
The earlier SuperKMeans + Accelerate + pruning run trained in 558.103 s, so the
improved PCA pipeline is 1.53× faster in training in this comparison.

| Original-space SSE after final reassignment (lower is better) | Previous PCA | Improved PCA |
| --- | ---: | ---: |
| GIST | 1039435.749 | 1039447.091 |
| Cohere | 396155328.144 | 396157794.062 |

SSE changes are +0.00109% on GIST and
+0.00062% on Cohere, measured by summing selected-centroid
squared distances in `f64`. All completed fits and final assignments have zero empty clusters.
Preserved variance is approximately 86.3867% / 77.8317%. SuperKMeans pruning still has
approximately 0.57% lower SSE on Cohere; the improvement here is in time and memory, not evidence
of better clustering quality. Different initial centroids and one seed limit cross-library claims.

## Separate backend and memory contributions

Fit plus projection only, using all rows, with loading outside the timer:

| PCA backend / centered scratch | GIST seconds, median of 3 | Cohere seconds, single run |
| --- | ---: | ---: |
| faer / full centered corpus | 4.259 | 200.655 |
| Accelerate / full centered corpus | 1.452 | 93.596 |
| faer / 16 MiB blocks | 4.131 | 30.529 |
| Accelerate / 16 MiB blocks | 1.618 | 17.459 |

The Cohere faer/full baseline comes from the earlier training-stage timers; combined Accelerate/
blocked comes from the new training-stage timers. Other ablations call `PCA::fit`/`transform`
directly, omitting the outer `fit_sample` fitting-budget validation. They isolate the backend and
buffer choices within normal run-to-run timing limits, rather than forming a hardware-isolated
paired experiment. Ablation unblocked cases use the same implementation with scratch expanded
to the full source; blocked covariance sum conversion adds only a final matrix copy in that case.

At 768D, the old transform held ~30.72 GB input + ~30.72 GB centered copy + ~5.12 GB output,
before other workspaces. The new transform needs the input and output plus at most 16 MiB
centered scratch. System memory compression and swapping were observed during full-buffer runs.
The recorded swap/compression activity is system-wide, so it is not treated as a precise
per-process measurement. `time -l` could not query `kern.clockrate` in this environment; no
measured peak-RSS number is claimed. Library stage timers and the completed result record are
independent of that resource-query failure.

Scratch sensitivity on full GIST (fit + projection, medians of three):

| Centered scratch | Seconds |
| --- | ---: |
| 4 MiB | 2.615 |
| 16 MiB | 1.618 |
| 64 MiB | 1.453 |

16 MiB avoids the 4 MiB scheduling overhead while using one quarter of the scratch of 64 MiB.
The latter saves about 0.16 s in this stage test; 16 MiB is the current bounded-memory tradeoff,
not a claim of a universal optimal block size. All these configurations fit every source row.

The subsequent [Accelerate refinements](./accelerate-refinements.md) isolate SYRK and
lower-triangle accumulation and report measurements against this blocked SGEMM implementation.
The historical training figures above predate that refinement.

## Validation and reproduction

- `make format lint lint-ast test-ast test-rust`; 80 library tests plus workspace tests/doctests.
- `PIP_NO_BUILD_ISOLATION=0 PIP_NO_INDEX=1 make test-python`, with the installed local build
  dependencies: 73 Python tests. The ordinary isolated build tried an unavailable package index.
- Linux `cargo check --offline --lib --target x86_64-unknown-linux-gnu`; Intel macOS
  `cargo check --offline --lib --tests --target x86_64-apple-darwin`. Cross-target tests were not
  executed. A Linux test-only C dependency requires an unavailable cross compiler, so Linux
  compilation evidence is for the library, not its test binary.
- Focused new tests cover NN/NT/TN/TT matrix layouts, padded output strides, accumulation,
  the portable fallback, and covariance/projection tails at the actual scratch boundary.
  Existing large-offset, overflow, numerical assignment guard and scalar-reference tests pass.
- The existing isolated assignment benchmark preparation script still builds. New PCA timing
  instrumentation is kept outside the production API; no permanent benchmark target was added.

Raw JSON records, source archive, per-case commands and logs are in the accompanying local
`gathers-gemm-parallelism/pca-improved` report directory. The external optimized harness uses:

```console
CARGO_TARGET_DIR=/private/tmp/superkmeans-gist-target cargo build --offline --release --features accelerate --bins --manifest-path /private/tmp/superkmeans-gist-bench/Cargo.toml
/private/tmp/superkmeans-gist-target/release/cohere pca /Users/keming.yang/workdir/datasets/cohere-large-10m.fvecs /private/tmp/cohere-improved-centroids.fvecs 16
/private/tmp/superkmeans-gist-target/release/pca_stages /private/tmp/pr65-real-data/gist.fvecs true 16 3 16
```

The external source archive includes exact-training and backend/scratch control hooks; none are
exposed in the production library. No random or capped datasets are used for performance evidence.
