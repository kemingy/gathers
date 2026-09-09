# Contributing

## Randomized tests

Randomized correctness tests intentionally choose a fresh seed on each run so they can exercise
inputs that a single fixed dataset may miss. Each test prints its seed before using a deterministic
`StdRng`. Cargo includes captured stderr when a test fails.

Reproduce a failure with the printed seed:

```console
GATHERS_TEST_SEED=123456 cargo test test_name -- --nocapture
```

Keep targeted regression tests deterministic when a particular input is part of the regression.
Do not replace the shared randomized-test helper with a fixed seed merely to make successful runs
identical.

## Benchmarks

Benchmarks also generate fresh representative inputs intentionally. They measure performance over
the input distribution rather than one permanently fixed sample; benchmark conclusions should be
confirmed across multiple samples or runs. Input preparation and cloning belong in Criterion setup,
outside the timed routine.
