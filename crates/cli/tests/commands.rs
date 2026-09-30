use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

fn fixture(path: &Path, dim: u32, rows: usize) {
    let mut file = File::create(path).unwrap();
    for row in 0..rows {
        file.write_all(&dim.to_le_bytes()).unwrap();
        for coordinate in 0..dim {
            let value = (row % 2) as f32 * 10.0 + row as f32 * 0.01 + coordinate as f32;
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }
}

fn cli() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_gathers"));
    command.args(["--threads", "2", "--seed", "42"]);
    command
}

fn json(command: &mut Command) -> serde_json::Value {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("ready: pid="));
    assert!(!stderr.contains("Attach the sampler"));
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn training_samples_before_loading_and_is_batch_size_independent() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = dir.path().join("vectors.fvecs");
    let centroids = dir.path().join("centroids.fvecs");
    fixture(&vectors, 3, 1000);
    let mut previous = None;
    for batch in ["1", "37", "4096"] {
        let result = json(
            cli()
                .args(["kmeans", "-i"])
                .arg(&vectors)
                .arg("-o")
                .arg(&centroids)
                .args(["-n", "2", "-m", "2", "--batch-rows", batch]),
        );
        assert_eq!(result["num_vectors"], 1000);
        assert_eq!(result["training_rows"], 512);
        let actual = std::fs::read(&centroids).unwrap();
        if let Some(expected) = previous {
            assert_eq!(actual, expected);
        }
        previous = Some(actual);
    }
    let rejected = cli()
        .args(["kmeans", "-i"])
        .arg(&vectors)
        .arg("-o")
        .arg(dir.path().join("rejected.fvecs"))
        .args(["-n", "2", "--memory-limit-gb", "1"])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("exceeds limit"));
    assert!(!dir.path().join("rejected.fvecs").exists());
}

#[test]
fn explicit_training_sample_size_is_validated_and_independent_of_read_mode() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = dir.path().join("vectors.fvecs");
    let centroids = dir.path().join("centroids.fvecs");
    fixture(&vectors, 3, 1000);
    let command = |samples: &str| {
        let mut command = cli();
        command
            .args(["kmeans", "-i"])
            .arg(&vectors)
            .arg("-o")
            .arg(&centroids)
            .args(["-n", "2", "-m", "1", "--training-samples", samples]);
        command
    };
    for (samples, message) in [
        ("0", "must be positive"),
        ("77", "at least 39"),
        ("1001", "no larger than the source"),
    ] {
        let output = command(samples).output().unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains(message));
        assert!(!centroids.exists());
    }
    // An explicit size may be below or above the default 256 * K, without resampling.
    for samples in [78, 700, 1000] {
        let report = json(&mut command(&samples.to_string()));
        assert_eq!(report["num_vectors"], 1000);
        assert_eq!(report["training_rows"], samples);
        assert_eq!(report["validate_all"], false);
        for field in ["index_sample_ms", "index_sort_ms", "sample_read_ms"] {
            assert!(report[field].as_f64().unwrap() >= 0.0);
        }
        let expected = std::fs::read(&centroids).unwrap();
        let report = json(command(&samples.to_string()).arg("--validate-all"));
        assert_eq!(report["validate_all"], true);
        assert_eq!(std::fs::read(&centroids).unwrap(), expected);
    }
}

#[test]
fn profiler_wait_accepts_enter_and_rejects_closed_stdin() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = dir.path().join("vectors.fvecs");
    let centroids = dir.path().join("centroids.fvecs");
    let trained = dir.path().join("trained.fvecs");
    fixture(&vectors, 2, 64);
    fixture(&centroids, 2, 2);
    for name in ["kmeans", "assign"] {
        let command = || {
            let mut command = cli();
            command.arg("--wait-for-profiler").arg(name);
            if name == "kmeans" {
                command
                    .arg("-i")
                    .arg(&vectors)
                    .arg("-o")
                    .arg(&trained)
                    .args(["-n", "1", "-m", "1"]);
            } else {
                command
                    .arg("--vectors")
                    .arg(&vectors)
                    .arg("--centroids")
                    .arg(&centroids)
                    .args(["--warmup", "0", "--repeats", "1"]);
            }
            command
        };
        let closed = command().stdin(Stdio::null()).output().unwrap();
        assert!(!closed.status.success());
        assert!(closed.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&closed.stderr)
                .contains("stdin closed while waiting for the profiler")
        );

        let mut child = command()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(b"\n").unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["command"], name);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("ready: pid="));
        assert!(stderr.contains("Attach the sampler"));
    }
}

#[test]
fn trained_fvecs_feed_reproducible_assignment() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = dir.path().join("vectors.fvecs");
    let centroids = dir.path().join("centroids.fvecs");
    fixture(&vectors, 3, 129);
    let trained = json(
        cli()
            .args(["kmeans", "-i"])
            .arg(&vectors)
            .arg("-o")
            .arg(&centroids)
            .args(["-n", "2", "-m", "2"]),
    );
    assert_eq!(trained["command"], "kmeans");
    assert_eq!(trained["num_centroids"], 2);
    assert_eq!(trained["dim"], 3);
    assert_eq!(
        std::fs::metadata(&centroids).unwrap().len(),
        2 * (4 + 3 * 4)
    );
    let assign = || {
        json(
            cli()
                .args(["assign", "--vectors"])
                .arg(&vectors)
                .arg("--centroids")
                .arg(&centroids)
                .args(["--warmup", "1", "--repeats", "2"]),
        )
    };
    let first = assign();
    let second = assign();
    assert_eq!(first["command"], "assign");
    assert_eq!(first["num_vectors"], 129);
    assert_eq!(first["threads"], 2);
    assert_eq!(first["query_ms"].as_array().unwrap().len(), 2);
    assert_eq!(first["label_hash"], second["label_hash"]);
    assert_eq!(first["metrics"], second["metrics"]);
    assert!(first["metrics"].as_str().unwrap().contains("queries(387)"));
    assert!(first.get("backend").is_none());
}

#[test]
fn reduction_methods_train_in_reduced_space_and_write_full_dimensional_centroids() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = dir.path().join("vectors.fvecs");
    fixture(&vectors, 4, 256);

    for method in ["pca", "srht"] {
        let centroids = dir.path().join(format!("{method}.fvecs"));
        let mut command = cli();
        command
            .args(["kmeans", "-i"])
            .arg(&vectors)
            .arg("-o")
            .arg(&centroids)
            .args([
                "-n",
                "2",
                "-m",
                "1",
                "--training-samples",
                "128",
                "--reduction",
                method,
                "--reduced-dim",
                "2",
            ]);
        if method == "pca" {
            command.args(["--projection-training-samples", "64"]);
        }
        let report = json(&mut command);
        assert_eq!(report["reduction"], method);
        assert_eq!(report["dim"], 4);
        assert_eq!(report["training_dim"], 2);
        assert!(report["reconstruction_empty_clusters"].as_u64().is_some());
        assert_eq!(
            report["projection_training_rows"],
            if method == "pca" { 64 } else { 0 }
        );
        assert_eq!(
            std::fs::metadata(&centroids).unwrap().len(),
            2 * (4 + 4 * 4)
        );
        for field in [
            "projection_fit_ms",
            "projection_transform_ms",
            "reconstruction_ms",
        ] {
            assert!(report[field].as_f64().unwrap() >= 0.0);
        }
        if method == "pca" {
            assert!(report["preserved_variance"].as_f64().unwrap() > 0.0);
        } else {
            assert!(report["preserved_variance"].is_null());
        }
    }
}

#[test]
fn cosine_normalizes_inputs_while_dot_preserves_magnitudes() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = dir.path().join("vectors.fvecs");
    let mut file = File::create(&vectors).unwrap();
    for row in 0..78 {
        file.write_all(&2_u32.to_le_bytes()).unwrap();
        let vector = if row % 2 == 0 {
            [100.0_f32, 0.0]
        } else {
            [0.0, 1.0]
        };
        for value in vector {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }

    let train = |distance: &str| {
        let centroids = dir.path().join(format!("{distance}.fvecs"));
        let report = json(
            cli()
                .args(["kmeans", "-i"])
                .arg(&vectors)
                .arg("-o")
                .arg(&centroids)
                .args(["-n", "1", "-m", "1", "--distance", distance]),
        );
        assert_eq!(report["distance"], distance);
        let bytes = std::fs::read(centroids).unwrap();
        [
            f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            f32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        ]
    };
    let cosine = train("cos");
    let dot = train("dot");
    let diagonal = 0.5_f32.sqrt();
    assert!((cosine[0] - diagonal).abs() < 1e-5);
    assert!((cosine[1] - diagonal).abs() < 1e-5);
    assert!(dot[0] > 0.999);
    assert!(dot[1] < 0.011);
}

#[test]
fn projected_cosine_writes_unit_original_space_centroids() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = dir.path().join("vectors.fvecs");
    let centroids = dir.path().join("centroids.fvecs");
    fixture(&vectors, 4, 128);
    let report = json(
        cli()
            .args(["kmeans", "-i"])
            .arg(&vectors)
            .arg("-o")
            .arg(&centroids)
            .args([
                "-n",
                "2",
                "-m",
                "1",
                "--distance",
                "cos",
                "--reduction",
                "srht",
                "--reduced-dim",
                "2",
            ]),
    );
    assert_eq!(report["distance"], "cos");
    let bytes = std::fs::read(centroids).unwrap();
    for row in bytes.as_chunks::<{ 4 + 4 * 4 }>().0 {
        let norm = row[4..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}

#[test]
fn projected_cosine_accepts_zero_projections() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = dir.path().join("vectors.fvecs");
    let centroids = dir.path().join("centroids.fvecs");
    let srht = gathers::reduction::SRHT::new(2, 1, 42).unwrap();
    let coefficients = srht.transform(&[1.0, 0.0, 0.0, 1.0]).unwrap();
    let null_row = [coefficients[1], -coefficients[0]];

    for (method, directions) in [
        ("pca", vec![[1.0_f32, 0.0]]),
        ("pca", vec![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        ("srht", vec![null_row]),
        ("srht", vec![null_row, [1.0, 0.0]]),
    ] {
        let rows = directions.repeat(39);
        gathers::utils::write_vecs(&vectors, &rows).unwrap();
        let report = json(
            cli()
                .args(["kmeans", "-i"])
                .arg(&vectors)
                .arg("-o")
                .arg(&centroids)
                .args([
                    "-n",
                    "1",
                    "-m",
                    "1",
                    "--distance",
                    "cos",
                    "--reduction",
                    method,
                    "--reduced-dim",
                    "1",
                ]),
        );
        assert_eq!(report["distance"], "cos");
        let mut expected = [0.0_f32; 2];
        for mut direction in directions {
            gathers::utils::try_normalize_rows(&mut direction, 2, false).unwrap();
            for (sum, value) in expected.iter_mut().zip(direction) {
                *sum += value;
            }
        }
        gathers::utils::try_normalize_rows(&mut expected, 2, false).unwrap();
        let bytes = std::fs::read(&centroids).unwrap();
        assert_eq!(bytes.len(), 12);
        for (coordinate, expected) in bytes[4..].as_chunks::<4>().0.iter().zip(expected) {
            let actual = f32::from_le_bytes(*coordinate);
            assert!(
                (actual - expected).abs() < 1e-5,
                "{method}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn projected_cosine_still_rejects_zero_original_rows() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = dir.path().join("vectors.fvecs");
    let centroids = dir.path().join("centroids.fvecs");
    let mut rows = [[1.0_f32, 0.0]; 39];
    rows[3] = [0.0, 0.0];
    gathers::utils::write_vecs(&vectors, &rows).unwrap();

    for method in ["pca", "srht"] {
        let output = cli()
            .args(["kmeans", "-i"])
            .arg(&vectors)
            .arg("-o")
            .arg(&centroids)
            .args([
                "-n",
                "1",
                "--distance",
                "cos",
                "--reduction",
                method,
                "--reduced-dim",
                "1",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("finite nonzero L2 norm"));
        assert!(!stderr.contains("panicked"));
        assert!(!centroids.exists());
    }
}

#[test]
fn assignment_rejects_mismatched_dimensions_and_invalid_options() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = dir.path().join("vectors.fvecs");
    let centroids = dir.path().join("centroids.fvecs");
    fixture(&vectors, 3, 2);
    fixture(&centroids, 2, 2);
    for (extra, message) in [
        (vec![], "query and centroid dimensions must match"),
        (vec!["--repeats", "0"], "repeats must be positive"),
        (
            vec!["--min-seconds", "NaN"],
            "min-seconds must be finite and nonnegative",
        ),
        (
            vec!["--num-vectors", "0"],
            "requested row count must be positive",
        ),
    ] {
        let output = cli()
            .args(["assign", "--vectors"])
            .arg(&vectors)
            .arg("--centroids")
            .arg(&centroids)
            .args(extra)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains(message));
    }
}

#[test]
fn input_errors_identify_the_file_and_invalid_row() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = dir.path().join("vectors.fvecs");
    let centroids = dir.path().join("centroids.fvecs");
    fixture(&centroids, 2, 2);
    let run = || {
        cli()
            .args(["assign", "--vectors"])
            .arg(&vectors)
            .arg("--centroids")
            .arg(&centroids)
            .output()
            .unwrap()
    };
    let missing = run();
    assert!(!missing.status.success());
    let error = String::from_utf8_lossy(&missing.stderr);
    assert!(error.contains("cannot open"));
    assert!(error.contains(vectors.to_str().unwrap()));

    fixture(&vectors, 2, 2);
    let mut bytes = std::fs::read(&vectors).unwrap();
    bytes[12..16].copy_from_slice(&3_u32.to_le_bytes());
    std::fs::write(&vectors, bytes).unwrap();
    let malformed = run();
    assert!(!malformed.status.success());
    let error = String::from_utf8_lossy(&malformed.stderr);
    assert!(error.contains(vectors.to_str().unwrap()));
    assert!(error.contains("row 2 has a different dimension"));
}
