use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

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
    serde_json::from_slice(&output.stdout).unwrap()
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
