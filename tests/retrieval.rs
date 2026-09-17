//! End-to-end retrieval tests: build a real index over a fixture tree, then check
//! that the queries from `project.txt` return the documents they should.
//!
//! This is the test that catches the failures unit tests cannot. A mean-pooled BGE
//! model, or one missing its query prefix, still produces well-formed unit vectors
//! and plausible-looking output — nothing errors, results just get quietly worse.
//! Only checking known-good answers end to end detects that.
//!
//! Requires the model, which is downloaded on first run. Set `WOM_SKIP_MODEL=1` to
//! skip the cases that need it; the lexical cases still run.

use std::path::{Path, PathBuf};
use std::process::Command;

/// A corpus mirroring the spec's examples closely enough to test its claims.
fn build_fixture(root: &Path) {
    let files: &[(&str, &str)] = &[
        (
            "employment_docs/2024_W2.txt",
            "Wage and Tax Statement 2024. Employer identification number 12-3456789. \
             Wages, tips, other compensation. Federal income tax withheld. \
             Social security wages. Medicare tax withheld.",
        ),
        (
            "employment_docs/2023_W2.txt",
            "Wage and Tax Statement 2023. Employer identification number 12-3456789. \
             Federal income tax withheld. State income tax.",
        ),
        (
            "employment_docs/procedural_notes.txt",
            "Onboarding procedure for new hires. Payroll setup, direct deposit \
             enrolment, benefits election window, and the employee handbook.",
        ),
        (
            "bosnia_croatia_trip_aug2024/dubrovnik_to_sarajevo.txt",
            "Driving route from Dubrovnik along the Adriatic coast, crossing at Neum, \
             then inland through Mostar to Sarajevo. Ferry times and border notes.",
        ),
        (
            "bosnia_croatia_trip_aug2024/packing_list.txt",
            "Passport, travel insurance, adapters, hiking boots, swimwear.",
        ),
        (
            "projects/2026_esp32epaper_display/src/main.cpp",
            "// ESP32 e-paper weather station.\n\
             #include <Wire.h>\n\
             #include \"bme280.h\"\n\
             // Reads temperature and humidity sensors, monitors prayer times from an \
             // HTTP endpoint, and refreshes the display every fifteen minutes.\n\
             void setup() { Wire.begin(); }",
        ),
        (
            "projects/webshop/src/checkout.js",
            "// Shopping cart checkout flow: validates the basket, applies discount \
             // codes, and posts the order to the payments API.\n\
             export function checkout(cart) { return post('/pay', cart); }",
        ),
        (
            "recipes/sourdough.md",
            "# Sourdough\n100g starter, 500g bread flour, 350g water, 10g salt. \
             Autolyse one hour, bulk ferment four hours with folds.",
        ),
    ];

    for (rel, body) in files {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, body).unwrap();
    }

    // Things the ignore rules must exclude. If any of these show up in results,
    // the walker regressed.
    std::fs::create_dir_all(root.join("projects/webshop/node_modules/left-pad")).unwrap();
    std::fs::write(
        root.join("projects/webshop/node_modules/left-pad/index.js"),
        "module.exports = function leftPad() {}",
    )
    .unwrap();
    std::fs::create_dir_all(root.join("projects/webshop/.git")).unwrap();
    std::fs::write(root.join("projects/webshop/.git/config"), "[core]").unwrap();
    std::fs::write(root.join("projects/build.o"), [0u8, 1, 2, 3]).unwrap();
}

struct Harness {
    home: PathBuf,
    corpus: PathBuf,
}

impl Harness {
    fn new(name: &str) -> Self {
        let base = std::env::temp_dir().join(format!("wom-it-{name}"));
        std::fs::remove_dir_all(&base).ok();
        let home = base.join("home");
        let corpus = base.join("corpus");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&corpus).unwrap();
        build_fixture(&corpus);
        Self { home, corpus }
    }

    fn bin() -> PathBuf {
        // Cargo puts integration-test binaries next to the crate binaries.
        let mut p = std::env::current_exe().unwrap();
        p.pop();
        if p.ends_with("deps") {
            p.pop();
        }
        p.join("wom")
    }

    fn run(&self, args: &[&str]) -> (String, String, bool) {
        let out = Command::new(Self::bin())
            .args(args)
            .env("WOM_HOME", &self.home)
            // Models are large; share one cache across harnesses when the caller
            // provides it, so the suite does not re-download per test.
            .env_remove("EDITOR")
            .env_remove("VISUAL")
            .output()
            .expect("running wom");
        (
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
            out.status.success(),
        )
    }

    /// Result lines for a query, with the corpus prefix stripped for readability.
    fn search(&self, args: &[&str]) -> Vec<String> {
        let (stdout, stderr, ok) = self.run(args);
        assert!(ok, "search {args:?} failed: {stderr}");
        stdout
            .lines()
            .map(|l| l.replace(&format!("{}/", self.corpus.display()), ""))
            .collect()
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if let Some(base) = self.home.parent() {
            std::fs::remove_dir_all(base).ok();
        }
    }
}

fn model_available() -> bool {
    std::env::var("WOM_SKIP_MODEL").is_err()
}

/// Assert some result line contains `needle`.
fn assert_contains(results: &[String], needle: &str, what: &str) {
    assert!(
        results.iter().any(|r| r.contains(needle)),
        "{what}: no result contained {needle:?}\n  got: {results:#?}"
    );
}

fn assert_none_contains(results: &[String], needle: &str, what: &str) {
    assert!(
        !results.iter().any(|r| r.contains(needle)),
        "{what}: a result unexpectedly contained {needle:?}\n  got: {results:#?}"
    );
}

#[test]
fn lexical_search_works_without_a_model() {
    let h = Harness::new("lexical");
    let corpus = h.corpus.to_string_lossy().into_owned();

    let (_, err, ok) = h.run(&["init", "--add", &corpus, "--yes"]);
    assert!(ok, "init failed: {err}");

    // Model-free fast path: builds SQLite+FTS only, never downloads a model.
    // This test always runs, even under WOM_SKIP_MODEL=1.
    let (_, err, ok) = h.run(&["index", "--no-embed", "--now"]);
    assert!(ok, "index --no-embed failed: {err}");

    // Directly from the spec's second example.
    let results = h.search(&["--lexical", "bosnia", "--no-tui"]);
    assert_contains(&results, "bosnia_croatia_trip_aug2024", "wom bosnia (lexical)");

    // The filename must be tokenised on underscores for this to match.
    let results = h.search(&["--lexical", "sarajevo", "--no-tui"]);
    assert_contains(&results, "dubrovnik_to_sarajevo", "wom sarajevo (lexical)");
}

#[test]
fn no_embed_index_reports_zero_embedded_and_stays_lexically_searchable() {
    // Cheap regression for the `--no-embed` flag itself: metadata + FTS land,
    // vectors do not, and `--lexical` still answers.
    let h = Harness::new("noembed");
    let corpus = h.corpus.to_string_lossy().into_owned();
    h.run(&["init", "--add", &corpus, "--yes"]);
    let (out, err, ok) = h.run(&["index", "--no-embed", "--now"]);
    assert!(ok, "index --no-embed failed: {err}");
    assert!(
        out.contains("0 embedded") || out.contains("embedded"),
        "expected embedded count in output, got: {out}"
    );
    let results = h.search(&["--lexical", "sourdough", "--no-tui"]);
    assert_contains(&results, "sourdough", "lexical after --no-embed");
}

#[test]
fn ignore_rules_keep_build_artefacts_out_of_the_index() {
    let h = Harness::new("ignores");
    let corpus = h.corpus.to_string_lossy().into_owned();
    h.run(&["init", "--add", &corpus, "--yes"]);
    // Lexical-only index suffices: ignore rules act at walk time.
    let (_, err, ok) = h.run(&["index", "--no-embed", "--now"]);
    assert!(ok, "index failed: {err}");

    // A query that would match the excluded files if they had been indexed.
    let results = h.search(&["--lexical", "leftpad", "-n", "50", "--no-tui"]);
    assert_none_contains(&results, "node_modules", "node_modules must be ignored");

    let results = h.search(&["--lexical", "config", "-n", "50", "--no-tui"]);
    assert_none_contains(&results, "/.git/", ".git must be ignored");
}

#[test]
fn spec_examples_return_their_expected_answers() {
    if !model_available() {
        return;
    }
    let h = Harness::new("spec");
    let corpus = h.corpus.to_string_lossy().into_owned();
    h.run(&["init", "--add", &corpus, "--yes"]);
    let (_, err, ok) = h.run(&["index"]);
    assert!(ok, "index failed: {err}");

    // Example 1: employment documents -> the directory and the W-2s.
    let r = h.search(&["employment", "documents", "--no-tui"]);
    assert_contains(&r, "directory:", "example 1 should surface a directory");
    assert_contains(&r, "employment_docs", "example 1");

    // Example 2: bosnia -> the trip directory.
    let r = h.search(&["bosnia", "--no-tui"]);
    assert_contains(&r, "bosnia_croatia_trip_aug2024", "example 2");

    // Example 3: scoped search for code.
    let scope = h.corpus.join("projects").to_string_lossy().into_owned();
    let r = h.search(&[&scope, "sensors", "monitor", "--no-tui"]);
    assert_contains(&r, "esp32", "example 3");
    // Scoping must exclude everything outside the named directory.
    assert_none_contains(&r, "employment_docs", "example 3 scope");
    assert_none_contains(&r, "recipes", "example 3 scope");
}

#[test]
fn semantic_queries_find_documents_that_share_no_words() {
    if !model_available() {
        return;
    }
    let h = Harness::new("semantic");
    let corpus = h.corpus.to_string_lossy().into_owned();
    h.run(&["init", "--add", &corpus, "--yes"]);
    let (_, err, ok) = h.run(&["index"]);
    assert!(ok, "index failed: {err}");

    // None of these queries share a word with the path they should find, so a
    // lexical index cannot answer them. This is the actual point of the tool, and
    // it is what breaks if pooling or the query prefix is wrong.
    for (query, expect) in [
        ("tax forms from my employer", "employment"),
        ("balkans holiday", "bosnia"),
        ("bread baking", "sourdough"),
        ("online payment flow", "checkout"),
    ] {
        let words: Vec<&str> = query.split(' ').collect();
        let mut args = words.clone();
        args.push("--no-tui");
        let r = h.search(&args);
        assert_contains(&r, expect, &format!("semantic query {query:?}"));
    }
}

#[test]
fn piped_output_is_plain_lines() {
    if !model_available() {
        return;
    }
    let h = Harness::new("piped");
    let corpus = h.corpus.to_string_lossy().into_owned();
    h.run(&["init", "--add", &corpus, "--yes"]);
    h.run(&["index"]);

    // Command::output() gives a non-tty stdout, so this exercises the piped path
    // the spec's examples show.
    let (stdout, _, ok) = h.run(&["bosnia"]);
    assert!(ok);
    for line in stdout.lines() {
        assert!(
            line.starts_with('/') || line.starts_with("directory:/"),
            "expected a bare path or directory: line, got {line:?}"
        );
    }
}

#[test]
fn reindex_is_incremental_and_notices_deletions() {
    if !model_available() {
        return;
    }
    let h = Harness::new("incremental");
    let corpus = h.corpus.to_string_lossy().into_owned();
    h.run(&["init", "--add", &corpus, "--yes"]);
    let (out, err, ok) = h.run(&["index"]);
    assert!(ok, "first index failed: {err}");
    assert!(out.contains("new"), "unexpected first-index output: {out}");

    // Nothing changed: no re-embedding at all.
    let (out, _, ok) = h.run(&["index", "--now"]);
    assert!(ok);
    assert!(
        out.contains("0 embedded"),
        "an unchanged rescan re-embedded files: {out}"
    );

    // Touching a file changes its mtime but not its text, so the content hash
    // should still spare it the model.
    let target = h.corpus.join("recipes/sourdough.md");
    let content = std::fs::read(&target).unwrap();
    std::fs::write(&target, &content).unwrap();
    let (out, _, ok) = h.run(&["index", "--now"]);
    assert!(ok);
    assert!(
        out.contains("0 embedded"),
        "a touched-but-unchanged file was re-embedded: {out}"
    );

    // Editing it must re-embed exactly that file.
    std::fs::write(&target, "# Sourdough\nNow with rye flour and a longer autolyse.").unwrap();
    let (out, _, ok) = h.run(&["index", "--now"]);
    assert!(ok);
    assert!(
        out.contains("1 embedded"),
        "an edited file was not re-embedded: {out}"
    );

    // Deleting it must remove it from results.
    std::fs::remove_file(&target).unwrap();
    let (out, _, ok) = h.run(&["index", "--now"]);
    assert!(ok);
    assert!(out.contains("removed 1 files"), "deletion not noticed: {out}");

    let r = h.search(&["--lexical", "sourdough", "--no-tui"]);
    assert_none_contains(&r, "sourdough", "deleted file still returned");
}

#[test]
fn status_verify_reports_a_consistent_index() {
    if !model_available() {
        return;
    }
    let h = Harness::new("verify");
    let corpus = h.corpus.to_string_lossy().into_owned();
    h.run(&["init", "--add", &corpus, "--yes"]);
    h.run(&["index"]);

    let (out, err, ok) = h.run(&["status", "--verify"]);
    assert!(ok, "status failed: {err}");
    assert!(
        out.contains("consistent") && !out.contains("INCONSISTENT"),
        "verify did not pass on a freshly built index: {out}"
    );
}

#[test]
fn a_query_with_no_index_fails_with_a_useful_message() {
    let h = Harness::new("noindex");
    let (_, err, ok) = h.run(&["something"]);
    assert!(!ok, "querying without an index should fail");
    assert!(
        err.contains("wom init") || err.contains("no index"),
        "unhelpful error: {err}"
    );
}
