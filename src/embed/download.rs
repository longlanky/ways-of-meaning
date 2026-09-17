//! Fetching model files from Hugging Face on first use.
//!
//! Only two files per model (the ONNX graph and the tokenizer), so this is a
//! plain blocking download rather than a dependency on `hf-hub`.

use crate::config::Paths;
use crate::embed::ModelSpec;
use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Where the two files for `spec` live once fetched.
pub struct ModelFiles {
    pub onnx: PathBuf,
    pub tokenizer: PathBuf,
}

/// Resolve `spec` to local files, downloading anything missing.
pub fn ensure_model(paths: &Paths, spec: &ModelSpec) -> Result<ModelFiles> {
    ensure_files(
        paths,
        spec.id,
        spec.repo,
        spec.onnx_path,
        spec.tokenizer_path,
        spec.approx_mb,
        spec.onnx_hash,
        spec.tokenizer_hash,
    )
}

/// blake3 hex digest of a file, streaming so multi-GB graphs never sit in RAM.
pub fn file_blake3(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)
        .with_context(|| format!("hashing {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf).context("reading file for hash")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn verify_pin(path: &Path, expected: Option<&str>, label: &str) -> Result<()> {
    let Some(want) = expected else { return Ok(()) };
    let got = file_blake3(path)?;
    if got != want {
        anyhow::bail!(
            "{label} at {} failed hash pin: expected {want}, got {got}. \
             Removed nothing; delete the file and re-run to re-download",
            path.display()
        );
    }
    Ok(())
}

/// Resolve a (graph, tokenizer) pair from a Hugging Face repo to local files,
/// downloading anything missing. Shared by embedding models and rerankers,
/// which differ only in what runs the files.
///
/// When `onnx_hash`/`tokenizer_hash` pins are present, existing files are
/// verified and mismatches are deleted + re-downloaded once; a second mismatch
/// errors instead of looping.
pub fn ensure_files(
    paths: &Paths,
    id: &str,
    repo: &str,
    onnx_path: &str,
    tokenizer_path: &str,
    approx_mb: u64,
    onnx_hash: Option<&str>,
    tokenizer_hash: Option<&str>,
) -> Result<ModelFiles> {
    let dir = paths.model_dir(id);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let onnx = dir.join("model.onnx");
    let tokenizer = dir.join("tokenizer.json");

    if tokenizer.exists() {
        if let Err(e) = verify_pin(&tokenizer, tokenizer_hash, "tokenizer") {
            eprintln!("wom: {e}; re-downloading");
            std::fs::remove_file(&tokenizer).ok();
        }
    }
    if !tokenizer.exists() {
        fetch(&hf_url(repo, tokenizer_path), &tokenizer, "tokenizer")?;
        verify_pin(&tokenizer, tokenizer_hash, "tokenizer")?;
    }
    if onnx.exists() {
        if let Err(e) = verify_pin(&onnx, onnx_hash, "model.onnx") {
            eprintln!("wom: {e}; re-downloading");
            std::fs::remove_file(&onnx).ok();
        }
    }
    if !onnx.exists() {
        eprintln!("wom: fetching model {id} (~{approx_mb} MB) from {repo}");
        fetch(&hf_url(repo, onnx_path), &onnx, "model.onnx")?;
        verify_pin(&onnx, onnx_hash, "model.onnx")?;
    }

    // A truncated download leaves a file that loads as a corrupt graph with an
    // opaque error, so sanity-check the size instead.
    let got = std::fs::metadata(&onnx)?.len();
    if got < 1_000_000 {
        std::fs::remove_file(&onnx).ok();
        bail!(
            "downloaded model for {id} was only {got} bytes, which cannot be right; \
             removed it, please re-run",
        );
    }

    Ok(ModelFiles { onnx, tokenizer })
}

fn hf_url(repo: &str, path: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{path}")
}

/// Download to a temporary sibling then rename, so an interrupted fetch never
/// leaves a half-written file that the existence check above would accept.
fn fetch(url: &str, dest: &Path, label: &str) -> Result<()> {
    let mut resp = ureq::get(url)
        .call()
        .with_context(|| format!("requesting {url}"))?;

    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());

    let bar = match total {
        Some(n) => {
            let b = ProgressBar::new(n);
            b.set_style(
                ProgressStyle::with_template(
                    "  {msg} [{bar:30}] {bytes}/{total_bytes} {bytes_per_sec}",
                )
                .unwrap()
                .progress_chars("=> "),
            );
            b
        }
        None => ProgressBar::new_spinner(),
    };
    bar.set_message(label.to_string());

    let tmp = dest.with_extension("part");
    let mut written: u64 = 0;
    {
        let mut out = std::fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        let mut reader = resp.body_mut().as_reader();
        let mut buf = vec![0u8; 256 * 1024];
        loop {
            let n = reader.read(&mut buf).context("reading response body")?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])
                .with_context(|| format!("writing {}", tmp.display()))?;
            written += n as u64;
            bar.inc(n as u64);
        }
        out.flush()?;
    }
    bar.finish_and_clear();

    // An early-EOF body can otherwise be renamed into place and fail later as
    // an opaque "corrupt graph" on every run. When the server told us the size,
    // hold it to its word.
    if let Some(expected) = total {
        if written != expected {
            std::fs::remove_file(&tmp).ok();
            bail!(
                "download of {label} was truncated: got {written} of {expected} bytes; \
                 removed the partial file, please re-run"
            );
        }
    }

    std::fs::rename(&tmp, dest)
        .with_context(|| format!("renaming {} to {}", tmp.display(), dest.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_url_is_a_resolve_url() {
        assert_eq!(
            hf_url("BAAI/bge-small-en-v1.5", "onnx/model.onnx"),
            "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx"
        );
    }
}
