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
    let dir = paths.model_dir(spec.id);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let onnx = dir.join("model.onnx");
    let tokenizer = dir.join("tokenizer.json");

    if !tokenizer.exists() {
        fetch(&hf_url(spec.repo, spec.tokenizer_path), &tokenizer, "tokenizer")?;
    }
    if !onnx.exists() {
        eprintln!(
            "wom: fetching model {} (~{} MB) from {}",
            spec.id, spec.approx_mb, spec.repo
        );
        fetch(&hf_url(spec.repo, spec.onnx_path), &onnx, "model.onnx")?;
    }

    // A truncated download leaves a file that loads as a corrupt graph with an
    // opaque error, so sanity-check the size instead.
    let got = std::fs::metadata(&onnx)?.len();
    if got < 1_000_000 {
        std::fs::remove_file(&onnx).ok();
        bail!(
            "downloaded model for {} was only {got} bytes, which cannot be right; \
             removed it, please re-run",
            spec.id
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
            bar.inc(n as u64);
        }
        out.flush()?;
    }
    bar.finish_and_clear();

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
