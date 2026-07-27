//! Directory traversal and the ignore rules.
//!
//! Sizing note that drives the defaults here: the author's XDG directories hold
//! ~450k files, ~357k of which survive basic ignore rules, because `~/projects`
//! contains full checked-out source trees. Being aggressive by default is not an
//! optimisation, it is the difference between a usable tool and an hours-long
//! first run.

use crate::config::{Config, IgnoreConfig, Profile};
use anyhow::{Context, Result};
use ignore::overrides::OverrideBuilder;
use ignore::{WalkBuilder, WalkState};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Patterns skipped unless the user unignores them. Two groups: directories that
/// are caches or dependency trees, and file types with no retrievable text.
pub const BUILTIN_IGNORES: &[&str] = &[
    // Version control and editor state.
    ".git/",
    ".hg/",
    ".svn/",
    ".bzr/",
    ".jj/",
    // Language and build caches.
    "node_modules/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".tox/",
    ".venv/",
    "venv/",
    "site-packages/",
    "target/debug/",
    "target/release/",
    ".gradle/",
    ".m2/",
    ".cargo/registry/",
    ".rustup/",
    ".npm/",
    ".pnpm-store/",
    ".yarn/",
    ".next/",
    ".nuxt/",
    ".svelte-kit/",
    ".terraform/",
    "vendor/",
    "third_party/",
    // Generic cache and trash locations.
    ".cache/",
    ".local/share/Trash/",
    ".thumbnails/",
    ".Trash-*/",
    // Compiled objects and archives: no text worth embedding.
    "*.pyc",
    "*.pyo",
    "*.o",
    "*.a",
    "*.so",
    "*.so.*",
    "*.ko",
    "*.obj",
    "*.lib",
    "*.dll",
    "*.dylib",
    "*.class",
    "*.jar",
    "*.war",
    "*.rlib",
    "*.rmeta",
    "*.d",
    "*.cmd",
    "*.gcda",
    "*.gcno",
    // Archives and images: opaque blobs.
    "*.tar",
    "*.tar.*",
    "*.tgz",
    "*.gz",
    "*.bz2",
    "*.xz",
    "*.zst",
    "*.7z",
    "*.rar",
    "*.deb",
    "*.rpm",
    "*.apk",
    "*.AppImage",
    "*.iso",
    "*.img",
    "*.qcow2",
    "*.vmdk",
    "*.dmg",
    // Machine-generated bookkeeping.
    "*.lock",
    "*.pack",
    "*.idx",
    "*.map",
    "*.min.js",
    "*.min.css",
    "*.mo",
    "*.pdb",
    "*.sqlite-wal",
    "*.sqlite-shm",
    // Fonts.
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.otf",
    "*.eot",
];

/// Extensions counted as source code when auto-detecting a root's profile.
const SOURCE_EXTS: &[&str] = &[
    "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "py", "rs", "go", "js", "mjs", "cjs", "ts", "tsx",
    "jsx", "java", "kt", "kts", "scala", "swift", "m", "mm", "cs", "rb", "php", "pl", "pm", "lua",
    "sh", "bash", "zsh", "fish", "sql", "s", "asm", "dts", "dtsi", "vhd", "v", "sv", "ex", "exs",
    "erl", "hs", "ml", "clj", "jl", "r", "f90", "for", "pas", "zig", "nim", "dart",
];

/// A file the walker decided to index.
pub struct Found {
    pub path: PathBuf,
    pub size: u64,
    pub mtime_ns: i64,
}

/// Build the ignore-aware walker for one root.
fn builder(root: &Path, ic: &IgnoreConfig, max_file_size_mb: u64) -> Result<WalkBuilder> {
    let mut ov = OverrideBuilder::new(root);
    // In the `ignore` crate a bare glob is a *whitelist*; prefixing `!` makes it
    // an exclusion. We only ever exclude, so every pattern is negated.
    let unignored: std::collections::HashSet<&str> =
        ic.unignore.iter().map(|s| s.as_str()).collect();
    for pat in BUILTIN_IGNORES {
        if unignored.contains(pat) {
            continue;
        }
        ov.add(&format!("!{pat}"))
            .with_context(|| format!("adding builtin ignore glob {pat}"))?;
    }
    for pat in &ic.extra {
        ov.add(&format!("!{pat}"))
            .with_context(|| format!("adding configured ignore glob {pat}"))?;
    }
    let overrides = ov.build().context("compiling ignore globs")?;

    let mut b = WalkBuilder::new(root);
    b.overrides(overrides)
        .hidden(!ic.hidden)
        .follow_links(ic.follow_symlinks)
        .git_ignore(ic.use_gitignore)
        .git_global(ic.use_gitignore)
        .git_exclude(ic.use_gitignore)
        .ignore(ic.use_gitignore)
        .parents(false)
        .max_filesize(Some(max_file_size_mb * 1024 * 1024))
        .threads(
            std::thread::available_parallelism()
                .map(|n| n.get().min(8))
                .unwrap_or(4),
        );
    Ok(b)
}

/// Walk `root` in parallel, calling `sink` for every file that passes the filters.
///
/// Returns how many entries were skipped because they could not be read. An
/// unreadable directory must not abort a 274k-file scan, but it must not be
/// invisible either — a whole subtree silently missing from the index is exactly
/// the kind of thing a user needs told.
///
/// `sink` runs on many threads at once and must handle that itself.
pub fn walk_root<F>(root: &Path, cfg: &Config, sink: F) -> Result<usize>
where
    F: Fn(Found) + Send + Sync,
{
    let skipped = AtomicUsize::new(0);
    let b = builder(root, &cfg.ignore, cfg.extract.max_file_size_mb)?;
    b.build_parallel().run(|| {
        Box::new(|entry| {
            let Ok(entry) = entry else {
                skipped.fetch_add(1, Ordering::Relaxed);
                return WalkState::Continue;
            };
            if entry.file_type().is_none_or(|t| !t.is_file()) {
                return WalkState::Continue;
            }
            if let Ok(md) = entry.metadata() {
                sink(Found {
                    path: entry.into_path(),
                    size: md.len(),
                    mtime_ns: mtime_ns(&md),
                });
            }
            WalkState::Continue
        })
    });
    Ok(skipped.load(Ordering::Relaxed))
}

/// Files a probe will look at before giving up and assuming a large tree.
///
/// Lives here rather than at the call site because `Probe::suggest` keys off
/// `truncated`, which this value defines — splitting them would let a caller
/// silently change which trees get the `Code` profile.
pub const PROBE_CAP: usize = 60_000;

/// What a cheap look at a directory suggests about how to index it.
#[derive(Debug, Clone, Copy)]
pub struct Probe {
    /// Files seen, capped at the probe limit.
    pub files: usize,
    /// True if the walk stopped early, so `files` is a floor not a total.
    pub truncated: bool,
    pub source_files: usize,
}

impl Probe {
    pub fn source_ratio(&self) -> f64 {
        if self.files == 0 {
            0.0
        } else {
            self.source_files as f64 / self.files as f64
        }
    }

    /// Profile suggested for a root with these characteristics.
    ///
    /// Large or source-dominated trees get `Code`: for a 5000-line driver the
    /// leading declarations identify the file and the rest is noise, so the
    /// cheaper profile is also the more accurate one.
    pub fn suggest(&self) -> Profile {
        if self.truncated || self.files > 20_000 || self.source_ratio() > 0.6 {
            Profile::Code
        } else {
            Profile::Content
        }
    }
}

/// Count files under `root`, stopping at [`PROBE_CAP`]. Used by `wom init` to show
/// the real cost before committing to it.
pub fn probe(root: &Path, cfg: &Config) -> Result<Probe> {
    let cap = PROBE_CAP;
    let files = AtomicUsize::new(0);
    let source = AtomicUsize::new(0);
    let stopped = AtomicUsize::new(0);

    let b = builder(root, &cfg.ignore, cfg.extract.max_file_size_mb)?;
    b.build_parallel().run(|| {
        Box::new(|entry| {
            let Ok(entry) = entry else {
                return WalkState::Continue;
            };
            if entry.file_type().is_none_or(|t| !t.is_file()) {
                return WalkState::Continue;
            }
            let n = files.fetch_add(1, Ordering::Relaxed) + 1;
            if is_source(entry.path()) {
                source.fetch_add(1, Ordering::Relaxed);
            }
            if n >= cap {
                stopped.store(1, Ordering::Relaxed);
                return WalkState::Quit;
            }
            WalkState::Continue
        })
    });

    Ok(Probe {
        files: files.load(Ordering::Relaxed).min(cap),
        truncated: stopped.load(Ordering::Relaxed) == 1,
        source_files: source.load(Ordering::Relaxed),
    })
}

fn is_source(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| SOURCE_EXTS.contains(&e.as_str()))
}

pub fn mtime_ns(md: &std::fs::Metadata) -> i64 {
    use std::os::unix::fs::MetadataExt;
    md.mtime()
        .saturating_mul(1_000_000_000)
        .saturating_add(md.mtime_nsec())
}

/// Drop roots that are nested inside another root, so a subtree is not walked
/// twice. The surviving root's profile then applies, and a deliberate override
/// for a subdirectory is still expressible via `Config::profile_for`, which
/// prefers the longest prefix.
pub fn dedupe_nested(roots: &mut Vec<PathBuf>) {
    roots.sort();
    roots.dedup();
    let mut keep: Vec<PathBuf> = Vec::with_capacity(roots.len());
    for r in roots.iter() {
        if keep.iter().any(|k| r.starts_with(k)) {
            continue;
        }
        keep.push(r.clone());
    }
    *roots = keep;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_ignores_are_unique() {
        let mut v = BUILTIN_IGNORES.to_vec();
        v.sort_unstable();
        let n = v.len();
        v.dedup();
        assert_eq!(v.len(), n, "duplicate pattern in BUILTIN_IGNORES");
    }

    #[test]
    fn source_detection_is_case_insensitive() {
        assert!(is_source(Path::new("/a/main.C")));
        assert!(is_source(Path::new("/a/b.rs")));
        assert!(!is_source(Path::new("/a/notes.txt")));
        assert!(!is_source(Path::new("/a/noext")));
    }

    #[test]
    fn large_or_source_heavy_trees_suggest_the_code_profile() {
        let big = Probe {
            files: 30_000,
            truncated: false,
            source_files: 0,
        };
        assert_eq!(big.suggest(), Profile::Code);

        let sourcey = Probe {
            files: 100,
            truncated: false,
            source_files: 80,
        };
        assert_eq!(sourcey.suggest(), Profile::Code);

        let docs = Probe {
            files: 500,
            truncated: false,
            source_files: 10,
        };
        assert_eq!(docs.suggest(), Profile::Content);
    }

    #[test]
    fn truncated_probe_assumes_a_large_tree() {
        let p = Probe {
            files: 50_000,
            truncated: true,
            source_files: 0,
        };
        assert_eq!(p.suggest(), Profile::Code);
    }

    #[test]
    fn empty_probe_has_zero_ratio_and_no_division_by_zero() {
        let p = Probe {
            files: 0,
            truncated: false,
            source_files: 0,
        };
        assert_eq!(p.source_ratio(), 0.0);
        assert_eq!(p.suggest(), Profile::Content);
    }

    #[test]
    fn nested_roots_are_dropped() {
        let mut roots = vec![
            PathBuf::from("/home/n/projects/sub"),
            PathBuf::from("/home/n/projects"),
            PathBuf::from("/home/n/Documents"),
        ];
        dedupe_nested(&mut roots);
        assert_eq!(
            roots,
            vec![
                PathBuf::from("/home/n/Documents"),
                PathBuf::from("/home/n/projects"),
            ]
        );
    }

    #[test]
    fn sibling_with_shared_prefix_is_kept() {
        // `projects-old` starts with the string "projects" but is not nested.
        let mut roots = vec![
            PathBuf::from("/home/n/projects"),
            PathBuf::from("/home/n/projects-old"),
        ];
        dedupe_nested(&mut roots);
        assert_eq!(roots.len(), 2, "sibling directory was wrongly treated as nested");
    }

    #[test]
    fn walker_skips_ignored_directories_and_extensions() {
        let tmp = std::env::temp_dir().join("wom-walk-test");
        std::fs::remove_dir_all(&tmp).ok();
        std::fs::create_dir_all(tmp.join("node_modules/pkg")).unwrap();
        std::fs::create_dir_all(tmp.join("src")).unwrap();
        std::fs::write(tmp.join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(tmp.join("src/main.o"), [0u8; 8]).unwrap();
        std::fs::write(tmp.join("node_modules/pkg/index.js"), "x").unwrap();
        std::fs::write(tmp.join("notes.txt"), "hello").unwrap();

        let cfg = Config::default();
        let found = std::sync::Mutex::new(Vec::new());
        let skipped = walk_root(&tmp, &cfg, |f| {
            found.lock().unwrap().push(f.path);
        })
        .unwrap();
        assert_eq!(skipped, 0, "nothing in the fixture should be unreadable");
        let mut names: Vec<String> = found
            .lock()
            .unwrap()
            .iter()
            .map(|p| p.strip_prefix(&tmp).unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();

        std::fs::remove_dir_all(&tmp).ok();
        assert_eq!(names, vec!["notes.txt", "src/main.rs"]);
    }
}
