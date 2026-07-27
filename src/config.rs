//! Configuration and on-disk layout.
//!
//! Config lives at `$XDG_CONFIG_HOME/wom/config.toml`, data (index, vectors,
//! models) under `$XDG_DATA_HOME/wom/`.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// How much of a file's content we bother embedding. Chosen per root, because
/// a 5000-line kernel driver and a two-page W-2 want very different budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
    /// Path + filename + first ~256 tokens of extracted text. For document dirs.
    Content,
    /// Path + filename + first ~64 tokens. For large source trees, where the
    /// header comment, imports and leading declarations are what identify a file
    /// and the middle contributes only noise.
    Code,
    /// Path and filename only. For media dirs.
    NameOnly,
}

impl Profile {
    /// Token budget for the extracted-body portion of the embedded text.
    pub fn body_tokens(self) -> usize {
        match self {
            Profile::Content => 256,
            Profile::Code => 64,
            Profile::NameOnly => 0,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Profile::Content => "content",
            Profile::Code => "code",
            Profile::NameOnly => "name-only",
        }
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "content" => Some(Profile::Content),
            "code" => Some(Profile::Code),
            "name-only" => Some(Profile::NameOnly),
            _ => None,
        }
    }
}

/// Which compute device to run the model on (spec item 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Device {
    /// CPU only.
    #[default]
    Cpu,
    /// Try the GPU, fall back to CPU with a warning if it is unavailable.
    Auto,
    /// Require the GPU; fail rather than silently running on the CPU.
    Gpu,
}

impl Device {
    pub fn wants_gpu(self) -> bool {
        matches!(self, Device::Auto | Device::Gpu)
    }

    /// Whether falling back to CPU is acceptable.
    pub fn may_fall_back(self) -> bool {
        !matches!(self, Device::Gpu)
    }
}

/// When to look for changes on disk (spec item 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Refresh {
    EveryRun,
    Daily,
    Weekly,
    Manual,
}

impl Refresh {
    /// Seconds after `last_scan_at` at which a rescan becomes due.
    pub fn interval_secs(self) -> Option<u64> {
        match self {
            Refresh::EveryRun => Some(0),
            Refresh::Daily => Some(24 * 60 * 60),
            Refresh::Weekly => Some(7 * 24 * 60 * 60),
            Refresh::Manual => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Root {
    pub path: PathBuf,
    pub profile: Profile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractConfig {
    /// Run exiftool/ffprobe over images, audio and video. Off by default: it
    /// spawns a process per file, which dominates walk time on a Pictures dir.
    pub media: bool,
    /// Largest file we will open for extraction.
    pub max_file_size_mb: u64,
    /// Bytes of a text file read before truncating.
    pub max_read_kb: u64,
    /// PDF pages passed to pdftotext.
    pub pdf_pages: u32,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            media: false,
            max_file_size_mb: 20,
            max_read_kb: 64,
            pdf_pages: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IgnoreConfig {
    /// Honour `.gitignore`/`.ignore` files found while walking.
    pub use_gitignore: bool,
    /// Walk into dotfiles and dot-directories.
    pub hidden: bool,
    /// Follow symlinks. Off by default — cycles and duplicate work.
    pub follow_symlinks: bool,
    /// Extra glob patterns to skip, appended to the builtin list.
    pub extra: Vec<String>,
    /// Builtin patterns to *not* apply, for when a default is wrong for you.
    pub unignore: Vec<String>,
}

impl Default for IgnoreConfig {
    fn default() -> Self {
        Self {
            use_gitignore: true,
            hidden: false,
            follow_symlinks: false,
            extra: Vec::new(),
            unignore: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Model id, e.g. `bge-small-en-v1.5-int8`. See `embed::registry`.
    pub model: String,
    /// Where to run the model. Only has an effect in a build with `--features gpu`.
    pub device: Device,
    pub refresh: Refresh,
    /// Results returned by a query.
    pub limit: usize,
    /// Overrides the configured model's own similarity floor. `None` uses
    /// `ModelSpec::default_min_similarity`, because the right value depends on how
    /// the model spaces related from unrelated text — the same reason pooling and
    /// the query prefix live in the registry.
    pub min_similarity: Option<f32>,
    pub roots: Vec<Root>,
    pub extract: ExtractConfig,
    pub ignore: IgnoreConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: crate::embed::DEFAULT_MODEL.to_string(),
            device: Device::default(),
            refresh: Refresh::Daily,
            limit: 40,
            min_similarity: None,
            roots: Vec::new(),
            extract: ExtractConfig::default(),
            ignore: IgnoreConfig::default(),
        }
    }
}

impl Config {
    pub fn load(paths: &Paths) -> Result<Self> {
        let p = paths.config_file();
        if !p.exists() {
            return Ok(Config::default());
        }
        let text = std::fs::read_to_string(&p)
            .with_context(|| format!("reading config {}", p.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing config {}", p.display()))
    }

    pub fn save(&self, paths: &Paths) -> Result<()> {
        let p = paths.config_file();
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let text = toml::to_string_pretty(self).context("serialising config")?;
        std::fs::write(&p, text).with_context(|| format!("writing config {}", p.display()))
    }

    /// The profile that applies to `path`: the most specific configured root
    /// containing it. Longest prefix wins so a nested root can override its
    /// parent.
    pub fn profile_for(&self, path: &Path) -> Option<Profile> {
        self.roots
            .iter()
            .filter(|r| path.starts_with(&r.path))
            .max_by_key(|r| r.path.as_os_str().len())
            .map(|r| r.profile)
    }

    /// Similarity floor for the configured model, unless overridden in config.
    pub fn resolved_min_similarity(&self) -> f32 {
        self.min_similarity.unwrap_or_else(|| {
            crate::embed::lookup(&self.model)
                .map(|m| m.default_min_similarity)
                .unwrap_or(0.55)
        })
    }
}

/// The most specific configured root containing `p`, or `None` if it lies outside
/// all of them.
///
/// Longest prefix wins, so a nested root overrides its parent. This is the rule
/// `Config::profile_for` and the directory rollup both depend on, so it has one
/// implementation.
pub fn deepest_root<'a>(roots: &'a [PathBuf], p: &Path) -> Option<&'a Path> {
    roots
        .iter()
        .filter(|r| p.starts_with(r))
        .max_by_key(|r| r.as_os_str().len())
        .map(|r| r.as_path())
}

/// Resolved filesystem locations. Constructed once and threaded through.
#[derive(Debug, Clone)]
pub struct Paths {
    config_dir: PathBuf,
    data_dir: PathBuf,
}

impl Paths {
    pub fn resolve() -> Result<Self> {
        // WOM_HOME collapses config and data into one directory. Tests and the
        // benchmark harness rely on this to stay clear of the real index.
        if let Some(home) = std::env::var_os("WOM_HOME") {
            let home = PathBuf::from(home);
            return Ok(Self {
                config_dir: home.clone(),
                data_dir: home,
            });
        }
        let dirs = directories::ProjectDirs::from("", "", "wom")
            .context("cannot determine XDG config/data directories")?;
        Ok(Self {
            config_dir: dirs.config_dir().to_path_buf(),
            data_dir: dirs.data_dir().to_path_buf(),
        })
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
    pub fn config_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }
    pub fn db_file(&self) -> PathBuf {
        self.data_dir.join("index.db")
    }
    pub fn file_vectors(&self) -> PathBuf {
        self.data_dir.join("vectors_files.i8")
    }
    pub fn dir_vectors(&self) -> PathBuf {
        self.data_dir.join("vectors_dirs.i8")
    }
    pub fn scan_lock(&self) -> PathBuf {
        self.data_dir.join("scan.lock")
    }
    pub fn model_dir(&self, model_id: &str) -> PathBuf {
        self.data_dir.join("models").join(model_id)
    }

    /// Every file `wom` derives from the filesystem, i.e. everything
    /// `wom index --rebuild` may safely delete. Kept here because `Paths` is the
    /// one place that knows the on-disk layout; a caller re-listing these would
    /// silently stop covering any artefact added later.
    pub fn derived_files(&self) -> Vec<PathBuf> {
        let db = self.db_file();
        vec![
            db.clone(),
            db.with_extension("db-wal"),
            db.with_extension("db-shm"),
            self.file_vectors(),
            self.dir_vectors(),
        ]
    }

    pub fn ensure_data_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.data_dir)
            .with_context(|| format!("creating data dir {}", self.data_dir.display()))
    }
}

/// Expand a leading `~` and make the path absolute, without requiring it to
/// exist yet.
pub fn expand_path(s: &str) -> Result<PathBuf> {
    let expanded = if s == "~" {
        home_dir()?
    } else if let Some(rest) = s.strip_prefix("~/") {
        home_dir()?.join(rest)
    } else {
        PathBuf::from(s)
    };
    if expanded.is_absolute() {
        Ok(normalise(&expanded))
    } else {
        let cwd = std::env::current_dir().context("getting current directory")?;
        Ok(normalise(&cwd.join(expanded)))
    }
}

fn home_dir() -> Result<PathBuf> {
    match std::env::var_os("HOME") {
        Some(h) if !h.is_empty() => Ok(PathBuf::from(h)),
        _ => bail!("$HOME is not set, cannot expand `~`"),
    }
}

/// Lexically remove `.`/`..` and trailing slashes. Deliberately *not*
/// `canonicalize`: we do not want to resolve symlinks, since indexed paths
/// should read back the way the user typed them.
fn normalise(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_strips_dots_and_trailing_slash() {
        assert_eq!(normalise(Path::new("/a/./b/")), PathBuf::from("/a/b"));
        assert_eq!(normalise(Path::new("/a/b/../c")), PathBuf::from("/a/c"));
    }

    #[test]
    fn expand_tilde_uses_home() {
        // SAFETY: single-threaded test, no other thread reads the environment.
        unsafe { std::env::set_var("HOME", "/home/tester") };
        assert_eq!(
            expand_path("~/Documents").unwrap(),
            PathBuf::from("/home/tester/Documents")
        );
        assert_eq!(expand_path("~").unwrap(), PathBuf::from("/home/tester"));
    }

    #[test]
    fn profile_for_prefers_most_specific_root() {
        let cfg = Config {
            roots: vec![
                Root {
                    path: PathBuf::from("/home/n/projects"),
                    profile: Profile::Code,
                },
                Root {
                    path: PathBuf::from("/home/n/projects/notes"),
                    profile: Profile::Content,
                },
            ],
            ..Config::default()
        };
        assert_eq!(
            cfg.profile_for(Path::new("/home/n/projects/a/b.c")),
            Some(Profile::Code)
        );
        assert_eq!(
            cfg.profile_for(Path::new("/home/n/projects/notes/x.md")),
            Some(Profile::Content)
        );
        assert_eq!(cfg.profile_for(Path::new("/etc/passwd")), None);
    }

    #[test]
    fn device_gpu_does_not_permit_a_silent_cpu_fallback() {
        assert!(Device::Gpu.wants_gpu());
        assert!(!Device::Gpu.may_fall_back());
        assert!(Device::Auto.wants_gpu());
        assert!(Device::Auto.may_fall_back());
        assert!(!Device::Cpu.wants_gpu());
    }

    #[test]
    fn device_defaults_to_cpu() {
        assert_eq!(Config::default().device, Device::Cpu);
    }

    #[test]
    fn config_roundtrips_through_toml() {
        let cfg = Config {
            roots: vec![Root {
                path: PathBuf::from("/home/n/Documents"),
                profile: Profile::Content,
            }],
            refresh: Refresh::Weekly,
            ..Config::default()
        };
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(back.refresh, Refresh::Weekly);
        assert_eq!(back.roots.len(), 1);
        assert_eq!(back.roots[0].profile, Profile::Content);
    }
}
