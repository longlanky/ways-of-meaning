//! Command-line surface.
//!
//! The common case is `wom <words>` with no subcommand, so the parser inserts an
//! implicit `search` when the first argument is not a known subcommand.

use crate::config::expand_path;
use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "wom",
    about = "Ways of Meaning - semantic search over your files",
    version,
    disable_help_subcommand = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Search the index (the default; `wom <words>` works too).
    Search(SearchArgs),
    /// Choose directories to index and write the initial config.
    Init(InitArgs),
    /// Scan for changes and embed what is new.
    Index(IndexArgs),
    /// Show index size, model, and when it was last refreshed.
    Status(StatusArgs),
    /// Print the resolved config and its path.
    Config,
    /// Inspect or change the embedding model.
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
    /// Measure indexing throughput and retrieval quality.
    Bench(BenchArgs),
}

#[derive(Debug, Args)]
pub struct SearchArgs {
    /// Optional leading directories to restrict the search to, then the query.
    ///
    /// Not `trailing_var_arg`: that would swallow `--scores` and `--lexical` as
    /// query words, so flags could only ever appear before the query. A query
    /// word that genuinely starts with `-` can be passed after `--`.
    pub words: Vec<String>,

    /// Restrict to a directory. Unambiguous alternative to a leading path.
    #[arg(long = "in", short = 'i', value_name = "DIR")]
    pub scope: Vec<String>,

    /// Maximum results.
    #[arg(long, short = 'n')]
    pub limit: Option<usize>,

    /// Skip the embedding model and use only full-text matching. Useful for
    /// checking the index before a model is available.
    #[arg(long)]
    pub lexical: bool,

    /// Skip full-text matching and use only vector similarity.
    #[arg(long, conflicts_with = "lexical")]
    pub dense: bool,

    /// Print plain lines even when attached to a terminal.
    #[arg(long)]
    pub no_tui: bool,

    /// Show the fused score and cosine next to each result.
    #[arg(long)]
    pub scores: bool,

    /// Rerank the top candidates with a cross-encoder. Uses the configured
    /// reranker, or the default when the config has it off.
    #[arg(long)]
    pub rerank: bool,

    /// Rerank with a specific cross-encoder model (listed by `wom model list`).
    #[arg(long, value_name = "MODEL")]
    pub rerank_model: Option<String>,

    /// Disable reranking even when the config enables it.
    #[arg(long, conflicts_with = "rerank", conflicts_with = "rerank_model")]
    pub no_rerank: bool,

    /// Override the dense-similarity floor for this query.
    #[arg(long, value_name = "COSINE")]
    pub min_similarity: Option<f32>,
}

#[derive(Debug, Args)]
pub struct StatusArgs {
    /// Also check the index for internal inconsistency.
    #[arg(long)]
    pub verify: bool,
}

#[derive(Debug, Args)]
pub struct InitArgs {
    /// Directory to index. Repeatable. Defaults to the XDG user directories.
    #[arg(long = "add", short = 'a', value_name = "DIR")]
    pub add: Vec<String>,

    /// Force a profile for every added directory instead of auto-detecting.
    #[arg(long, value_parser = ["content", "code", "name-only"])]
    pub profile: Option<String>,

    /// Accept the proposed roots without prompting.
    #[arg(long, short = 'y')]
    pub yes: bool,

    /// Overwrite an existing config.
    #[arg(long)]
    pub force: bool,
}

#[derive(Debug, Args)]
pub struct IndexArgs {
    /// Scan now regardless of the configured refresh schedule.
    #[arg(long)]
    pub now: bool,

    /// Re-embed everything, ignoring the mtime and content-hash caches.
    #[arg(long)]
    pub force: bool,

    /// Discard the index and vectors, then build from scratch.
    #[arg(long)]
    pub rebuild: bool,

    /// Walk and report what would be indexed without writing anything.
    #[arg(long)]
    pub dry_run: bool,

    /// Limit to one configured root.
    #[arg(long = "root", value_name = "DIR")]
    pub root: Option<String>,
}

#[derive(Debug, Subcommand)]
pub enum ModelAction {
    /// List known models with their sizes and costs.
    List,
    /// Switch models. Invalidates every stored vector.
    Set { id: String },
    /// Check the loaded model reproduces expected pooling and prefix behaviour.
    SelfTest,
}

#[derive(Debug, Args)]
pub struct BenchArgs {
    /// Models to compare. Defaults to the configured model.
    #[arg(long = "model", short = 'm')]
    pub models: Vec<String>,

    /// Documents to embed when measuring throughput.
    #[arg(long, default_value_t = 512)]
    pub docs: usize,

    /// Run the golden retrieval set and report recall.
    #[arg(long)]
    pub recall: bool,
}

impl Cli {
    /// Parse `std::env::args`, inserting an implicit `search` subcommand when the
    /// first argument is neither a subcommand nor a global flag. Without this,
    /// `wom employment documents` would fail as an unknown subcommand.
    pub fn parse_with_implicit_search() -> Self {
        let argv: Vec<String> = std::env::args().collect();
        Cli::parse_from(insert_implicit_search(argv))
    }
}

const SUBCOMMANDS: &[&str] = &[
    "search", "init", "index", "status", "config", "model", "bench", "help",
];

/// Flags that belong to `wom` itself rather than to a search.
const GLOBAL_FLAGS: &[&str] = &["-h", "--help", "-V", "--version"];

fn insert_implicit_search(mut argv: Vec<String>) -> Vec<String> {
    let first = match argv.get(1) {
        Some(a) => a.as_str(),
        // Bare `wom` should show help, not an empty search.
        None => return argv,
    };
    // Anything that is not a subcommand or a global flag starts a search — and
    // that includes search flags like `--lexical`, which must reach `search`
    // rather than be rejected at the top level.
    let is_explicit = SUBCOMMANDS.contains(&first) || GLOBAL_FLAGS.contains(&first);
    if !is_explicit {
        argv.insert(1, "search".to_string());
    }
    argv
}

/// A parsed search request: where to look, and what to look for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    pub scope: Vec<PathBuf>,
    pub text: String,
}

impl SearchArgs {
    pub fn to_query(&self) -> Result<Query> {
        let (mut scope, text) = split_scope_and_query(&self.words)?;
        for s in &self.scope {
            scope.push(expand_path(s)?);
        }
        Ok(Query { scope, text })
    }
}

/// Split leading directory arguments from the query words, per the spec's
/// `wom ~/projects/ sensors monitor`.
///
/// A leading argument is treated as a scope only when it both exists on disk
/// *and* looks like a path — it contains a separator, or starts with `~`, `.`
/// or `/`. Requiring a path-ish shape is what stops `wom bosnia` from being
/// swallowed as a directory just because `./bosnia` happens to exist.
pub fn split_scope_and_query(words: &[String]) -> Result<(Vec<PathBuf>, String)> {
    let mut scope = Vec::new();
    let mut rest = words;
    while let Some(first) = rest.first() {
        if !looks_like_path(first) {
            break;
        }
        let expanded = expand_path(first)?;
        if !expanded.is_dir() {
            break;
        }
        scope.push(expanded);
        rest = &rest[1..];
    }
    Ok((scope, rest.join(" ")))
}

fn looks_like_path(s: &str) -> bool {
    s.contains('/') || s == "~" || s == "." || s == ".."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implicit_search_is_inserted_for_bare_words() {
        let got = insert_implicit_search(vec!["wom".into(), "employment".into(), "docs".into()]);
        assert_eq!(got, vec!["wom", "search", "employment", "docs"]);
    }

    #[test]
    fn explicit_subcommands_are_left_alone() {
        for sub in ["init", "index", "status", "model", "bench"] {
            let got = insert_implicit_search(vec!["wom".into(), sub.into()]);
            assert_eq!(got, vec!["wom", sub], "subcommand {sub} was rewritten");
        }
    }

    #[test]
    fn global_flags_are_left_alone() {
        for f in ["--version", "-V", "--help", "-h"] {
            assert_eq!(
                insert_implicit_search(vec!["wom".into(), f.into()]),
                vec!["wom", f]
            );
        }
        assert_eq!(insert_implicit_search(vec!["wom".into()]), vec!["wom"]);
    }

    #[test]
    fn search_flags_reach_the_search_subcommand() {
        // Regression: `--lexical` was rejected at the top level because any
        // `--`-prefixed first argument suppressed the implicit `search`.
        let cli = Cli::parse_from(insert_implicit_search(vec![
            "wom".into(),
            "--lexical".into(),
            "bosnia".into(),
        ]));
        match cli.command {
            Command::Search(a) => {
                assert!(a.lexical);
                assert_eq!(a.to_query().unwrap().text, "bosnia");
            }
            other => panic!("expected Search, got {other:?}"),
        }
    }

    #[test]
    fn flags_are_accepted_after_the_query_words() {
        // `wom employment documents --scores` must not treat `--scores` as a word.
        let cli = Cli::parse_from(insert_implicit_search(vec![
            "wom".into(),
            "employment".into(),
            "documents".into(),
            "--scores".into(),
            "-n".into(),
            "5".into(),
        ]));
        match cli.command {
            Command::Search(a) => {
                assert!(a.scores);
                assert_eq!(a.limit, Some(5));
                assert_eq!(a.to_query().unwrap().text, "employment documents");
            }
            other => panic!("expected Search, got {other:?}"),
        }
    }

    #[test]
    fn bare_words_parse_as_a_search() {
        let cli = Cli::parse_from(insert_implicit_search(vec![
            "wom".into(),
            "employment".into(),
            "documents".into(),
        ]));
        match cli.command {
            Command::Search(a) => {
                assert_eq!(a.to_query().unwrap().text, "employment documents")
            }
            other => panic!("expected Search, got {other:?}"),
        }
    }

    #[test]
    fn leading_existing_directory_becomes_scope() {
        let tmp = std::env::temp_dir();
        let words = vec![
            format!("{}/", tmp.display()),
            "sensors".into(),
            "monitor".into(),
        ];
        let (scope, text) = split_scope_and_query(&words).unwrap();
        assert_eq!(scope.len(), 1, "temp dir should have been taken as scope");
        assert_eq!(text, "sensors monitor");
    }

    #[test]
    fn a_bare_word_is_never_a_scope_even_if_a_matching_dir_exists() {
        // The regression this guards: `wom bosnia` run from a directory that
        // contains ./bosnia must still search for "bosnia".
        let dir = std::env::temp_dir().join("wom-test-scope-guard");
        std::fs::create_dir_all(&dir).unwrap();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(std::env::temp_dir()).unwrap();

        let words = vec!["wom-test-scope-guard".to_string()];
        let (scope, text) = split_scope_and_query(&words).unwrap();

        std::env::set_current_dir(prev).unwrap();
        std::fs::remove_dir_all(&dir).ok();

        assert!(scope.is_empty(), "bare word must not be treated as a scope");
        assert_eq!(text, "wom-test-scope-guard");
    }

    #[test]
    fn path_shaped_but_nonexistent_stays_in_the_query() {
        let words = vec!["a/b/definitely-not-here".to_string(), "notes".into()];
        let (scope, text) = split_scope_and_query(&words).unwrap();
        assert!(scope.is_empty());
        assert_eq!(text, "a/b/definitely-not-here notes");
    }

    #[test]
    fn query_with_no_words_is_empty_not_an_error() {
        let (scope, text) = split_scope_and_query(&[]).unwrap();
        assert!(scope.is_empty());
        assert_eq!(text, "");
    }
}
