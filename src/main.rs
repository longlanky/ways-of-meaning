//! `wom` - Ways of Meaning: semantic search over your files.

mod actions;
mod bench;
mod cli;
mod config;
mod db;
mod embed;
mod index;
mod rerank;
mod search;
mod tui;
mod vectors;

use anyhow::{Context, Result, bail};
use cli::{Cli, Command, InitArgs, ModelAction};
use config::{Config, Paths, Profile, Root, expand_path};
use db::Db;
use rerank::Reranker;
use std::io::Write;
use std::path::{Path, PathBuf};
use vectors::VectorStore;

fn main() {
    if let Err(e) = run() {
        eprintln!("wom: {e}");
        for cause in e.chain().skip(1) {
            eprintln!("  caused by: {cause}");
        }
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse_with_implicit_search();
    let paths = Paths::resolve()?;

    match cli.command {
        Command::Init(args) => cmd_init(&paths, args),
        Command::Status(args) => cmd_status(&paths, args),
        Command::Config => cmd_config(&paths),
        Command::Model { action } => cmd_model(&paths, action),
        Command::Index(args) => cmd_index(&paths, args),
        Command::Search(args) => cmd_search(&paths, args),
        Command::Bench(args) => cmd_bench(&paths, args),
    }
}

// ---------------------------------------------------------------- init

/// Directories proposed when the user does not name any: the XDG user dirs plus
/// a couple of conventional code locations. Cache-like and system locations are
/// never proposed.
fn default_candidate_roots() -> Vec<PathBuf> {
    let home = match std::env::var_os("HOME") {
        Some(h) => PathBuf::from(h),
        None => return Vec::new(),
    };
    let mut out = Vec::new();

    // Prefer the user's real XDG configuration over hardcoded names, since
    // these are renamed on non-English systems.
    if let Some(ud) = directories::UserDirs::new() {
        for d in [
            ud.document_dir(),
            ud.desktop_dir(),
            ud.download_dir(),
            ud.picture_dir(),
            ud.video_dir(),
            ud.audio_dir(),
            ud.public_dir(),
        ] {
            if let Some(d) = d {
                out.push(d.to_path_buf());
            }
        }
    }
    for name in ["projects", "Projects", "src", "code", "Notes", "notes"] {
        out.push(home.join(name));
    }

    out.retain(|p| p.is_dir());
    index::walk::dedupe_nested(&mut out);
    out
}

fn cmd_init(paths: &Paths, args: InitArgs) -> Result<()> {
    let cfg_path = paths.config_file();
    if cfg_path.exists() && !args.force {
        bail!(
            "config already exists at {}. Pass --force to overwrite it, \
             or edit it directly.",
            cfg_path.display()
        );
    }

    let mut cfg = Config::default();

    let candidates: Vec<PathBuf> = if args.add.is_empty() {
        default_candidate_roots()
    } else {
        let mut v = Vec::new();
        for a in &args.add {
            let p = expand_path(a)?;
            if !p.is_dir() {
                bail!("{} is not a directory", p.display());
            }
            v.push(p);
        }
        index::walk::dedupe_nested(&mut v);
        v
    };
    if candidates.is_empty() {
        bail!("no directories to index. Name some with `wom init --add ~/Documents`.");
    }

    let forced = match &args.profile {
        Some(s) => {
            Some(Profile::from_str_opt(s).with_context(|| format!("unknown profile `{s}`"))?)
        }
        None => None,
    };

    // Probing costs a capped walk per root, which is seconds even on a large
    // tree, and it is what lets us show the real cost before committing.
    eprintln!("Examining directories...\n");

    let spec = embed::lookup(&cfg.model)?;
    let mut rows = Vec::new();
    let mut total_files = 0usize;
    let mut truncated_any = false;

    for path in &candidates {
        let probe = index::walk::probe(path, &cfg)?;
        if probe.files == 0 {
            continue;
        }
        let profile = forced.unwrap_or_else(|| probe.suggest());
        total_files += probe.files;
        truncated_any |= probe.truncated;
        rows.push((path.clone(), probe, profile));
    }

    if rows.is_empty() {
        bail!("every candidate directory came back empty after ignore rules");
    }

    println!("{:<44} {:>10}  {}", "DIRECTORY", "FILES", "PROFILE");
    for (path, probe, profile) in &rows {
        let count = if probe.truncated {
            format!("{}+", probe.files)
        } else {
            probe.files.to_string()
        };
        println!(
            "{:<44} {:>10}  {}",
            elide(&path.to_string_lossy(), 44),
            count,
            profile.as_str()
        );
    }

    // Prefer a rate actually measured on this machine, which `wom bench` and every
    // completed scan record. The registry figure is only a starting guess.
    let measured = Db::open(&paths.db_file())
        .ok()
        .and_then(|db| db.get_meta(db::META_INDEX_RATE).ok().flatten())
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|r| *r > 0.0);
    let rate = measured.unwrap_or(spec.rough_docs_per_sec.max(1) as f64);

    let est_secs = total_files as f64 / rate;
    println!(
        "\n{}{} files, model {} ({} MB download).",
        total_files,
        if truncated_any { "+" } else { "" },
        spec.id,
        spec.approx_mb
    );
    println!(
        "First index will take roughly {} at {:.0} docs/sec ({}).\n\
         Later runs only process what changed.",
        human_duration(est_secs),
        rate,
        match measured {
            Some(_) => "measured on this machine",
            None => "rough estimate; `wom bench` measures it properly",
        }
    );
    if truncated_any {
        println!(
            "Note: at least one directory hit the {} file probe cap, so the\n\
             real totals are higher than shown.",
            index::walk::PROBE_CAP
        );
    }

    if !args.yes && !confirm("\nWrite this config?")? {
        println!("Nothing written.");
        return Ok(());
    }

    cfg.roots = rows
        .into_iter()
        .map(|(path, _, profile)| Root { path, profile })
        .collect();
    cfg.save(paths)?;
    paths.ensure_data_dir()?;

    let db = Db::open(&paths.db_file())?;
    db.set_meta("model_id", &cfg.model)?;
    db.set_meta("dim", &spec.dim.to_string())?;

    println!("\nWrote {}", paths.config_file().display());
    println!("Created {}", paths.db_file().display());
    println!("\nNext: `wom index` to build the index.");
    Ok(())
}

// ---------------------------------------------------------------- search

fn cmd_search(paths: &Paths, args: cli::SearchArgs) -> Result<()> {
    let cfg = Config::load(paths)?;
    let query = args.to_query()?;
    if query.text.trim().is_empty() {
        bail!("nothing to search for. Try `wom employment documents`.");
    }

    let db_path = paths.db_file();
    if !db_path.exists() {
        bail!("no index yet. Run `wom init` then `wom index`.");
    }
    let db = Db::open(&db_path)?;

    // Spec item 3: refresh on the configured schedule. The scan runs detached so
    // this query answers from the index that already exists — waiting minutes for
    // a rescan before showing results would defeat the point of the tool.
    maybe_refresh(paths, &cfg, &db);

    let mode = if args.lexical {
        search::Mode::Lexical
    } else if args.dense {
        search::Mode::Dense
    } else {
        search::Mode::Hybrid
    };

    // Lexical-only search must not pay for loading a model.
    let embedder = match mode {
        search::Mode::Lexical => None,
        _ => Some(embed::onnx::OnnxEmbedder::load_on(paths, &cfg.model, cfg.device)?),
    };

    // An interrupted scan leaves a good lexical index over an empty vector
    // store, and hybrid search over it silently degrades to exact matching.
    // Say so once, here, rather than inside the per-keystroke TUI refresh.
    if let Some(e) = embedder.as_ref().map(|e| e as &dyn embed::Embedder) {
        if let Some(gap) = search::dense_gap(paths, e) {
            eprintln!("wom: {gap}; results are full-text only. Run `wom index --now`.");
        }
    }

    // A reranker is loaded once per invocation, next to the embedder. Lexical
    // mode is the no-model fast path and skips it there too.
    let reranker = match (mode, effective_reranker(&cfg, &args)) {
        (search::Mode::Lexical, _) => None,
        (_, Some(id)) => Some(rerank::OnnxReranker::load(paths, &id)?),
        (_, None) => None,
    };

    let req = search::Request {
        text: query.text,
        scope: query.scope,
        limit: args.limit.unwrap_or(cfg.limit),
        mode,
        min_similarity: args
            .min_similarity
            .unwrap_or_else(|| cfg.resolved_min_similarity()),
    };

    let results = search::search(
        paths,
        &db,
        embedder.as_ref().map(|e| e as &dyn embed::Embedder),
        reranker.as_ref().map(|r| r as &dyn rerank::Reranker),
        &req,
    )?;

    if results.is_empty() {
        eprintln!("no matches");
        return Ok(());
    }

    // The TUI only makes sense when a human is watching an interactive terminal.
    // Piped or redirected output falls through to plain lines below, so
    // `wom foo | head` and `$(wom foo)` behave like any other Unix tool.
    use std::io::IsTerminal;
    let interactive = std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
    if interactive && !args.no_tui {
        let mut app = tui::App::new(
            paths,
            &db,
            embedder.as_ref().map(|e| e as &dyn embed::Embedder),
            reranker.as_ref().map(|r| r as &dyn rerank::Reranker),
            &req,
            results,
        );
        return tui::run(&mut app);
    }

    // Plain lines when piped, matching the spec's examples, so `wom foo | head`
    // and shell substitution both work.
    for r in &results {
        if args.scores {
            // Show the cosine and rerank logit, which are interpretable,
            // alongside the fused rank score, which is not.
            let cos = match r.cosine {
                Some(c) => format!("cos={c:.3}"),
                None => "cos=-    ".to_string(),
            };
            let rr = match r.rerank {
                Some(s) => format!(" rr={s:+.2}"),
                None => String::new(),
            };
            println!("{:.4}  {cos}{rr}\t{}", r.score, r.display());
        } else {
            println!("{}", r.display());
        }
    }
    Ok(())
}

/// Which reranker to load, if any. CLI beats config; `--rerank` with the
/// config at "off" turns the default model on; "off" means none.
fn effective_reranker(cfg: &Config, args: &cli::SearchArgs) -> Option<String> {
    if args.no_rerank {
        return None;
    }
    if let Some(m) = &args.rerank_model {
        return Some(m.clone());
    }
    match cfg.reranker.as_str() {
        "off" => args.rerank.then(|| rerank::DEFAULT_RERANKER.to_string()),
        id => Some(id.to_string()),
    }
}

/// Kick off a scheduled refresh in the background if one is due.
///
/// Failures here are reported but never fatal: a query should still answer from
/// the existing index even if the refresh cannot be started.
fn maybe_refresh(paths: &Paths, cfg: &Config, db: &Db) {
    let last = db.get_meta_i64("last_scan_at").ok().flatten();
    match index::is_due(cfg.refresh, last, index::now_secs()) {
        index::Due::No => {}
        index::Due::Never => {
            // Nothing indexed yet, so there is no point starting a silent
            // background scan the user will not see the results of.
            eprintln!("wom: index is empty; run `wom index` to build it");
        }
        index::Due::Elapsed => {
            if index::scan_in_progress(&paths.scan_lock()) {
                eprintln!("wom: a refresh is already running; showing current results");
                return;
            }
            match index::spawn_background_scan(paths) {
                Ok(()) => eprintln!("wom: refreshing the index in the background"),
                Err(e) => eprintln!("wom: could not start background refresh: {e}"),
            }
        }
    }
}

// ---------------------------------------------------------------- index

fn cmd_index(paths: &Paths, args: cli::IndexArgs) -> Result<()> {
    let cfg = Config::load(paths)?;
    if cfg.roots.is_empty() {
        bail!("no directories configured. Run `wom init` first.");
    }

    if args.rebuild {
        discard_index(paths)?;
    }

    // Without --now, respect the schedule, so `wom index` can be put on a cron or
    // a shell hook without re-scanning on every invocation.
    if !args.now && !args.force && !args.rebuild && !args.dry_run {
        let db = Db::open(&paths.db_file())?;
        let last = db.get_meta_i64("last_scan_at")?;
        if index::is_due(cfg.refresh, last, index::now_secs()) == index::Due::No {
            let ago = last.map(|t| human_duration(age_secs(t) as f64));
            println!(
                "Index is up to date{} per the `{:?}` schedule. Pass --now to scan anyway.",
                ago.map(|a| format!(" (scanned {a} ago)")).unwrap_or_default(),
                cfg.refresh
            );
            return Ok(());
        }
    }

    let only_root = match &args.root {
        Some(r) => {
            let p = expand_path(r)?;
            if !cfg.roots.iter().any(|x| x.path == p) {
                bail!(
                    "{} is not a configured root. `wom status` lists them.",
                    p.display()
                );
            }
            Some(p)
        }
        None => None,
    };

    // Loading the model costs a download on first use and a second or two after,
    // so skip it entirely when nothing will be embedded.
    let embedder = if args.dry_run {
        None
    } else {
        Some(embed::onnx::OnnxEmbedder::load_on(paths, &cfg.model, cfg.device)?)
    };

    let opts = index::ScanOptions {
        force: args.force,
        dry_run: args.dry_run,
        only_root,
        embed: !args.dry_run,
        progress: true,
    };

    let stats = index::scan(
        paths,
        &cfg,
        embedder.as_ref().map(|e| e as &dyn embed::Embedder),
        &opts,
    )?;

    if args.dry_run {
        println!(
            "dry run: {} files would be indexed ({} new, {} changed, {} unchanged)",
            stats.seen, stats.added, stats.updated, stats.unchanged
        );
        return Ok(());
    }

    println!(
        "indexed {} files in {} ({} new, {} changed, {} unchanged, {} embedded)",
        stats.seen,
        human_duration(stats.elapsed_secs),
        stats.added,
        stats.updated,
        stats.unchanged,
        stats.embedded
    );
    println!("{} directories", stats.dirs);
    if stats.skipped > 0 {
        // Unreadable paths are normal but should not be invisible.
        println!(
            "{} paths skipped as unreadable (permissions, or removed mid-scan)",
            stats.skipped
        );
    }
    if stats.deleted_files > 0 || stats.deleted_dirs > 0 {
        println!(
            "removed {} files and {} directories that are gone from disk",
            stats.deleted_files, stats.deleted_dirs
        );
    }
    if let Some(rate) = stats.embed_rate() {
        println!("embedding rate {rate:.0} docs/sec");
    }
    Ok(())
}

/// Delete the derived index so the next scan starts clean. Only ever removes
/// files `wom` created itself.
fn discard_index(paths: &Paths) -> Result<()> {
    for p in paths.derived_files() {
        match std::fs::remove_file(&p) {
            Ok(()) => eprintln!("removed {}", p.display()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(anyhow::Error::from(e).context(format!("removing {}", p.display())));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- status

fn cmd_status(paths: &Paths, args: cli::StatusArgs) -> Result<()> {
    let cfg = Config::load(paths)?;
    println!("config   {}", paths.config_file().display());
    println!("data     {}", paths.data_dir().display());

    let db_path = paths.db_file();
    if !db_path.exists() {
        println!("\nNo index yet. Run `wom init` then `wom index`.");
        return Ok(());
    }
    let db = Db::open(&db_path)?;

    let model = db.get_meta("model_id")?.unwrap_or_else(|| "-".into());
    let dim = db.get_meta("dim")?.and_then(|d| d.parse::<usize>().ok());
    match dim {
        Some(d) => println!("model    {model} ({d}-dim)"),
        None => println!("model    {model}"),
    }
    if model != cfg.model && model != "-" {
        println!(
            "         (config says {}; run `wom index --rebuild` to switch)",
            cfg.model
        );
    }
    println!("refresh  {:?}", cfg.refresh);
    print_vectors_line(paths, &model, dim);

    let files = db.count_files()?;
    let embedded = db.count_embedded()?;
    let dirs = db.count_dirs()?;
    println!("\nfiles    {files} indexed, {embedded} embedded");
    println!("dirs     {dirs}");

    let coverage = db.extract_coverage()?;
    if !coverage.is_empty() {
        let parts: Vec<String> = coverage
            .iter()
            .map(|(kind, n)| format!("{n} {kind}"))
            .collect();
        println!("text     {}", parts.join(", "));
    }

    match db.get_meta_i64("last_scan_at")? {
        Some(ts) => println!("scanned  {} ago", human_duration(age_secs(ts) as f64)),
        None => println!("scanned  never"),
    }

    if args.verify {
        // The full-text index and the file table are maintained by separate
        // statements, so a crash between them could in principle leave them out
        // of step. Cheap to check, and it is the one invariant that would show up
        // as "that file exists but never matches".
        print!("\nverify   full-text index ... ");
        if db.fts_is_consistent()? {
            println!("consistent");
        } else {
            println!("INCONSISTENT - run `wom index --rebuild`");
        }
    }

    println!("\nroots");
    if cfg.roots.is_empty() {
        println!("  (none configured)");
    }
    for r in &cfg.roots {
        let missing = if r.path.is_dir() { "" } else { "  [MISSING]" };
        println!(
            "  {:<10} {}{}",
            r.profile.as_str(),
            r.path.display(),
            missing
        );
    }
    Ok(())
}

/// The `vectors  ...` status line: how many rows each store holds, with the
/// remedy spelled out for the two gap cases. Never creates the files —
/// `VectorStore::open` would, so existence is checked first.
fn print_vectors_line(paths: &Paths, model: &str, dim: Option<usize>) {
    match dim {
        Some(d) if model != "-" => {
            let files = vector_count(&paths.file_vectors(), d, model);
            let dirs = vector_count(&paths.dir_vectors(), d, model);
            println!("vectors  {files} file, {dirs} dir");
            match files.as_str() {
                "none" => println!("         (no vector index; run `wom index` to build it)"),
                "0" => println!(
                    "         (empty - indexing was likely interrupted; run `wom index --now`)"
                ),
                _ => {}
            }
        }
        _ => println!("vectors  -"),
    }
}

fn vector_count(path: &Path, dim: usize, model: &str) -> String {
    if !path.exists() {
        return "none".into();
    }
    match VectorStore::open(path, dim, model) {
        Ok(s) => s.high_water().to_string(),
        // A dim/model mismatch is already reported on the model line above.
        Err(_) => "incompatible".into(),
    }
}

fn cmd_config(paths: &Paths) -> Result<()> {
    let cfg = Config::load(paths)?;
    let p = paths.config_file();
    if p.exists() {
        println!("# {}", p.display());
    } else {
        println!("# {} (does not exist yet; showing defaults)", p.display());
    }
    print!("{}", toml::to_string_pretty(&cfg)?);
    Ok(())
}

// ---------------------------------------------------------------- model

fn cmd_model(paths: &Paths, action: ModelAction) -> Result<()> {
    match action {
        ModelAction::List => {
            let cfg = Config::load(paths)?;
            // The model the current index was actually built with, if there is
            // one — distinct from the configured model after `wom model set`
            // and before the rebuild that makes them agree again.
            let indexed = match paths.db_file().exists() {
                true => Db::open(&paths.db_file())
                    .and_then(|db| db.get_meta("model_id"))
                    .ok()
                    .flatten(),
                false => None,
            };
            println!(
                "{:<26} {:>5} {:>7} {:>9}  {}",
                "MODEL", "DIM", "SIZE", "POOLING", "NOTE"
            );
            for m in embed::REGISTRY {
                let marker = if m.id == cfg.model {
                    "*"
                } else if indexed.as_deref() == Some(m.id) {
                    "i"
                } else {
                    " "
                };
                println!(
                    "{marker}{:<25} {:>5} {:>6}M {:>9}  {}",
                    m.id,
                    m.dim,
                    m.approx_mb,
                    format!("{:?}", m.pooling).to_lowercase(),
                    m.note
                );
            }
            println!("\n* = configured, i = built the current index.");
            println!("Switching models requires `wom index --rebuild`.");

            println!("\n{:<42} {:>7}  {}", "RERANKER", "SIZE", "NOTE");
            for m in rerank::RERANKER_REGISTRY {
                let marker = if m.id == cfg.reranker { "*" } else { " " };
                println!("{marker}{:<41} {:>6}M  {}", m.id, m.approx_mb, m.note);
            }
            println!("\n* = configured (\"off\" in config means none). Enable per query with --rerank.");
            Ok(())
        }
        ModelAction::Set { id } => {
            let spec = embed::lookup(&id)?;
            let mut cfg = Config::load(paths)?;
            if cfg.model == spec.id {
                println!("Already using {}.", spec.id);
                return Ok(());
            }
            let old = std::mem::replace(&mut cfg.model, spec.id.to_string());
            cfg.save(paths)?;
            println!("Model: {old} -> {}", spec.id);
            println!(
                "Vectors from the old model are not comparable.\n\
                 Run `wom index --rebuild` to re-embed everything."
            );
            Ok(())
        }
        ModelAction::SelfTest => {
            let cfg = Config::load(paths)?;
            paths.ensure_data_dir()?;
            embed::onnx::self_test(paths, &cfg.model, cfg.device)
        }
    }
}

// ---------------------------------------------------------------- bench

fn cmd_bench(paths: &Paths, args: cli::BenchArgs) -> Result<()> {
    let cfg = Config::load(paths)?;
    paths.ensure_data_dir()?;

    let models: Vec<String> = if args.models.is_empty() {
        vec![cfg.model.clone()]
    } else {
        args.models.clone()
    };
    // Fail on a bad name before spending minutes downloading the good ones.
    for m in &models {
        embed::lookup(m)?;
    }

    // Project against the real corpus when there is one, so the numbers answer
    // "how long would my index take" rather than an abstraction.
    let corpus = Db::open(&paths.db_file())
        .ok()
        .and_then(|db| db.count_files().ok())
        .filter(|n| *n > 0)
        .unwrap_or(0) as usize;

    println!("Embedding throughput ({} sample docs each)\n", args.docs);
    println!(
        "{:<26} {:>5} {:>7} {:>10} {:>9} {:>8} {:>8}",
        "MODEL", "DIM", "LOAD", "DOCS/SEC", "QUERY", "MARGIN", "COSINES"
    );

    let mut reports = Vec::new();
    for id in &models {
        let r = bench::measure_model(paths, id, args.docs, cfg.device)?;
        println!(
            "{:<26} {:>5} {:>6.1}s {:>10.0} {:>7.1}ms {:>8.3} {:>8}",
            r.model_id,
            r.dim,
            r.load_secs,
            r.docs_per_sec,
            r.query_ms,
            r.margin(),
            format!("{:.2}/{:.2}", r.paraphrase, r.unrelated)
        );
        if r.on_gpu {
            println!("  (running on GPU via WebGPU/Dawn)");
        }
        reports.push(r);
    }

    if corpus > 0 {
        // DOCS/SEC above is inference alone. A real scan also extracts text and
        // writes to SQLite, which on this corpus roughly halves it — so project
        // from a measured end-to-end rate when one exists, and say which is used.
        let measured_end_to_end = Db::open(&paths.db_file())
            .ok()
            .and_then(|db| db.get_meta(db::META_INDEX_RATE).ok().flatten())
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|r| *r > 0.0);

        println!("\nProjected full index of {corpus} files:");
        for r in &reports {
            match measured_end_to_end.filter(|_| r.model_id == cfg.model) {
                Some(rate) => println!(
                    "  {:<26} {} (at the {rate:.0} docs/sec this machine actually sustained)",
                    r.model_id,
                    human_duration(corpus as f64 / rate)
                ),
                None => println!(
                    "  {:<26} {} (inference only; extraction and writes will add to this)",
                    r.model_id,
                    human_duration(r.hours_for(corpus) * 3600.0)
                ),
            }
        }
    }
    println!(
        "\nMARGIN is paraphrase-minus-unrelated cosine: higher means relevance is\n\
         easier to separate from noise. COSINES shows the pair it came from."
    );

    // What `--rerank` costs on this machine: one load plus one committed-query
    // pool of 80 pairs (the pool a default-limit search reranks).
    if cfg.reranker != "off" {
        let rep = bench::measure_reranker(paths, &cfg.reranker, 80)?;
        println!(
            "\nReranker {}: load {:.1}s, {} pairs in {:.0}ms ({:.1}ms/pair)",
            rep.model_id,
            rep.load_secs,
            rep.pairs,
            rep.total_ms,
            rep.total_ms / rep.pairs as f64
        );
    }

    if let (Some(r), Ok(db)) = (reports.first(), Db::open(&paths.db_file())) {
        if r.model_id == cfg.model {
            db.set_meta(db::META_MODEL_RATE, &format!("{:.1}", r.docs_per_sec))
                .ok();
        }
    }

    if args.recall {
        println!("\nRetrieval quality against the live index");
        let db_path = paths.db_file();
        if !db_path.exists() {
            println!("  no index yet; run `wom index` first");
            return Ok(());
        }
        let db = Db::open(&db_path)?;
        let emb = embed::onnx::OnnxEmbedder::load_on(paths, &cfg.model, cfg.device)?;
        // Recall is measured the way searches actually run, reranker included.
        let rr = match cfg.reranker.as_str() {
            "off" => None,
            id => Some(rerank::OnnxReranker::load(paths, id)?),
        };
        if let Some(rr) = &rr {
            println!("  (with {} reranking)", rr.model_id());
        }
        let rep = bench::measure_recall(
            paths,
            &cfg,
            &db,
            &emb,
            rr.as_ref().map(|r| r as &dyn rerank::Reranker),
            cfg.limit,
        )?;
        println!(
            "  {}/{} golden queries found their expected answer",
            rep.hits, rep.total
        );
        println!("  latency p50 {:.0}ms, p95 {:.0}ms", rep.p50_ms, rep.p95_ms);
        for (query, top) in &rep.misses {
            println!("  MISS {query:?}");
            for t in top {
                println!("       got {t}");
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- helpers

fn confirm(prompt: &str) -> Result<bool> {
    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() {
        // Non-interactive: refuse rather than guess. `--yes` is the way to
        // proceed unattended.
        bail!("not a terminal, so cannot prompt; pass --yes to accept");
    }
    print!("{prompt} [y/N] ");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes" | "Yes"))
}

/// Seconds since `ts`, clamped at zero so a clock that moved backwards reads as
/// "just now" rather than as a huge negative age.
fn age_secs(ts: i64) -> i64 {
    (index::now_secs() - ts).max(0)
}

fn human_duration(secs: f64) -> String {
    if secs < 1.0 {
        "less than a second".to_string()
    } else if secs < 90.0 {
        format!("{secs:.0} seconds")
    } else if secs < 90.0 * 60.0 {
        format!("{:.0} minutes", secs / 60.0)
    } else if secs < 48.0 * 3600.0 {
        format!("{:.1} hours", secs / 3600.0)
    } else {
        format!("{:.0} days", secs / 86400.0)
    }
}

/// Shorten a long path from the left, keeping the informative tail.
fn elide(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    let tail: String = s.chars().skip(n - (max - 3)).collect();
    format!("...{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_duration_picks_sensible_units() {
        assert_eq!(human_duration(0.2), "less than a second");
        assert_eq!(human_duration(45.0), "45 seconds");
        assert_eq!(human_duration(600.0), "10 minutes");
        assert_eq!(human_duration(7200.0), "2.0 hours");
    }

    #[test]
    fn elide_keeps_the_tail_and_respects_the_limit() {
        let out = elide("/home/nolan/projects/very/deep/path/file.txt", 20);
        assert_eq!(out.chars().count(), 20);
        assert!(out.starts_with("..."));
        assert!(out.ends_with("file.txt"));
        assert_eq!(elide("/short", 20), "/short");
    }

    #[test]
    fn elide_handles_multibyte_without_panicking() {
        // Char-based slicing, not byte-based: this would panic on a byte split.
        let out = elide("/home/nolan/Документы/файл-с-длинным-именем.txt", 20);
        assert_eq!(out.chars().count(), 20);
    }

    #[test]
    fn age_is_never_negative_for_a_future_timestamp() {
        assert_eq!(age_secs(i64::MAX / 2), 0);
    }
}
