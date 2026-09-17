//! What the keys in the TUI actually do: open things, and save results.

use crate::search::SearchResult;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The directory a result refers to: itself if it is a directory, otherwise its
/// parent.
pub fn dir_of(r: &SearchResult) -> PathBuf {
    match r.kind {
        crate::search::ResultKind::Dir => r.path.clone(),
        crate::search::ResultKind::File => r
            .path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("/")),
    }
}

/// True when a desktop session is present, so `xdg-open` has somewhere to open to.
pub fn has_desktop() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some() || std::env::var_os("DISPLAY").is_some()
}

/// Spawn a terminal emulator with its working directory set to `dir`.
///
/// `$TERMINAL` is honoured first, then a list of common emulators. Each takes its
/// own flag for "run this command", so the invocation cannot be uniform.
pub fn open_terminal(dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        bail!("{} is not a directory", dir.display());
    }

    if let Ok(term) = std::env::var("TERMINAL") {
        if !term.trim().is_empty() {
            return spawn_detached(
                Command::new(&term).current_dir(dir),
                &format!("terminal {term}"),
            );
        }
    }

    // (binary, args needed to start in a directory). Ordered by how likely the
    // user is to consider it "their" terminal.
    let candidates: &[(&str, &[&str])] = &[
        ("foot", &[]),
        ("alacritty", &["--working-directory", "."]),
        ("kitty", &[]),
        ("wezterm", &["start", "--cwd", "."]),
        ("ghostty", &[]),
        ("gnome-terminal", &[]),
        ("konsole", &["--workdir", "."]),
        ("xfce4-terminal", &[]),
        ("terminator", &[]),
        ("urxvt", &[]),
        ("xterm", &[]),
    ];

    for (bin, args) in candidates {
        if !which(bin) {
            continue;
        }
        let mut cmd = Command::new(bin);
        cmd.current_dir(dir);
        // "." works because the child's cwd is already `dir`.
        cmd.args(args.iter());
        return spawn_detached(&mut cmd, &format!("terminal {bin}"));
    }

    bail!(
        "no terminal emulator found. Set $TERMINAL to the one you use, \
         e.g. TERMINAL=foot. (Tried: {})",
        candidates
            .iter()
            .map(|(b, _)| *b)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// The editor command for `path`, for the caller to run *in the current terminal*.
///
/// Returns a command rather than running it because the TUI has to leave the
/// alternate screen and restore the terminal first; an editor spawned underneath a
/// live ratatui session would fight it for the tty.
pub fn editor_command(path: &Path) -> Command {
    let visual = std::env::var("VISUAL").ok();
    let editor = std::env::var("EDITOR").ok();
    let chosen = resolve_editor(visual.as_deref(), editor.as_deref(), &|c| which(c));
    build_command(&chosen, path)
}

/// Pick an editor command line, preferring `$VISUAL`, then `$EDITOR`, then
/// whichever common editor is installed.
///
/// Takes its inputs as parameters instead of reading the environment directly so
/// it can be tested without mutating process-global state, which races under the
/// parallel test harness.
fn resolve_editor(
    visual: Option<&str>,
    editor: Option<&str>,
    installed: &dyn Fn(&str) -> bool,
) -> String {
    for candidate in [visual, editor].into_iter().flatten() {
        if !candidate.trim().is_empty() {
            return candidate.trim().to_string();
        }
    }
    for c in ["nvim", "vim", "nano", "vi"] {
        if installed(c) {
            return c.to_string();
        }
    }
    "vi".to_string()
}

/// Split a command line that may carry arguments, e.g. `code -w` or `emacs -nw`.
fn build_command(cmdline: &str, path: &Path) -> Command {
    let mut parts = cmdline.split_whitespace();
    let bin = parts.next().unwrap_or("vi");
    let mut cmd = Command::new(bin);
    for a in parts {
        cmd.arg(a);
    }
    cmd.arg(path);
    cmd
}

/// Hand `path` to the desktop's default application.
pub fn xdg_open(path: &Path) -> Result<()> {
    if !has_desktop() {
        bail!("no desktop session ($DISPLAY and $WAYLAND_DISPLAY are both unset)");
    }
    if !which("xdg-open") {
        bail!("xdg-open not found (install xdg-utils)");
    }
    spawn_detached(Command::new("xdg-open").arg(path), "xdg-open")
}

/// Copy `text` to the system clipboard, falling back through the usual tools.
pub fn copy_to_clipboard(text: &str) -> Result<&'static str> {
    let tools: &[(&str, &[&str])] = &[
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
        ("xsel", &["--clipboard", "--input"]),
    ];
    for (bin, args) in tools {
        if !which(bin) {
            continue;
        }
        let mut child = Command::new(bin)
            .args(args.iter())
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("spawning {bin}"))?;
        if let Some(stdin) = child.stdin.as_mut() {
            use std::io::Write;
            stdin
                .write_all(text.as_bytes())
                .with_context(|| format!("writing to {bin}"))?;
        }
        // wl-copy stays resident to serve the selection, so do not wait on it.
        if *bin != "wl-copy" {
            child.wait().ok();
        }
        return Ok(bin);
    }
    bail!("no clipboard tool found (install wl-clipboard, xclip, or xsel)")
}

/// Write the result list to `path`, one per line.
///
/// Sandboxed to a bare filename in the current directory: absolute paths,
/// parent components (`..`, `/`, `~`-leading) are rejected, so the one-keystroke
/// `s` action cannot write outside the drawer you are in. Pipe/redirect remains
/// the escape hatch for saving elsewhere.
///
/// Uses `create_new` (O_CREAT|O_EXCL) so the no-clobber check is atomic: no
/// TOCTOU window, and dangling symlinks fail instead of being followed.
pub fn save_results(results: &[SearchResult], query: &str, path: &Path) -> Result<usize> {
    let s = path.to_string_lossy();
    if s.starts_with('~') {
        bail!("`~` is not expanded here; save to a plain filename or redirect output instead");
    }
    if path.is_absolute() {
        bail!(
            "{} is absolute; `s` saves to the current directory only",
            path.display()
        );
    }
    // Exactly one normal component: rejects `a/b`, `..`, `/`, empty, prefixes.
    {
        use std::path::Component;
        let mut comps = path.components();
        match (comps.next(), comps.next()) {
            (Some(Component::Normal(_)), None) => {}
            _ => bail!(
                "{} is not a plain filename; `s` saves to the current directory only",
                path.display()
            ),
        }
    }
    let mut out = String::new();
    out.push_str(&format!("# wom results for: {query}\n"));
    out.push_str(&format!("# {} matches\n\n", results.len()));
    for r in results {
        out.push_str(&r.display());
        out.push('\n');
    }
    {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    anyhow::anyhow!("{} already exists; pick another name", path.display())
                } else {
                    anyhow::Error::from(e).context(format!("writing {}", path.display()))
                }
            })?;
        f.write_all(out.as_bytes())
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(results.len())
}

/// Start a process fully detached, so it outlives `wom` and does not inherit the
/// terminal.
pub fn spawn_detached(cmd: &mut Command, what: &str) -> Result<()> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("starting {what}"))?;
    Ok(())
}

/// Whether `cmd` is an executable on `$PATH`. The single copy in the crate.
pub fn which(cmd: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|d| {
        let p = d.join(cmd);
        p.is_file() && is_executable(&p)
    })
}

fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p)
        .map(|m| m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::ResultKind;

    fn res(kind: ResultKind, path: &str) -> SearchResult {
        SearchResult {
            kind,
            path: PathBuf::from(path),
            score: 1.0,
            snippet: String::new(),
            dense_rank: None,
            cosine: None,
            lexical_rank: None,
            rerank: None,
            file_count: None,
        }
    }

    #[test]
    fn dir_of_a_file_is_its_parent() {
        let r = res(ResultKind::File, "/a/b/c.txt");
        assert_eq!(dir_of(&r), PathBuf::from("/a/b"));
    }

    #[test]
    fn dir_of_a_directory_is_itself() {
        let r = res(ResultKind::Dir, "/a/b");
        assert_eq!(dir_of(&r), PathBuf::from("/a/b"));
    }

    #[test]
    fn dir_of_a_root_level_file_does_not_panic() {
        let r = res(ResultKind::File, "/c.txt");
        assert_eq!(dir_of(&r), PathBuf::from("/"));
    }

    #[test]
    fn visual_wins_over_editor() {
        let got = resolve_editor(Some("vim"), Some("nano"), &|_| true);
        assert_eq!(got, "vim");
    }

    #[test]
    fn editor_is_used_when_visual_is_unset_or_blank() {
        assert_eq!(resolve_editor(None, Some("nano"), &|_| true), "nano");
        assert_eq!(resolve_editor(Some("  "), Some("nano"), &|_| true), "nano");
    }

    #[test]
    fn falls_back_to_the_first_installed_editor() {
        // Only nano present.
        let got = resolve_editor(None, None, &|c| c == "nano");
        assert_eq!(got, "nano");
        // Nothing present at all still yields something runnable.
        assert_eq!(resolve_editor(None, None, &|_| false), "vi");
    }

    #[test]
    fn editor_arguments_are_split_into_argv() {
        let cmd = build_command("emacs -nw", Path::new("/tmp/x"));
        assert_eq!(cmd.get_program(), "emacs");
        let args: Vec<_> = cmd.get_args().collect();
        assert_eq!(args, vec!["-nw", "/tmp/x"]);
    }

    #[test]
    fn editor_without_arguments_passes_only_the_path() {
        let cmd = build_command("vim", Path::new("/tmp/x"));
        assert_eq!(cmd.get_program(), "vim");
        let args: Vec<_> = cmd.get_args().collect();
        assert_eq!(args, vec!["/tmp/x"]);
    }

    #[test]
    fn save_results_writes_one_line_per_result() {
        // Sandboxed to cwd: use a unique bare filename in the test process's
        // cwd (no set_current_dir — that races under the parallel harness).
        let name = format!("wom-save-test-{}.txt", std::process::id());
        let path = PathBuf::from(&name);
        std::fs::remove_file(&path).ok();

        let results = vec![
            res(ResultKind::Dir, "/h/employment_docs"),
            res(ResultKind::File, "/h/employment_docs/2024_W2.pdf"),
        ];
        let n = save_results(&results, "employment documents", &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(n, 2);
        assert!(text.contains("# wom results for: employment documents"));
        assert!(text.contains("directory:/h/employment_docs"));
        assert!(text.contains("/h/employment_docs/2024_W2.pdf"));
    }

    #[test]
    fn save_results_refuses_to_overwrite() {
        let name = format!("wom-save-clobber-{}.txt", std::process::id());
        let path = PathBuf::from(&name);
        std::fs::write(&path, "precious").unwrap();

        let err = save_results(&[], "q", &path).unwrap_err().to_string();
        let still_there = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(err.contains("already exists"), "got: {err}");
        assert_eq!(still_there, "precious", "existing file was clobbered");
    }

    #[test]
    fn save_results_rejects_paths_outside_the_cwd() {
        for bad in [
            "/tmp/abs.txt",
            "../evil.txt",
            "a/b.txt",
            "..",
            "/",
            "~/out.txt",
            "~",
            "",
        ] {
            let err = save_results(&[], "q", Path::new(bad))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("current directory")
                    || err.contains("plain filename")
                    || err.contains("not expanded"),
                "got for {bad:?}: {err}"
            );
        }
    }

    #[test]
    fn open_terminal_rejects_a_non_directory() {
        assert!(open_terminal(Path::new("/definitely/not/a/dir")).is_err());
    }
}
