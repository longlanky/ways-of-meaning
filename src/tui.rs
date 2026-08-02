//! The interactive result browser.

use crate::actions;
use crate::config::Paths;
use crate::db::Db;
use crate::embed::Embedder;
use crate::rerank::Reranker;
use crate::search::{self, Request, ResultKind, SearchResult};
use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use std::io::Stdout;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// How long a transient status message stays on screen.
const STATUS_TTL: Duration = Duration::from_secs(4);

/// What the event loop decided to do after a key press.
enum Flow {
    Continue,
    Quit,
    /// Leave the alternate screen, run something interactive, then come back.
    Suspend(std::process::Command),
}

pub struct App<'a> {
    paths: &'a Paths,
    db: &'a Db,
    embedder: Option<&'a dyn Embedder>,
    reranker: Option<&'a dyn Reranker>,

    /// The resolved request, owned outright. Rebuilding it from `Config` on every
    /// refresh silently discarded the user's `-n`, `--min-similarity` and
    /// `--dense` the moment they pressed `/`.
    req: Request,
    results: Vec<SearchResult>,
    state: ListState,

    /// Editing the query in place, via `/`.
    editing: bool,
    /// Prompting for a filename, via `s`.
    saving: Option<String>,
    status: Option<(String, Instant, bool)>,
    show_scores: bool,
}

impl<'a> App<'a> {
    pub fn new(
        paths: &'a Paths,
        db: &'a Db,
        embedder: Option<&'a dyn Embedder>,
        reranker: Option<&'a dyn Reranker>,
        req: &Request,
        results: Vec<SearchResult>,
    ) -> Self {
        let mut state = ListState::default();
        if !results.is_empty() {
            state.select(Some(0));
        }
        Self {
            paths,
            db,
            embedder,
            reranker,
            req: req.clone(),
            results,
            state,
            editing: false,
            saving: None,
            status: None,
            show_scores: false,
        }
    }

    fn selected(&self) -> Option<&SearchResult> {
        self.state.selected().and_then(|i| self.results.get(i))
    }

    fn note(&mut self, msg: impl Into<String>) {
        self.status = Some((msg.into(), Instant::now(), false));
    }

    fn warn(&mut self, msg: impl Into<String>) {
        self.status = Some((msg.into(), Instant::now(), true));
    }

    fn move_by(&mut self, delta: isize) {
        if self.results.is_empty() {
            return;
        }
        let len = self.results.len() as isize;
        let cur = self.state.selected().unwrap_or(0) as isize;
        // Clamp rather than wrap: wrapping from the last result back to the first
        // makes it easy to lose your place in a long list.
        let next = (cur + delta).clamp(0, len - 1);
        self.state.select(Some(next as usize));
    }

    /// Re-run the search with the current query text, preserving every other
    /// parameter the user asked for.
    fn refresh(&mut self) {
        // Reranking costs tens of milliseconds: worth it on a committed query,
        // not on every keystroke while the query is still being typed.
        let rr = if self.editing { None } else { self.reranker };
        match search::search(self.paths, self.db, self.embedder, rr, &self.req) {
            Ok(r) => {
                self.results = r;
                self.state
                    .select(if self.results.is_empty() { None } else { Some(0) });
            }
            Err(e) => self.warn(format!("search failed: {e}")),
        }
    }
}

/// Run the browser. Returns when the user exits.
pub fn run(app: &mut App) -> Result<()> {
    let mut term = setup()?;
    let result = event_loop(&mut term, app);
    // Restore the terminal even if the loop failed, so a panic or error does not
    // leave the user in raw mode on the alternate screen.
    restore(&mut term)?;
    result
}

fn setup() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode().context("enabling raw mode")?;
    let mut out = std::io::stdout();
    crossterm::execute!(out, EnterAlternateScreen).context("entering alternate screen")?;
    Terminal::new(CrosstermBackend::new(out)).context("creating terminal")
}

fn restore(term: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().context("disabling raw mode")?;
    crossterm::execute!(term.backend_mut(), LeaveAlternateScreen)
        .context("leaving alternate screen")?;
    term.show_cursor().context("showing cursor")?;
    Ok(())
}

fn event_loop(term: &mut Terminal<CrosstermBackend<Stdout>>, app: &mut App) -> Result<()> {
    loop {
        term.draw(|f| draw(f, app))?;

        // Expire here and only here, so `draw_status` can trust the field.
        if app
            .status
            .as_ref()
            .is_some_and(|(_, t, _)| t.elapsed() > STATUS_TTL)
        {
            app.status = None;
        }

        // Poll rather than block so an expiring status message redraws on time.
        if !event::poll(Duration::from_millis(250))? {
            continue;
        }

        let Event::Key(key) = event::read()? else {
            continue;
        };
        // Without this, every press fires twice on terminals that report both
        // press and release.
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match handle_key(app, key)? {
            Flow::Continue => {}
            Flow::Quit => return Ok(()),
            Flow::Suspend(mut cmd) => {
                restore(term)?;
                let status = cmd.status();
                *term = setup()?;
                term.clear()?;
                match status {
                    Ok(_) => {}
                    Err(e) => app.warn(format!("could not run editor: {e}")),
                }
            }
        }
    }
}

fn handle_key(app: &mut App, key: KeyEvent) -> Result<Flow> {
    // Modal states first: while typing, most keys are text.
    if app.editing {
        return Ok(handle_edit_key(app, key));
    }
    if app.saving.is_some() {
        return Ok(handle_save_key(app, key));
    }

    match key.code {
        // Exit. `Z` and Esc are from the spec; q and ctrl-c are conventional.
        KeyCode::Char('Z') | KeyCode::Char('q') | KeyCode::Esc => return Ok(Flow::Quit),
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            return Ok(Flow::Quit);
        }

        KeyCode::Down | KeyCode::Char('j') => app.move_by(1),
        KeyCode::Up | KeyCode::Char('k') => app.move_by(-1),
        KeyCode::PageDown => app.move_by(10),
        KeyCode::PageUp => app.move_by(-10),
        KeyCode::Home => app.state.select(Some(0)),
        KeyCode::End => {
            if !app.results.is_empty() {
                app.state.select(Some(app.results.len() - 1));
            }
        }

        // `d`: open a terminal in the directory.
        KeyCode::Char('d') => {
            if let Some(dir) = app.selected().map(actions::dir_of) {
                open_terminal_at(app, &dir);
            }
        }

        // `e`: edit in this terminal. Needs the TUI out of the way first.
        KeyCode::Char('e') => {
            if let Some(r) = app.selected() {
                return Ok(Flow::Suspend(actions::editor_command(&r.path)));
            }
        }

        // Enter: whichever of the two the selection implies.
        KeyCode::Enter => {
            if let Some(r) = app.selected() {
                match r.kind {
                    ResultKind::File => {
                        return Ok(Flow::Suspend(actions::editor_command(&r.path)));
                    }
                    ResultKind::Dir => {
                        let dir = r.path.clone();
                        open_terminal_at(app, &dir);
                    }
                }
            }
        }

        // `x`: hand to the desktop.
        KeyCode::Char('x') => {
            if let Some(r) = app.selected() {
                let path = r.path.clone();
                match actions::xdg_open(&path) {
                    Ok(()) => app.note(format!("opened {}", path.display())),
                    Err(e) => app.warn(e.to_string()),
                }
            }
        }

        // `s`: save the result list.
        KeyCode::Char('s') => {
            app.saving = Some("wom-results.txt".to_string());
        }

        // `y`: copy the path.
        KeyCode::Char('y') => {
            if let Some(r) = app.selected() {
                let text = r.path.display().to_string();
                match actions::copy_to_clipboard(&text) {
                    Ok(tool) => app.note(format!("copied path with {tool}")),
                    Err(e) => app.warn(e.to_string()),
                }
            }
        }

        // `/`: refine the query in place.
        KeyCode::Char('/') => app.editing = true,

        KeyCode::Char('S') => {
            app.show_scores = !app.show_scores;
        }

        _ => {}
    }
    Ok(Flow::Continue)
}

fn open_terminal_at(app: &mut App, dir: &std::path::Path) {
    match actions::open_terminal(dir) {
        Ok(()) => app.note(format!("opened terminal in {}", dir.display())),
        Err(e) => app.warn(e.to_string()),
    }
}

fn handle_edit_key(app: &mut App, key: KeyEvent) -> Flow {
    match key.code {
        KeyCode::Esc => app.editing = false,
        KeyCode::Enter => {
            app.editing = false;
            app.refresh();
        }
        KeyCode::Backspace => {
            app.req.text.pop();
            // Live results while typing; the index is fast enough that waiting
            // for Enter would feel sluggish.
            app.refresh();
        }
        KeyCode::Char(c) => {
            app.req.text.push(c);
            app.refresh();
        }
        _ => {}
    }
    Flow::Continue
}

fn handle_save_key(app: &mut App, key: KeyEvent) -> Flow {
    let Some(name) = app.saving.as_mut() else {
        return Flow::Continue;
    };
    match key.code {
        KeyCode::Esc => app.saving = None,
        KeyCode::Backspace => {
            name.pop();
        }
        KeyCode::Char(c) => name.push(c),
        KeyCode::Enter => {
            let path = PathBuf::from(name.clone());
            let query = app.req.text.clone();
            let res = actions::save_results(&app.results, &query, &path);
            app.saving = None;
            match res {
                Ok(n) => app.note(format!("saved {n} results to {}", path.display())),
                Err(e) => app.warn(e.to_string()),
            }
        }
        _ => {}
    }
    Flow::Continue
}

// ------------------------------------------------------------------ drawing

fn draw(f: &mut ratatui::Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // query line
            Constraint::Min(3),    // results + preview
            Constraint::Length(1), // help / status
        ])
        .split(f.area());

    draw_query(f, app, chunks[0]);

    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(chunks[1]);

    draw_results(f, app, panes[0]);
    draw_preview(f, app, panes[1]);
    draw_status(f, app, chunks[2]);
}

fn draw_query(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let mut spans = vec![Span::styled(
        "wom ",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )];
    if !app.req.scope.is_empty() {
        spans.push(Span::styled(
            format!(
                "[{}] ",
                app.req
                    .scope
                    .iter()
                    .map(|p| search::tilde(p))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            Style::default().fg(Color::DarkGray),
        ));
    }
    spans.push(Span::raw(app.req.text.clone()));
    if app.editing {
        spans.push(Span::styled(
            "\u{2588}",
            Style::default().fg(Color::Cyan),
        ));
    }
    spans.push(Span::styled(
        format!("   {} matches", app.results.len()),
        Style::default().fg(Color::DarkGray),
    ));
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_results(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let inner_width = area.width.saturating_sub(2) as usize;

    let items: Vec<ListItem> = app
        .results
        .iter()
        .map(|r| {
            let mut spans = Vec::new();
            if app.show_scores {
                spans.push(Span::styled(
                    match r.cosine {
                        Some(c) => format!("{c:.2} "),
                        None => "  -  ".to_string(),
                    },
                    Style::default().fg(Color::DarkGray),
                ));
                if let Some(rr) = r.rerank {
                    spans.push(Span::styled(
                        format!("rr{rr:+.1} "),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
            }
            match r.kind {
                ResultKind::Dir => {
                    spans.push(Span::styled(
                        "directory:",
                        Style::default().fg(Color::Yellow),
                    ));
                    spans.push(Span::styled(
                        search::tilde(&r.path),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ));
                }
                ResultKind::File => {
                    // Dim the directory part so the filename stands out.
                    let full = search::tilde(&r.path);
                    match full.rfind('/') {
                        Some(i) => {
                            spans.push(Span::styled(
                                full[..=i].to_string(),
                                Style::default().fg(Color::DarkGray),
                            ));
                            spans.push(Span::raw(full[i + 1..].to_string()));
                        }
                        None => spans.push(Span::raw(full)),
                    }
                }
            }
            ListItem::new(Line::from(truncate_spans(spans, inner_width)))
        })
        .collect();

    let title = if app.results.is_empty() {
        " no matches ".to_string()
    } else {
        " results ".to_string()
    };
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");
    f.render_stateful_widget(list, area, &mut app.state);
}

fn draw_preview(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let text = match app.selected() {
        None => "Nothing selected.".to_string(),
        Some(r) => {
            let mut s = String::new();
            s.push_str(&search::tilde(&r.path));
            s.push_str("\n\n");
            if let Some(c) = r.cosine {
                s.push_str(&format!("similarity  {c:.3}\n"));
            }
            let arms = match (r.dense_rank, r.lexical_rank) {
                (Some(d), Some(l)) => format!("matched by  meaning (#{d}) and text (#{l})"),
                (Some(d), None) => format!("matched by  meaning (#{d})"),
                (None, Some(l)) => format!("matched by  text (#{l})"),
                (None, None) => "matched by  contents of this directory".to_string(),
            };
            s.push_str(&arms);
            s.push_str("\n\n");
            match r.file_count {
                Some(n) => s.push_str(&format!("directory holding {n} indexed files")),
                None if r.snippet.is_empty() => {
                    s.push_str("(no extracted text; matched on path and name)")
                }
                None => s.push_str(&r.snippet),
            }
            s
        }
    };
    f.render_widget(
        Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title(" preview "))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_status(f: &mut ratatui::Frame, app: &App, area: Rect) {
    // A prompt or message takes precedence over the key hints.
    if let Some(name) = &app.saving {
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("save to: ", Style::default().fg(Color::Cyan)),
                Span::raw(name.clone()),
                Span::styled("\u{2588}", Style::default().fg(Color::Cyan)),
                Span::styled("   (Enter to write, Esc to cancel)", Style::default().fg(Color::DarkGray)),
            ])),
            area,
        );
        return;
    }
    if let Some((msg, _, is_warning)) = &app.status {
        let style = if *is_warning {
            Style::default().fg(Color::Red)
        } else {
            Style::default().fg(Color::Green)
        };
        f.render_widget(Paragraph::new(Span::styled(msg.clone(), style)), area);
        return;
    }

    let mut hints = vec![
        ("j/k", "move"),
        ("Enter", "open"),
        ("d", "terminal"),
        ("e", "editor"),
    ];
    if actions::has_desktop() {
        hints.push(("x", "xdg-open"));
    }
    hints.extend([("s", "save"), ("y", "copy"), ("/", "refine"), ("q", "quit")]);

    let mut spans = Vec::new();
    for (i, (key, what)) in hints.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ", Style::default()));
        }
        spans.push(Span::styled(
            *key,
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!(" {what}"),
            Style::default().fg(Color::DarkGray),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Trim a styled line to `width` display columns, keeping the tail of the path,
/// which is the informative end.
fn truncate_spans(spans: Vec<Span<'static>>, width: usize) -> Vec<Span<'static>> {
    if width == 0 {
        return Vec::new();
    }
    let total: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    if total <= width {
        return spans;
    }
    // Drop leading characters until it fits, then mark the elision.
    let mut excess = total - width + 1;
    let mut out: Vec<Span<'static>> = Vec::new();
    let mut marked = false;
    for s in spans {
        let n = s.content.chars().count();
        if excess == 0 {
            out.push(s);
            continue;
        }
        if n <= excess {
            excess -= n;
            continue;
        }
        let kept: String = s.content.chars().skip(excess).collect();
        excess = 0;
        if !marked {
            out.push(Span::styled("…", Style::default().fg(Color::DarkGray)));
            marked = true;
        }
        out.push(Span::styled(kept, s.style));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(parts: &[&str]) -> Vec<Span<'static>> {
        parts
            .iter()
            .map(|p| Span::raw(p.to_string()))
            .collect()
    }

    fn rendered(spans: &[Span<'static>]) -> String {
        spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn truncate_spans_keeps_short_lines_untouched() {
        let got = truncate_spans(spans(&["/a/", "b.txt"]), 40);
        assert_eq!(rendered(&got), "/a/b.txt");
    }

    #[test]
    fn truncate_spans_keeps_the_filename_and_fits_the_width() {
        let got = truncate_spans(spans(&["/home/nolan/very/deep/path/", "main.cpp"]), 20);
        let text = rendered(&got);
        assert!(text.chars().count() <= 20, "too wide: {text:?}");
        assert!(text.ends_with("main.cpp"), "lost the filename: {text:?}");
        assert!(text.starts_with('…'), "no elision marker: {text:?}");
    }

    #[test]
    fn truncate_spans_handles_zero_width() {
        assert!(truncate_spans(spans(&["abc"]), 0).is_empty());
    }

    #[test]
    fn truncate_spans_does_not_split_multibyte_characters() {
        // Char-based, not byte-based: a byte split here would panic.
        let got = truncate_spans(spans(&["/дом/очень/длинный/путь/", "файл.txt"]), 15);
        let text = rendered(&got);
        assert!(text.chars().count() <= 15);
        assert!(text.ends_with("файл.txt"));
    }
}
