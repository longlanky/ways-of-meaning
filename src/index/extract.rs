//! Turning a file on disk into the text we embed and index.
//!
//! Two outputs per file: a `body` (extracted content, truncated to the profile's
//! budget) and `name_tokens` (the path and filename split into words). The name
//! tokens matter as much as the body — they are what lets `2024_W2.pdf` answer
//! "employment documents" even when the PDF text extracts poorly.

use crate::config::{Config, Profile};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use super::geo;

/// Which extractor produced a body, recorded so `wom status` can report coverage
/// and so a later run can retry files that were skipped for a fixable reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Text,
    Pdf,
    Office,
    Media,
    /// No body: either a binary we cannot read, or the `name-only` profile. The
    /// file is still indexed and findable by path.
    NameOnly,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Pdf => "pdf",
            Kind::Office => "office",
            Kind::Media => "media",
            Kind::NameOnly => "name-only",
        }
    }
}

pub struct Extracted {
    pub kind: Kind,
    pub body: String,
    /// First ~300 characters, for the TUI preview pane.
    pub snippet: String,
}

/// Which optional external tools exist on this machine. Probed once at startup:
/// a missing tool degrades that file type to name-only rather than failing.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    pub pdftotext: bool,
    pub exiftool: bool,
}

impl Capabilities {
    pub fn detect() -> Self {
        Self {
            pdftotext: crate::actions::which("pdftotext"),
            exiftool: crate::actions::which("exiftool"),
        }
    }

    /// Human-readable summary for `wom index`, so a user who wonders why their
    /// PDFs are not searchable gets told rather than having to guess.
    pub fn describe_gaps(&self, geo: bool) -> Vec<String> {
        let mut out = Vec::new();
        if !self.pdftotext {
            out.push(
                "pdftotext not found: PDFs will be indexed by filename only \
                 (install poppler-utils)"
                    .to_string(),
            );
        }
        if geo && !self.exiftool {
            out.push(
                "extract.geo is on but exiftool was not found: GPS locations will \
                 not be indexed (install libimage-exiftool-perl)"
                    .to_string(),
            );
        }
        out
    }
}

// ------------------------------------------------------------------ dispatch

/// Extract the body text for one file.
///
/// Never returns an error for an individual unreadable file: at this scale a
/// permission error or a malformed PDF is routine, and aborting a 100k-file scan
/// over one of them would be wrong. Such files degrade to `NameOnly`.
pub fn extract(path: &Path, profile: Profile, cfg: &Config, caps: &Capabilities) -> Extracted {
    let budget_chars = body_budget_chars(profile);
    if budget_chars == 0 {
        return Extracted {
            kind: Kind::NameOnly,
            body: String::new(),
            snippet: String::new(),
        };
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    let (kind, body) = match ext.as_str() {
        "pdf" if caps.pdftotext => (Kind::Pdf, pdf_text(path, cfg).unwrap_or_default()),
        "docx" | "odt" | "pptx" | "odp" | "xlsx" | "ods" => {
            (Kind::Office, office_text(path).unwrap_or_default())
        }
        _ if is_media_ext(&ext) => {
            if cfg.extract.media {
                (Kind::Media, media_text(path, caps, cfg.extract.geo).unwrap_or_default())
            } else {
                (Kind::NameOnly, String::new())
            }
        }
        _ => match text_head(path, cfg, budget_chars) {
            Some(t) => (Kind::Text, t),
            None => (Kind::NameOnly, String::new()),
        },
    };

    let body = normalise_ws(&body, budget_chars);
    let snippet = take_chars(&body, 300);
    // An extractor that ran but produced nothing is recorded as name-only, so the
    // coverage numbers in `wom status` reflect what is actually searchable rather
    // than which extractor was attempted.
    let kind = if body.is_empty() { Kind::NameOnly } else { kind };
    Extracted {
        kind,
        body,
        snippet,
    }
}

/// Character budget for the body.
///
/// The profile budget is in tokens; English averages roughly 4 characters per
/// token, so this converts approximately. Erring long is harmless because the
/// tokenizer truncates to the model's positional limit anyway; erring short
/// would silently discard content.
fn body_budget_chars(profile: Profile) -> usize {
    profile.body_tokens() * 4
}

fn is_media_ext(ext: &str) -> bool {
    matches!(
        ext,
        "jpg"
            | "jpeg"
            | "png"
            | "gif"
            | "webp"
            | "tiff"
            | "tif"
            | "bmp"
            | "heic"
            | "avif"
            | "raw"
            | "cr2"
            | "nef"
            | "mp4"
            | "mkv"
            | "avi"
            | "mov"
            | "webm"
            | "flv"
            | "wmv"
            | "mp3"
            | "flac"
            | "wav"
            | "ogg"
            | "opus"
            | "m4a"
            | "aac"
    )
}

// ------------------------------------------------------------------ text

/// Read the head of a file as text, or `None` if it looks binary.
///
/// `budget_chars` is how much text the caller will actually keep. Reading much
/// past that is wasted I/O, and at this corpus size the waste dominates: with a
/// flat 64 KB read, indexing 274k files moved ~17 GB off disk to retain a few tens
/// of megabytes. The `code` profile keeps ~256 characters per file, so it was
/// reading roughly 250x what it used.
fn text_head(path: &Path, cfg: &Config, budget_chars: usize) -> Option<String> {
    let want = read_size(cfg, budget_chars);
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; want];
    let n = f.read(&mut buf).ok()?;
    buf.truncate(n);
    if looks_binary(&buf) {
        return None;
    }
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Bytes to read for a body of `budget_chars` characters.
///
/// 8 bytes per kept character covers multibyte text and the whitespace that
/// collapsing removes. The 4 KB floor keeps the binary sniff meaningful and costs
/// nothing, being well under one readahead window; the configured ceiling wins
/// over the floor if the user set it lower.
fn read_size(cfg: &Config, budget_chars: usize) -> usize {
    let ceiling = ((cfg.extract.max_read_kb * 1024) as usize).min(1 << 20);
    // Deliberately not `clamp(4096, ceiling)`: that panics when the configured
    // ceiling is below the floor. Applying the ceiling last lets an explicitly
    // small `max_read_kb` win, which is what the user asked for.
    budget_chars.saturating_mul(8).max(4096).min(ceiling)
}

/// A NUL byte in the first 8 KB is the standard heuristic for "binary", and it is
/// what `grep` and `git` use. Also treats a high proportion of control bytes as
/// binary, which catches files that happen to avoid NUL.
pub fn looks_binary(buf: &[u8]) -> bool {
    let head = &buf[..buf.len().min(8192)];
    if head.contains(&0) {
        return true;
    }
    if head.is_empty() {
        return false;
    }
    let weird = head
        .iter()
        .filter(|b| **b < 0x09 || (**b > 0x0d && **b < 0x20))
        .count();
    weird * 100 / head.len() > 10
}

// ------------------------------------------------------------------ pdf

fn pdf_text(path: &Path, cfg: &Config) -> Option<String> {
    let out = Command::new("pdftotext")
        .arg("-q")
        .arg("-l")
        .arg(cfg.extract.pdf_pages.to_string())
        .arg("-enc")
        .arg("UTF-8")
        .arg(path)
        .arg("-")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    // pdftotext exits non-zero on damaged files while still emitting usable text
    // for the pages it managed, so take whatever came out.
    let s = String::from_utf8_lossy(&out.stdout).into_owned();
    if s.trim().is_empty() { None } else { Some(s) }
}

// ------------------------------------------------------------------ office

/// Pull text out of an OOXML or ODF container.
///
/// Pure Rust via zip + quick-xml rather than shelling out to libreoffice, which
/// takes seconds per file and would dominate a scan.
fn office_text(path: &Path) -> Option<String> {
    let f = std::fs::File::open(path).ok()?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(f)).ok()?;

    // Parts worth reading. Names differ between OOXML and ODF. Collecting indices
    // rather than names avoids opening each entry twice.
    let wanted: Vec<usize> = (0..zip.len())
        .filter(|i| {
            zip.by_index_raw(*i)
                .map(|e| {
                    let n = e.name();
                    n == "word/document.xml"
                        || n == "content.xml"
                        || n == "xl/sharedStrings.xml"
                        || (n.starts_with("ppt/slides/slide") && n.ends_with(".xml"))
                })
                .unwrap_or(false)
        })
        .collect();

    let mut out = String::new();
    for i in wanted {
        let Ok(mut entry) = zip.by_index(i) else {
            continue;
        };
        let mut xml = String::new();
        if entry.read_to_string(&mut xml).is_err() {
            continue;
        }
        out.push_str(&xml_text(&xml));
        out.push(' ');
        if out.len() > 200_000 {
            break;
        }
    }
    if out.trim().is_empty() { None } else { Some(out) }
}

/// Concatenate the character data of an XML document, ignoring markup.
fn xml_text(xml: &str) -> String {
    use quick_xml::events::Event;
    let mut r = quick_xml::Reader::from_str(xml);
    r.config_mut().trim_text(true);
    let mut out = String::new();
    let mut buf = Vec::new();
    loop {
        match r.read_event_into(&mut buf) {
            Ok(Event::Text(t)) => {
                if let Ok(s) = t.decode() {
                    out.push_str(&s);
                    out.push(' ');
                }
            }
            // Word wraps every run in <w:t>, so paragraphs need an explicit break
            // or words from adjacent paragraphs would run together.
            Ok(Event::End(e)) if e.name().as_ref().ends_with(b"p") => out.push('\n'),
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    out
}

// ------------------------------------------------------------------ media

fn media_text(path: &Path, caps: &Capabilities, geo: bool) -> Option<String> {
    if !caps.exiftool {
        return None;
    }
    // Values only, in the order the tags were requested. GPS goes last so the
    // signed-decimal pair (`-n`) lands at the end of the output, which is where
    // parse_media_output looks for it.
    let mut args = vec!["-s", "-s", "-s", "-Title", "-Description", "-Keywords", "-Artist", "-Album"];
    if geo {
        args.extend(["-n", "-GPSLatitude", "-GPSLongitude"]);
    }
    let out = Command::new("exiftool")
        .args(&args)
        .arg(path)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).into_owned();
    if s.trim().is_empty() {
        return None;
    }
    if !geo {
        return Some(s);
    }
    let (text, coords) = parse_media_output(&s);
    match coords.and_then(|(lat, lon)| geo::nearest_place(lat, lon)) {
        // The place goes first: body truncation keeps the head, and where a
        // photo was taken is the most durable thing to say about it.
        Some(place) => Some(format!("{place}\n{text}")),
        None => Some(text),
    }
}

/// Split exiftool's values-only output into metadata text and an optional GPS
/// pair. With `-n`, coordinates print as signed decimals one per line, and the
/// GPS tags were requested last — so two trailing in-range floats mean GPS
/// data was present. Everything else stays text.
fn parse_media_output(s: &str) -> (String, Option<(f64, f64)>) {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    let n = lines.len();
    if n >= 2 {
        let lat = lines[n - 2].trim().parse::<f64>().ok();
        let lon = lines[n - 1].trim().parse::<f64>().ok();
        if let (Some(lat), Some(lon)) = (lat, lon) {
            if (-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon) {
                return (lines[..n - 2].join("\n"), Some((lat, lon)));
            }
        }
    }
    (s.to_string(), None)
}

// ------------------------------------------------------------------ names

/// Split a path into searchable words: the filename and each parent directory,
/// broken on separators and camelCase boundaries.
///
/// This is what makes `wom sensors monitor` find
/// `esp32_prayerTimesWithWeather_n5endpoint/src/main.cpp` — the meaning is in the
/// path, not in the C++.
pub fn name_tokens(path: &Path, root: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let mut words: Vec<String> = Vec::new();
    for comp in rel.components() {
        let s = comp.as_os_str().to_string_lossy();
        split_words(&s, &mut words);
    }
    dedupe_preserving_order(&mut words);
    words.join(" ")
}

/// Break one path component into words on non-alphanumeric separators and at
/// lower->upper case transitions.
pub fn split_words(s: &str, out: &mut Vec<String>) {
    let mut cur = String::new();
    let mut prev_lower = false;
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            // camelCase / PascalCase boundary.
            if prev_lower && ch.is_uppercase() && !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            cur.push(ch.to_ascii_lowercase());
            prev_lower = ch.is_lowercase();
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            prev_lower = false;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
}

fn dedupe_preserving_order(words: &mut Vec<String>) {
    let mut seen = std::collections::HashSet::new();
    words.retain(|w| w.len() > 1 && seen.insert(w.clone()));
}

/// Assemble the text handed to the embedding model.
///
/// Order is deliberate: the relative path first, then the name words, then the
/// body. With CLS pooling the leading tokens carry disproportionate weight, and
/// for most files the path is the most reliable signal of what it is.
pub fn embed_text(path: &Path, root: &Path, name_tokens: &str, body: &str) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let mut s = String::with_capacity(body.len() + 256);
    s.push_str(&rel.to_string_lossy());
    s.push('\n');
    s.push_str(name_tokens);
    if !body.is_empty() {
        s.push_str("\n\n");
        s.push_str(body);
    }
    s
}

// ------------------------------------------------------------------ helpers

/// Collapse runs of whitespace and truncate to `budget` characters.
///
/// Extracted text is full of layout whitespace, which wastes tokens the model
/// could spend on words.
fn normalise_ws(s: &str, budget: usize) -> String {
    let mut out = String::with_capacity(s.len().min(budget + 16));
    // Counting kept characters as we go, rather than calling `out.chars().count()`
    // per input character, which made this quadratic in the body length — over
    // 274k files with budgets up to 1024 characters, that dominated extraction.
    let mut kept = 0usize;
    let mut last_ws = true;
    for ch in s.chars() {
        if kept >= budget {
            break;
        }
        if ch.is_whitespace() {
            if !last_ws {
                out.push(' ');
                kept += 1;
                last_ws = true;
            }
        } else if ch.is_control() {
            continue;
        } else {
            out.push(ch);
            kept += 1;
            last_ws = false;
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// blake3 of the text we will embed. Change detection compares this so a file
/// whose mtime moved but whose content did not is never re-embedded.
pub fn content_hash(text: &str) -> Vec<u8> {
    blake3::hash(text.as_bytes()).as_bytes()[..16].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn binary_detection_uses_nul_and_control_density() {
        assert!(looks_binary(b"\x7fELF\x02\x01\x01\x00"));
        assert!(!looks_binary(b"fn main() {}\n"));
        assert!(!looks_binary(b""));
        assert!(!looks_binary("héllo wörld\n".as_bytes()));
        // Dense control bytes without a NUL.
        assert!(looks_binary(&[0x01u8; 64]));
    }

    #[test]
    fn split_words_breaks_on_separators_and_camel_case() {
        let mut w = Vec::new();
        split_words("esp32_prayerTimesWithWeather_n5endpoint", &mut w);
        assert_eq!(
            w,
            vec!["esp32", "prayer", "times", "with", "weather", "n5endpoint"]
        );
    }

    #[test]
    fn split_words_handles_the_specs_filenames() {
        let mut w = Vec::new();
        split_words("2024_W2.pdf", &mut w);
        assert_eq!(w, vec!["2024", "w2", "pdf"]);

        w.clear();
        split_words("dubrovnik_to_sarajevo.pdf", &mut w);
        assert_eq!(w, vec!["dubrovnik", "to", "sarajevo", "pdf"]);
    }

    #[test]
    fn name_tokens_include_parent_directories() {
        let toks = name_tokens(
            Path::new("/r/bosnia_croatia_trip_aug2024/dubrovnik_to_sarajevo.pdf"),
            Path::new("/r"),
        );
        for want in ["bosnia", "croatia", "trip", "dubrovnik", "sarajevo"] {
            assert!(toks.contains(want), "{want:?} missing from {toks:?}");
        }
    }

    #[test]
    fn name_tokens_deduplicate_repeated_components() {
        // Deep source trees repeat the project name at several levels; keeping one
        // copy avoids drowning the body in redundant tokens.
        let toks = name_tokens(
            Path::new("/r/wom/src/wom/wom.rs"),
            Path::new("/r"),
        );
        assert_eq!(toks.matches("wom").count(), 1, "got {toks:?}");
    }

    #[test]
    fn name_tokens_drop_single_characters() {
        let toks = name_tokens(Path::new("/r/a/b/notes.md"), Path::new("/r"));
        assert_eq!(toks, "notes md");
    }

    #[test]
    fn embed_text_leads_with_the_path() {
        let s = embed_text(
            Path::new("/r/employment_docs/2024_W2.pdf"),
            Path::new("/r"),
            "employment docs 2024 w2 pdf",
            "Wage and Tax Statement",
        );
        assert!(s.starts_with("employment_docs/2024_W2.pdf\n"));
        assert!(s.contains("Wage and Tax Statement"));
    }

    #[test]
    fn embed_text_omits_the_blank_line_when_there_is_no_body() {
        let s = embed_text(Path::new("/r/a.png"), Path::new("/r"), "a png", "");
        assert_eq!(s, "a.png\na png");
    }

    #[test]
    fn normalise_ws_collapses_runs_and_respects_the_budget() {
        assert_eq!(normalise_ws("a  \n\t b   c ", 100), "a b c");
        assert_eq!(normalise_ws("abcdefghij", 4), "abcd");
    }

    #[test]
    fn normalise_ws_budget_counts_characters_not_bytes() {
        // A byte-based truncation here would split a multibyte char and panic.
        let out = normalise_ws("ααααααααα", 4);
        assert_eq!(out.chars().count(), 4);
    }

    #[test]
    fn name_only_profile_extracts_no_body() {
        let dir = std::env::temp_dir().join("wom-extract-nameonly");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.txt");
        std::fs::write(&p, "some real text here").unwrap();

        let got = extract(&p, Profile::NameOnly, &Config::default(), &Capabilities::detect());
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(got.kind, Kind::NameOnly);
        assert!(got.body.is_empty());
    }

    #[test]
    fn read_size_scales_with_the_profile_budget() {
        // Regression guard for the 250x-over-read: the code profile must not pull
        // 64 KB off disk to retain 256 characters.
        let cfg = Config::default();
        let ceiling = (cfg.extract.max_read_kb * 1024) as usize;

        let code = read_size(&cfg, body_budget_chars(Profile::Code));
        let content = read_size(&cfg, body_budget_chars(Profile::Content));

        assert!(code < content, "code should read less than content");
        assert!(
            code < ceiling / 4,
            "code profile still reads {code} of a {ceiling} ceiling"
        );
        assert!(code >= 4096, "must still read enough to sniff for binary");
        assert!(content <= ceiling, "cannot exceed the configured ceiling");
    }

    #[test]
    fn read_size_respects_a_tiny_configured_ceiling() {
        let mut cfg = Config::default();
        cfg.extract.max_read_kb = 1;
        // The 4 KB floor must not override an explicit, smaller ceiling.
        assert!(read_size(&cfg, 100_000) <= 4096);
    }

    #[test]
    fn long_file_still_fills_the_content_budget_after_the_read_shrank() {
        // The read has to stay generous enough that truncation is driven by the
        // budget, not by how much was read.
        let dir = std::env::temp_dir().join("wom-extract-fill");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("long.txt");
        std::fs::write(&p, "alpha beta gamma delta ".repeat(2000)).unwrap();

        let got = extract(&p, Profile::Content, &Config::default(), &Capabilities::detect());
        std::fs::remove_dir_all(&dir).ok();

        let budget = body_budget_chars(Profile::Content);
        assert!(
            got.body.chars().count() >= budget - 32,
            "body was {} chars, expected close to the {budget} budget",
            got.body.chars().count()
        );
    }

    #[test]
    fn code_profile_truncates_more_aggressively_than_content() {
        let dir = std::env::temp_dir().join("wom-extract-budget");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("big.txt");
        std::fs::write(&p, "word ".repeat(4000)).unwrap();
        let cfg = Config::default();
        let caps = Capabilities::detect();

        let code = extract(&p, Profile::Code, &cfg, &caps);
        let content = extract(&p, Profile::Content, &cfg, &caps);
        std::fs::remove_dir_all(&dir).ok();

        assert!(code.body.len() < content.body.len());
        assert!(code.body.len() <= body_budget_chars(Profile::Code));
        assert!(content.body.len() <= body_budget_chars(Profile::Content));
    }

    #[test]
    fn binary_file_degrades_to_name_only_without_erroring() {
        let dir = std::env::temp_dir().join("wom-extract-binary");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("blob.dat");
        std::fs::write(&p, [0u8, 1, 2, 3, 0, 5]).unwrap();

        let got = extract(&p, Profile::Content, &Config::default(), &Capabilities::detect());
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(got.kind, Kind::NameOnly);
        assert!(got.body.is_empty());
    }

    #[test]
    fn unreadable_file_degrades_instead_of_panicking() {
        let missing = PathBuf::from("/definitely/not/here.txt");
        let got = extract(
            &missing,
            Profile::Content,
            &Config::default(),
            &Capabilities::detect(),
        );
        assert_eq!(got.kind, Kind::NameOnly);
    }

    #[test]
    fn xml_text_extracts_character_data_and_breaks_paragraphs() {
        let xml = r#"<w:document><w:body>
            <w:p><w:r><w:t>Wage and Tax</w:t></w:r><w:r><w:t>Statement</w:t></w:r></w:p>
            <w:p><w:r><w:t>Employer</w:t></w:r></w:p>
        </w:body></w:document>"#;
        let out = xml_text(xml);
        assert!(out.contains("Wage and Tax"));
        assert!(out.contains("Statement"));
        assert!(out.contains("Employer"));
        // The paragraph break must stop "Statement" and "Employer" merging.
        assert!(!out.contains("StatementEmployer"));
    }

    #[test]
    fn content_hash_is_stable_and_discriminating() {
        assert_eq!(content_hash("abc"), content_hash("abc"));
        assert_ne!(content_hash("abc"), content_hash("abd"));
        assert_eq!(content_hash("abc").len(), 16);
    }

    #[test]
    fn media_output_parses_a_trailing_gps_pair() {
        let (text, coords) = parse_media_output("Sunset over the old bridge\n43.8563\n18.4131");
        assert_eq!(text, "Sunset over the old bridge");
        let (lat, lon) = coords.expect("a lat/lon pair");
        assert!((lat - 43.8563).abs() < 1e-9);
        assert!((lon - 18.4131).abs() < 1e-9);
        // ...and that pair is in fact Sarajevo; nearest_place agrees it is
        // Bosnia, which is the whole point of the feature.
        let place = geo::nearest_place(lat, lon).unwrap();
        assert!(place.contains("Bosnia"), "got {place}");
    }

    #[test]
    fn media_output_without_gps_stays_text() {
        let (text, coords) = parse_media_output("A title\nSome keywords");
        assert_eq!(text, "A title\nSome keywords");
        assert!(coords.is_none());

        // One float line is not a pair.
        let (_, coords) = parse_media_output("43.8563");
        assert!(coords.is_none());

        // In-range-looking but out-of-range numbers are not coordinates.
        let (_, coords) = parse_media_output("shot 2024\n120.5");
        assert!(coords.is_none());
    }
}
