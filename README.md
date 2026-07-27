# Ways of Meaning (`wom`)

Semantic file search for Linux. Ask for what you mean, not for what the filename
happens to say.

```
$ wom employment documents
directory:~/Documents/employment_docs
~/Documents/employment_docs/2024_W2.pdf
~/Documents/employment_docs/procedural_notes.txt
~/Documents/employment_docs/2023_W2.pdf

$ wom bosnia
directory:~/Pictures/bosnia_croatia_trip_aug2024
~/Documents/travel/dubrovnik_to_sarajevo.pdf

$ wom ~/projects/ sensors monitor
~/projects/2026_esp32epaper_display/src/main.cpp
```

None of those queries are substring matches. "employment documents" finds a W-2
because the model knows what a W-2 is; "balkans holiday" would find the same trip
directory as "bosnia".

## Install

```sh
cargo build --release
install -Dm755 target/release/wom ~/.local/bin/wom
```

Optional external tools, detected at runtime and degraded gracefully if absent:

| Tool | Gives you |
|---|---|
| `pdftotext` (poppler-utils) | PDF text. Without it, PDFs are found by filename only. |
| `exiftool` | Image/audio/video metadata, when `extract.media = true`. |
| `wl-clipboard` / `xclip` / `xsel` | The `y` (copy path) key in the browser. |

Office formats (`.docx`, `.odt`, `.xlsx`, `.pptx`, …) need nothing extra — they
are read directly.

## Getting started

```sh
wom init      # choose directories, see the real file count and time cost first
wom index     # build the index (one-off; later runs only process changes)
wom <words>   # search
```

`wom init` proposes your XDG user directories plus `~/projects`-style locations,
prints how many files each holds after ignore rules, and asks before writing
anything. Add directories explicitly with `wom init --add ~/notes --add ~/work`.

## Searching

```sh
wom employment documents            # interactive browser on a terminal
wom bosnia | head -3                # plain lines when piped
wom ~/projects/ sensors monitor     # leading directories restrict the search
wom -i ~/projects sensors monitor   # same thing, unambiguously
wom notes --lexical                 # full-text only, no model load (~5 ms)
wom notes --dense                   # vector similarity only
wom notes --scores                  # show fused score and cosine
```

A leading argument becomes a search scope only if it both exists as a directory
*and* looks like a path (contains `/`, or starts with `~`/`.`). So `wom bosnia`
searches for "bosnia" even if `./bosnia` exists.

### Keys in the browser

| Key | Action |
|---|---|
| `j`/`k`, `↑`/`↓`, PgUp/PgDn, Home/End | Move |
| `Enter` | Open — directories in a terminal, files in `$EDITOR` |
| `d` | Open the directory in `$TERMINAL` |
| `e` | Open in `$EDITOR` (suspends the browser) |
| `x` | `xdg-open` (only offered under a desktop session) |
| `s` | Save the result list to a file |
| `y` | Copy the path |
| `/` | Refine the query, re-searching as you type |
| `S` | Toggle similarity scores |
| `Z`, `Esc`, `q` | Exit |

## How it works

One vector per file, from its path, its filename split into words, and the head of
its extracted text. Filename and path carry real signal — `2024_W2.pdf` inside
`employment_docs/` is recognisable even when the PDF itself extracts badly.

Retrieval fuses five ranked lists with Reciprocal Rank Fusion:

- files by vector similarity, and by full-text (BM25, filename weighted 3×)
- directories by vector similarity, and by full-text
- directories credited by how well their children ranked

Both halves are necessary. Vector search understands "employment documents" but is
weak on proper nouns it never trained on; `wom bosnia` reaches
`bosnia_croatia_trip_aug2024` through the lexical arm. RRF combines them without
needing cosine and BM25 to be on a comparable scale.

Directories are searchable in their own right. A directory's vector blends its
name with the centroid of its contents, weighted by how *coherent* those contents
are (`‖mean‖` of the member unit vectors). Without that weighting, a directory
holding a bit of everything lands near the middle of embedding space and scores
0.6+ against every unrelated query — measured, not hypothetical.

Vectors are int8-quantised in a flat mmap'd file and searched by brute force. No
approximate index: no build step, no tuning, no staleness, and exact results.

## Measured on a real corpus

274,379 files (the author's XDG directories plus two source trees, after ignore
rules), on a Ryzen 7 PRO 6850U with 16 threads and ~3.2 GB free RAM:

| | |
|---|---|
| First index | **77 minutes** at 59 docs/sec end-to-end |
| Peak resident memory | 1.5 GB |
| File vectors | 103 MB (274,379 × 384-dim int8) |
| Directory vectors | 9 MB (21,696 directories) |
| SQLite index | 488 MB (mostly the FTS5 body text) |
| Search latency | **p50 21 ms, p95 52 ms** in process |
| Whole `wom <query>` invocation | ~140 ms, dominated by loading the model |
| Rescan with nothing changed | seconds |
| Text extracted from | 251,148 files; 23,017 by name only |
| Golden retrieval set | 6/6, including queries sharing no words with the answer |

`wom bench` reports *inference-only* throughput (123 docs/sec here). Extraction and
database writes roughly halve that, which is why the projection prefers a rate a
real scan measured, and says which of the two it used.

The ~140 ms per invocation is graph loading, not search: a CLI starts fresh each
time. Caching ORT's already-optimised graph next to the model cut it from 300 ms,
and `--lexical` skips the model entirely at under 10 ms.

### Profiles

Directories are indexed under one of three profiles, auto-detected at `init` and
overridable per root in the config:

| Profile | Embeds | For |
|---|---|---|
| `content` | Path, name, first ~256 tokens | Documents |
| `code` | Path, name, first ~64 tokens | Source trees |
| `name-only` | Path and name | Media |

`code` is not just cheaper — for a 5000-line driver the leading comment, includes
and declarations identify the file, and the middle is noise.

## Keeping it current

```toml
refresh = "daily"   # "every-run" | "daily" | "weekly" | "manual"
```

A query that finds the schedule due starts a scan in a detached background
process and answers immediately from the existing index. Rescans are cheap: files
whose `(mtime, size)` are unchanged are skipped without being opened, and files
whose text hashes the same are updated without being re-embedded.

```sh
wom index          # respects the schedule
wom index --now    # scan regardless
wom index --force  # re-embed everything
wom index --rebuild
wom status --verify
```

## Models

```sh
wom model list
wom model set bge-base-en-v1.5   # invalidates stored vectors; then `wom index --rebuild`
wom model self-test              # checks pooling and the query prefix
wom bench --recall               # throughput and retrieval quality on your corpus
```

Default is `bge-small-en-v1.5-int8`. `bge-base` and `bge-large` are available and
substantially slower; `wom bench` will tell you what that costs on your machine
rather than leaving you to guess.

Two details this implementation gets right that are easy to get wrong, because
both fail *silently* — producing well-formed vectors and merely worse results:

- **BGE uses CLS pooling, not mean pooling** (per its `1_Pooling/config.json`).
- **BGE is asymmetric**: queries get the prefix `"Represent this sentence for
  searching relevant passages: "`, documents get none.

`wom model self-test` asserts both, and `cargo test --test retrieval` checks
queries that share no vocabulary with their expected answers — which is what
would actually break if either were wrong.

## GPU acceleration

Short answer: it works, and it is not worth using. Measured, not assumed.

`wom` can run the model on the GPU through ONNX Runtime's WebGPU execution
provider, which uses Dawn and can target Vulkan. That is the only viable GPU route
for an AMD iGPU on Linux — the ROCm provider needs a ROCm install, and DirectML is
Windows-only.

```sh
cargo build --release --features gpu
# The Dawn shared library is not installed system-wide:
export LD_LIBRARY_PATH="$PWD/target/release:$LD_LIBRARY_PATH"
```

Then set `device = "auto"` (fall back to CPU if unavailable) or `device = "gpu"`
(fail rather than silently fall back).

On a Radeon 680M iGPU (Ryzen 7 PRO 6850U, 16 CPU threads, Vulkan via Mesa RADV),
512 sample documents:

| Model | Device | docs/sec | Single query |
|---|---|---:|---:|
| bge-small **int8** | **CPU** | **64** | **15 ms** |
| bge-small int8 | GPU | 20 | 309 ms |
| bge-small fp32 | GPU | 51 | 61 ms |
| bge-small fp32 | CPU | 44 | 26 ms |

Two things fall out of that:

- **For fp32 the GPU genuinely helps** (51 vs 44), so the plumbing is working.
- **int8 on the GPU collapses** (20 vs 64). Quantised operators are not implemented
  in the WebGPU provider, so they fall back to CPU per-node and every fallback pays
  a round trip across the bus.
- The fastest configuration overall is still **int8 on the CPU**, which beats the
  best GPU configuration by 25%. And single-query latency — the number an
  interactive CLI actually lives on — is 2–20× worse on the GPU, because kernel
  launch overhead dwarfs the work in one small batch.

An integrated GPU also shares system memory with the CPU, so there is no bandwidth
advantage to offset any of this.

There is one further caveat: the process **reliably dumps core during exit** when
Dawn tears down. Results are computed and printed correctly first, but the exit
status is unusable for scripting. That alone would keep this off by default.

The default is therefore `device = "cpu"`, and the `gpu` feature is not compiled in
unless you ask for it. The `Embedder` trait keeps the door open for a better
backend later.

## Configuration

`~/.config/wom/config.toml`; data in `~/.local/share/wom/`. Set `WOM_HOME` to put
both somewhere else.

```toml
model = "bge-small-en-v1.5-int8"
refresh = "daily"
limit = 40
# min_similarity = 0.55  # optional; defaults to the model's own floor

[[roots]]
path = "/home/you/Documents"
profile = "content"

[[roots]]
path = "/home/you/projects"
profile = "code"

[extract]
media = false           # exiftool per media file; slow
max_file_size_mb = 20
max_read_kb = 64
pdf_pages = 5

[ignore]
use_gitignore = true
hidden = false
follow_symlinks = false
extra = ["*.bak", "scratch/"]
unignore = ["vendor/"]   # switch off a builtin rule
```

`min_similarity` defaults to a per-model value from the registry, because the
right floor depends on how a model spaces related from unrelated text — on
bge-small, unrelated short texts sit around 0.40–0.46 and genuine matches above
0.67, so 0.55 falls in the gap. Set it explicitly to override, or pass
`--min-similarity` for one query. The lexical arm ignores it, so exact filename
matches always come through.

## Testing

```sh
cargo test                    # unit tests
cargo test --test retrieval   # end-to-end; downloads the model on first run
WOM_SKIP_MODEL=1 cargo test   # skip the cases needing a model
```
