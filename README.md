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
| `exiftool` | Image/audio/video metadata, when `extract.media = true`; GPS coordinates resolved to place names when `extract.geo = true`. |
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
wom notes --rerank                  # cross-encoder second opinion on the top results
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

Optionally, a **cross-encoder reranker** gives the top fused candidates a second,
more expensive opinion: the embedding model scores query and document separately,
while the reranker reads the (query, document) pair together and is correspondingly
more accurate. Off by default; enable per query with `--rerank`, or set
`reranker = "ms-marco-MiniLM-L-6-v2-int8"` in the config (`--no-rerank` overrides).
In the interactive browser it runs on the committed query only, never per
keystroke. `wom model list` shows the registered rerankers, and `wom bench`
measures what it costs on your machine — here, 80 pairs in ~0.7 s (~9 ms/pair),
so a reranked query lands at ~1 s all-in against ~0.25 s without.

Directories are searchable in their own right. A directory's vector blends its
name with the centroid of its contents, weighted by how *coherent* those contents
are (`‖mean‖` of the member unit vectors). Without that weighting, a directory
holding a bit of everything lands near the middle of embedding space and scores
0.6+ against every unrelated query — measured, not hypothetical.

Vectors are int8-quantised in a flat mmap'd file and searched by brute force. No
approximate index: no build step, no tuning, no staleness, and exact results.

### Photo locations

With `extract.media = true` and `extract.geo = true`, a photo's EXIF GPS
coordinates are resolved to the nearest city and indexed as text — fully
offline, using an embedded GeoNames extract (`data/cities.tsv`, CC-BY 4.0), so
photo locations never leave the machine. `wom iceland` then finds the pictures
you took near Akureyri even though the files are named `IMG_1399.jpg`.

Already-indexed photos are skipped by change detection, so enabling geo on an
existing index takes one `wom index --force` to re-read them.

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

Default is `bge-small-en-v1.5-int8`. Larger and alternative families are
registered — `bge-base`/`bge-large` (also int8), `mxbai-embed-large-v1` (also
int8), `gte-large-en-v1.5` (also int8), and `all-MiniLM-L6-v2` — and `wom bench`
will tell you what each costs on your machine rather than leaving you to guess.

Measured here on the 8,029-file validation corpus (end-to-end indexing rate,
single-query latency, the paraphrase-vs-unrelated cosine margin `wom model
self-test` reports, and the golden retrieval set):

| Model | docs/sec | Query | Margin | Recall |
|---|---:|---:|---:|---:|
| bge-small-en-v1.5-int8 | 58 | 4.9 ms | 0.34 | 6/6 |
| bge-large-en-v1.5-int8 | 10 | 30 ms | 0.53 | 6/6 |
| mxbai-embed-large-v1-int8 | 10 | 28 ms | 0.58 | 6/6 |
| gte-large-en-v1.5-int8 | 1 | 592 ms | **0.09** | 4/6 |

The large int8 models separate related from unrelated text noticeably better
(margin 0.53–0.58 vs 0.34) at ~6x the indexing cost and ~5x the query latency —
worth it if you search more than you index. mxbai has the widest margin.
gte-large-en-v1.5 measured badly on this machine in every dimension — 60x
slower than bge-small, a thin margin, and worse recall — so its similarity
floor defaults to 0.80 and it stays registered only for completeness.

### Which model should you use?

| Tier | Model | Cost (this machine) | When |
|---|---|---|---|
| **Light** (default) | `bge-small-en-v1.5-int8` | 8k files in ~2 min; 274k in ~80 min; 5 ms queries | Daily driver; quality is already good |
| **Medium** | `bge-base-en-v1.5` | ~3x light (estimated, not measured) | No compelling niche — skip to heavy if quality matters |
| **Heavy** | `mxbai-embed-large-v1-int8` | 8k files in ~13 min; 274k in ~8 h; 28 ms queries | Best separation (margin 0.58); search more than you index |

`bge-large-en-v1.5-int8` is the equally good heavy alternative (margin 0.53).
The fp32 originals are ~2.5x slower with no measured benefit, and
`gte-large-en-v1.5` is not recommended at any tier (see the table above).
Switching takes two commands:

```sh
wom model set mxbai-embed-large-v1-int8
wom index --rebuild   # required — vectors from different models are not comparable
```

### Rerankers

A cross-encoder reranker re-orders the top fused candidates by reading each
(query, document) pair together — more accurate than the embedding model's
independent scores, at a measured ~9 ms per pair (~0.7 s for the 80-pair pool
a default search reranks; a reranked query lands at ~1 s all-in vs ~0.25 s
without). Off by default.

```sh
wom tax forms --rerank                          # one-off, with the default
wom tax forms --rerank-model jina-reranker-v2-base-multilingual-int8
wom tax forms --no-rerank                       # override the config for one query
wom bench                                       # reports reranker load + pair cost
```

Or persist it in `config.toml`:

```toml
reranker = "ms-marco-MiniLM-L-6-v2-int8"   # or "off"
```

Two are registered (`wom model list` shows them):

| Reranker | Size | Languages | Choose when |
|---|---:|---|---|
| `ms-marco-MiniLM-L-6-v2-int8` (default) | 22 MB | English | Everything English — tiny and fast |
| `jina-reranker-v2-base-multilingual-int8` | 267 MB | Multilingual | You search content in languages other than English |

(`bge-reranker-v2-m3` is deliberately absent: nobody publishes a single-file
ONNX export of it, and the downloader fetches exactly two files per model.)

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
geo = false             # needs media = true; index photo GPS as place names
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
