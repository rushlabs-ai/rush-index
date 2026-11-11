# Rush Index

**One local search over your documents and photos.** Type what you are looking for and get the
matching text files and photos together, ranked by meaning, without uploading anything.

```text
$ rush-index search "receipt from the hardware store"       # illustrative output
 1. /home/me/Documents/expenses-2026.md:14
    March: hardware store, screws and a drill bit, receipt scanned...
 2. [photo] /home/me/Pictures/2026-03-02-receipt.jpg  (match 87%)
```

- **Documents**: text files, PDFs and Word (.docx) files. [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
  for meaning, plus BM25 (tantivy) for exact words, fused with reciprocal rank fusion.
- **Photos**: [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224) matches your words to the
  pictures themselves. A photo appears only when SigLIP2's calibrated match probability clears a
  threshold, so a science question does not return holiday snaps.
- **One binary** (Rust), models run on ONNX Runtime: CPU everywhere, NVIDIA GPU with `--gpu`.
- **Private**: nothing leaves your machine after the one-time model download.

## Install

Download a build for your platform from the [releases](https://github.com/rushlabs-ai/rush-index/releases)
(Linux x86_64 and ARM64, macOS Apple silicon, Windows x86_64), check it against `SHA256SUMS`, and
optionally verify its build provenance with `gh attestation verify <file> --repo rushlabs-ai/rush-index`.
Or build from source (Rust 1.88+): `cargo install --locked --git https://github.com/rushlabs-ai/rush-index`.

Then download the models and ONNX Runtime once:

```sh
rush-index fetch          # CPU: about 3.9 GB (full-precision models)
rush-index fetch --gpu    # NVIDIA: about 2.4 GB (fp16 models, CUDA 12 runtime)
```

Every file is pinned by URL and SHA-256 in [src/models.rs](src/models.rs) and checked after
download. `fetch` uses your system's `curl`, so the binary carries no TLS stack of its own.

## Use

```sh
rush-index index ~/Documents ~/Pictures    # re-run any time: unchanged files are skipped
rush-index search "how do I reset the router"
rush-index search "dog on a beach" --photos
rush-index similar ~/Pictures/IMG_2041.jpg
rush-index search --batch queries.tsv --json   # many queries, models loaded once
```

Documents are split into overlapping chunks of about 1,200 characters:

- **Text files**: Markdown, plain text, code, CSV, HTML and more.
- **Word (.docx)**: read by Rush Index itself (body text; size-limited, no macros or embedded objects).
- **PDF**: read with Poppler's `pdftotext` (Linux: `poppler-utils`; macOS: `brew install poppler`;
  Windows: a Poppler build on your `PATH`). Results show the page number. Without `pdftotext`, PDFs
  are skipped and picked up on a later run once it is installed. Scanned PDFs without a text layer
  are not read (no OCR). `pdftotext` runs as a separate process with a 60-second limit per file;
  keep Poppler updated through your system's packages, since it parses every PDF you index.

Photos: JPEG, PNG and WebP. Hidden folders, `node_modules`, `target` and similar are skipped, and
symlinks are never followed.

## How good is it?

`tools/bench/modal_bench.py` runs a local release build (`cargo build --release`) on a 4-CPU
container and on an NVIDIA L4, on public data: **BEIR SciFact** (5,183 scientific abstracts, 300
test questions with graded relevance) and **500 COCO photos** searched by their human-written
captions.

Latest run, on Rush Index 0.1.0 built from source (`tools/bench/results-*.json`):

| | 4-CPU container | NVIDIA L4 |
|---|---|---|
| Index 5,183 documents | 2.9 files/s (30 min) | **48 files/s** (1.8 min) |
| Index 500 photos | 15 photos/s | 70 photos/s (incl. model load) |
| Re-index with nothing changed | 0.7 s | 0.7 s |
| Search latency, hybrid, models loaded (p50) | 107 ms | 44 ms |

| Quality | CPU (fp32) | GPU (fp16) |
|---|---|---|
| SciFact nDCG@10, BM25 only | 0.640 | 0.640 |
| SciFact nDCG@10, dense only | 0.684 | 0.684 |
| SciFact nDCG@10, **hybrid (default)** | **0.694** | **0.695** |
| COCO: photo found from its caption, recall@1 / @5 / @10 (500 photos) | 73% / 93% / 99% | 73% / 93% / 99% |
| Photos appearing in the top 10 of science questions | 0.01 per query | 0.01 per query |
| Caption queries with the right photo in the top 10, documents and photos mixed | 82% | 82% |

Hybrid search beats either ranking alone, and the photo threshold keeps photos out of
document-style questions while still surfacing them for photo-style ones. CPU and GPU give the
same quality; the GPU is about 16x faster at indexing text.

## Accuracy of the model files

`tools/verify/verify_onnx.py` compares the ONNX files with the original PyTorch models.

| Model | ONNX file used | Cosine vs original | int8 file (not used) |
|---|---|---|---|
| Qwen3-Embedding-0.6B | fp32 (CPU), fp16 (GPU) | 1.000 | 0.87–0.91 |
| SigLIP2 base/224, image | fp32 / fp16 | 1.000 | 0.66–0.89 |
| SigLIP2 base/224, text | fp32 / fp16 | 1.000 | 0.97 |

The int8 files would halve the download but measurably change the results, so they are not used.
Preprocessing in Rust matches the original pipeline exactly (pixel values identical; last-token
pooling with left padding for Qwen3).

## GPU

`--gpu` loads ONNX Runtime's CUDA 12 build and the fp16 models. It needs an NVIDIA driver plus
the CUDA 12 runtime and cuDNN 9 libraries on the machine (the "single binary" promise holds on
CPU). If CUDA cannot be initialised, ONNX Runtime falls back to the CPU.

## Supply chain

- Exact versions in `Cargo.toml`, `Cargo.lock` committed, builds with `--locked`.
- `tools/check-crate-ages.py` fails CI if any locked crate is less than 7 days old or yanked.
- `cargo deny` checks advisories (RustSec), licences (permissive only), and sources.
- `ort` uses `load-dynamic`: no binaries are downloaded at build time; the runtime archive is
  pinned by SHA-256 like the models.

## Licences

Rush Index is Apache-2.0. Models: Qwen3-Embedding-0.6B (Apache-2.0), SigLIP2 (Apache-2.0), ONNX
exports by onnx-community. ONNX Runtime: MIT. Evaluation data: SciFact (CC BY-SA 4.0), COCO
(CC BY 4.0).

Built by [Rush Labs](https://rushlabs.ai), an applied AI lab.
