"""Benchmark and evaluate the rush-index binary on Modal: a 4-CPU container and an L4 GPU.

    cargo build --release && modal run tools/bench/modal_bench.py            # both
    modal run tools/bench/modal_bench.py --only cpu                          # one side

Corpus, built inside the container:
  - documents: BEIR SciFact (5,183 scientific abstracts, CC BY-SA 4.0) as .txt files; its 300 test
    questions and graded relevance give nDCG@10 for the dense, BM25 and hybrid modes
  - photos: 500 COCO val2017 photos (CC BY 4.0); each photo's first human caption is its query,
    scored by recall@1/5/10
  - mixed: the same index holds both, so we also count how often a photo leaks into a
    document query's top 10 (photos must clear SigLIP2's match probability to appear)

Reports files/s while indexing and query latency. Results go to tools/bench/results-<cpu|gpu>.json.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import modal

HERE = Path(__file__).parent
# Resolved on your machine only; inside the container this file sits at /root and the path is unused.
BIN = HERE.parents[1] / "target" / "release" / "rush-index" if len(HERE.parents) > 1 else Path("/unused")
SCIFACT = ("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
           "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165")
COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
N_PHOTOS = 500

volume = modal.Volume.from_name("rush-index", create_if_missing=True)
app = modal.App("rush-index-bench")
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates")
    .add_local_file(BIN, "/usr/local/bin/rush-index")
)
# ONNX Runtime's CUDA 12 build needs the CUDA 12 runtime and cuDNN 9 on the machine.
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc",
        add_python="3.12")
    .apt_install("curl", "ca-certificates")
    .add_local_file(BIN, "/usr/local/bin/rush-index")
)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kw)


def prepare() -> dict:
    """SciFact as text files and COCO photos, cached in the Volume."""
    root = Path("/vol/corpus")
    docs, photos = root / "docs", root / "photos"
    marker = root / "ready.json"
    if marker.exists():
        return json.loads(marker.read_text())
    root.mkdir(parents=True, exist_ok=True)
    z = root / "scifact.zip"
    sh(["curl", "-fsSL", "-o", str(z), SCIFACT[0]])
    sh(["sh", "-c", f"echo '{SCIFACT[1]}  {z}' | sha256sum -c -"])
    with zipfile.ZipFile(z) as f:
        corpus = [json.loads(x) for x in f.read("scifact/corpus.jsonl").decode().splitlines()]
        queries = {json.loads(x)["_id"]: json.loads(x)["text"] for x in f.read("scifact/queries.jsonl").decode().splitlines()}
        qrels: dict[str, dict[str, int]] = {}
        for line in f.read("scifact/qrels/test.tsv").decode().splitlines()[1:]:
            q, d, s = line.split("\t")
            qrels.setdefault(q, {})[d] = int(s)
    docs.mkdir(exist_ok=True)
    for d in corpus:
        (docs / f"{d['_id']}.txt").write_text(f"{d['title']}\n\n{d['text']}\n")
    ann = root / "ann.zip"
    sh(["curl", "-fsSL", "-o", str(ann), COCO_ANN])
    with zipfile.ZipFile(ann) as f:
        caps = json.loads(f.read("annotations/captions_val2017.json"))
    first_caption: dict[int, str] = {}
    for c in sorted(caps["annotations"], key=lambda c: c["id"]):
        first_caption.setdefault(c["image_id"], c["caption"].strip())
    chosen = sorted(i["id"] for i in caps["images"])[:N_PHOTOS]
    photos.mkdir(exist_ok=True)
    for i in chosen:
        dest = photos / f"{i:012d}.jpg"
        if not dest.exists():
            urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{i:012d}.jpg", dest)
    meta = {"docs": len(corpus), "photos": len(chosen),
            "doc_queries": {q: queries[q] for q in qrels if q in queries}, "qrels": qrels,
            "photo_queries": {f"{i:012d}": first_caption[i] for i in chosen}}
    marker.write_text(json.dumps(meta))
    volume.commit()
    return meta


def ndcg10(ranked: list[str], rel: dict[str, int]) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def batch(args: list[str], queries: dict[str, str], env: dict) -> list[dict]:
    qfile = Path("/tmp/queries.tsv")
    qfile.write_text("".join(f"{k}\t{v.replace(chr(9), ' ').replace(chr(10), ' ')}\n" for k, v in queries.items()))
    out = sh(["rush-index", "search", "--batch", str(qfile), *args], capture_output=True, env=env).stdout
    return [json.loads(x) for x in out.splitlines() if x.strip()]


def run(gpu: bool) -> dict:
    volume.reload()
    meta = prepare()
    models = "/vol/models-gpu" if gpu else "/vol/models-cpu"
    env = dict(os.environ, RUSH_INDEX_MODELS=models)
    flag = ["--gpu"] if gpu else []
    sh(["rush-index", "fetch", *flag], env=env)
    report: dict = {"hardware": "L4 GPU" if gpu else "4 CPU", "corpus": {"docs": meta["docs"], "photos": meta["photos"]}}
    if gpu:
        report["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                              capture_output=True, text=True).stdout.strip()
    # One index: documents first, then photos added on top (documents are reused), each timed.
    db = "/tmp/db"
    t0 = time.time()
    sh(["rush-index", "index", "/vol/corpus/docs", "--db", db, *flag], env=env)
    secs = time.time() - t0
    report["index_docs"] = {"files": meta["docs"], "seconds": round(secs, 1), "files_per_second": round(meta["docs"] / secs, 2)}
    t0 = time.time()
    sh(["rush-index", "index", "/vol/corpus/docs", "/vol/corpus/photos", "--db", db, *flag], env=env)
    secs = time.time() - t0
    report["index_photos"] = {"files": meta["photos"], "seconds": round(secs, 1), "files_per_second": round(meta["photos"] / secs, 2)}
    t0 = time.time()
    sh(["rush-index", "index", "/vol/corpus/docs", "/vol/corpus/photos", "--db", db, *flag], env=env)
    report["reindex_unchanged_seconds"] = round(time.time() - t0, 2)

    # Document retrieval quality per mode.
    for mode in ("bm25", "dense", "hybrid"):
        rows = batch(["--db", db, "--docs", "-k", "10", "--mode", mode, *flag], meta["doc_queries"], env)
        scores = [ndcg10([Path(h["path"]).stem for h in r["results"]], meta["qrels"][r["id"]]) for r in rows]
        lat = sorted(r["ms"] for r in rows)
        report[f"scifact_{mode}"] = {"ndcg@10": round(sum(scores) / len(scores), 4), "queries": len(rows),
                                     "latency_ms_p50": lat[len(lat) // 2]}
    # Photo retrieval by caption.
    rows = batch(["--db", db, "--photos", "-k", "10", "--photo-min", "0", *flag], meta["photo_queries"], env)
    ranks = []
    for r in rows:
        names = [Path(h["path"]).stem for h in r["results"]]
        ranks.append(names.index(r["id"]) + 1 if r["id"] in names else 99)
    report["coco_caption_to_photo"] = {f"recall@{k}": round(sum(x <= k for x in ranks) / len(ranks), 3) for k in (1, 5, 10)}

    # Mixed results: do photos leak into science queries, and do photos surface for captions?
    sci = batch(["--db", db, "-k", "10", *flag], dict(list(meta["doc_queries"].items())[:100]), env)
    cap = batch(["--db", db, "-k", "10", *flag], dict(list(meta["photo_queries"].items())[:100]), env)
    report["mixed"] = {
        "photos_in_top10_for_science_queries": round(sum(sum(h["type"] == "photo" for h in r["results"]) for r in sci) / len(sci), 2),
        "caption_queries_with_target_photo_in_top10": round(sum(any(Path(h["path"]).stem == r["id"] for h in r["results"]) for r in cap) / len(cap), 3),
    }
    return report


@app.function(image=cpu_image, cpu=2, memory=8192, volumes={"/vol": volume}, timeout=3600)
def prepare_corpus() -> dict:
    meta = prepare()
    return {"docs": meta["docs"], "photos": meta["photos"]}


@app.function(image=cpu_image, cpu=4, memory=16384, volumes={"/vol": volume}, timeout=4 * 3600)
def bench_cpu() -> dict:
    return run(gpu=False)


@app.function(image=gpu_image, gpu="L4", cpu=4, memory=16384, volumes={"/vol": volume}, timeout=2 * 3600)
def bench_gpu() -> dict:
    return run(gpu=True)


@app.local_entrypoint()
def main(only: str = "") -> None:
    print("corpus:", prepare_corpus.remote())  # once, before both benchmarks read it
    calls = {}
    if only in ("", "gpu"):
        calls["gpu"] = bench_gpu.spawn()
    if only in ("", "cpu"):
        calls["cpu"] = bench_cpu.spawn()
    for name, call in calls.items():
        res = call.get()
        (HERE / f"results-{name}.json").write_text(json.dumps(res, indent=2))
        print(name, json.dumps(res, indent=2))
