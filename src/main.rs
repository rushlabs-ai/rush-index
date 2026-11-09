//! rush-index: one local search over your documents and photos.

mod docs;
mod embed;
mod files;
mod models;
mod search;
mod store;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use embed::{Accel, IMAGE_DIM, ImageEmbedder, TEXT_DIM, TextEmbedder};
use files::Kind;
use store::{Chunk, Meta, Photo, Store};

const USAGE: &str = "rush-index: search your documents and photos by meaning, locally.

Usage:
  rush-index fetch [--gpu]                 download the models (about 3.9 GB; 2.4 GB with --gpu)
  rush-index index <folder>... [--gpu]     index documents and photos (re-run to update)
  rush-index search <query> [-k N]         search documents and photos together
  rush-index similar <file> [-k N]         find documents or photos like this one
  rush-index status                        what is indexed

Options:
  --db <dir>          index location (default: $RUSH_INDEX_DB or ~/.local/share/rush-index/default)
  --models <dir>      model location (default: $RUSH_INDEX_MODELS or ~/.cache/rush-index/models)
  --gpu               use an NVIDIA GPU (CUDA 12) with the fp16 models
  --trt               use NVIDIA TensorRT on the GPU (implies --gpu; needs TensorRT 10). Engines are
                      built on first use and cached ($RUSH_INDEX_TRT_CACHE or <models>/trt-cache)
  --threads <n>       CPU threads for inference (default: all)
  -k <n>              number of results (default 10)
  --docs | --photos   only documents, or only photos
  --photo-min <p>     minimum photo match probability, 0-1 (default 0.05)
  --json              machine-readable output
  --mode <m>          hybrid (default), dense or bm25: which document rankings to fuse
  --batch <file>      search: run one query per line (\"id<TAB>query\" or just the query), JSON lines out
";

#[derive(Default)]
struct Args {
    cmd: String,
    positional: Vec<String>,
    db: Option<PathBuf>,
    models: Option<PathBuf>,
    gpu: bool,
    trt: bool,
    threads: usize,
    k: usize,
    docs_only: bool,
    photos_only: bool,
    photo_min: f32,
    json: bool,
    mode: String,
    batch: Option<PathBuf>,
}

fn parse() -> Result<Args> {
    use lexopt::prelude::*;
    let mut a = Args { k: 10, photo_min: 0.05, mode: "hybrid".into(), ..Default::default() };
    let mut p = lexopt::Parser::from_env();
    while let Some(arg) = p.next()? {
        match arg {
            Long("db") => a.db = Some(p.value()?.into()),
            Long("models") => a.models = Some(p.value()?.into()),
            Long("gpu") => a.gpu = true,
            Long("trt") => {
                a.trt = true;
                a.gpu = true;
            }
            Long("threads") => a.threads = p.value()?.parse()?,
            Short('k') => a.k = p.value()?.parse()?,
            Long("docs") => a.docs_only = true,
            Long("photos") => a.photos_only = true,
            Long("photo-min") => a.photo_min = p.value()?.parse()?,
            Long("json") => a.json = true,
            Long("mode") => a.mode = p.value()?.string()?,
            Long("batch") => a.batch = Some(p.value()?.into()),
            Long("version") => {
                println!("rush-index {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            Short('h') | Long("help") => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            Value(v) if a.cmd.is_empty() => a.cmd = v.string()?,
            Value(v) => a.positional.push(v.string()?),
            _ => return Err(arg.unexpected().into()),
        }
    }
    Ok(a)
}

fn home() -> PathBuf {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(|| ".".into())
}

fn db_dir(a: &Args) -> PathBuf {
    a.db.clone()
        .or_else(|| std::env::var_os("RUSH_INDEX_DB").map(PathBuf::from))
        .unwrap_or_else(|| home().join(".local/share/rush-index/default"))
}

fn models_dir(a: &Args) -> PathBuf {
    a.models.clone()
        .or_else(|| std::env::var_os("RUSH_INDEX_MODELS").map(PathBuf::from))
        .unwrap_or_else(|| home().join(".cache/rush-index/models"))
}

fn threads(a: &Args) -> usize {
    if a.threads > 0 { a.threads } else { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("rush-index: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let a = parse()?;
    match a.cmd.as_str() {
        "fetch" => {
            let lib = models::fetch(&models_dir(&a), a.gpu)?;
            eprintln!("Ready. ONNX Runtime: {}", lib.display());
            Ok(())
        }
        "index" => index(&a),
        "search" => search_cmd(&a),
        "similar" => similar(&a),
        "status" => status(&a),
        "" => {
            print!("{USAGE}");
            Ok(())
        }
        other => bail!("unknown command {other:?}; see rush-index --help"),
    }
}

struct Engines {
    env: ort::environment::Environment,
    models: PathBuf,
    accel: Accel,
    threads: usize,
}

impl Engines {
    fn new(a: &Args) -> Result<Self> {
        let dir = models_dir(a);
        models::require(&dir, a.gpu)?;
        let lib = models::runtime_library(&dir, a.gpu)?;
        let accel = if a.trt {
            Accel::TensorRt(std::env::var_os("RUSH_INDEX_TRT_CACHE").map(PathBuf::from).unwrap_or_else(|| dir.join("trt-cache")))
        } else if a.gpu {
            Accel::Cuda
        } else {
            Accel::Cpu
        };
        Ok(Self { env: embed::environment(&lib)?, models: dir, accel, threads: threads(a) })
    }
    fn text(&self) -> Result<TextEmbedder> {
        TextEmbedder::load(&self.env, &self.models, &self.accel, self.threads)
    }
    fn image(&self) -> Result<ImageEmbedder> {
        ImageEmbedder::load(&self.env, &self.models, &self.accel, self.threads)
    }
}

fn index(a: &Args) -> Result<()> {
    if a.positional.is_empty() {
        bail!("usage: rush-index index <folder>...");
    }
    let roots: Vec<PathBuf> = a
        .positional
        .iter()
        .map(|r| std::fs::canonicalize(r).with_context(|| format!("folder {r}")))
        .collect::<Result<_>>()?;
    let db = db_dir(a);
    let old = Store::load(&db)?;
    let started = Instant::now();
    let mut found = Vec::new();
    for r in &roots {
        found.extend(files::walk(r));
    }
    if !docs::pdftotext_available() {
        let before = found.len();
        found.retain(|f| !files::is_pdf(&f.path));
        if found.len() < before {
            eprintln!(
                "Skipping {} PDF files: install Poppler's pdftotext to index them (Linux: poppler-utils, macOS: brew install poppler)",
                before - found.len()
            );
        }
    }
    let mut new = Store {
        meta: Meta {
            version: store::VERSION,
            roots: roots.iter().map(|r| r.display().to_string()).collect(),
            precision: if a.gpu { "fp16" } else { "fp32" }.to_string(),
            files: Default::default(),
        },
        ..Default::default()
    };

    // Reuse the vectors of files that have not changed since the last run.
    let mut old_chunks: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, c) in old.chunks.iter().enumerate() {
        old_chunks.entry(c.path.as_str()).or_default().push(i);
    }
    let old_photos: HashMap<&str, usize> = old.photos.iter().enumerate().map(|(i, p)| (p.path.as_str(), i)).collect();
    let (mut todo_text, mut todo_img) = (Vec::new(), Vec::new());
    for f in &found {
        let key = f.path.display().to_string();
        let unchanged = old.meta.files.get(&key) == Some(&f.state);
        match (f.kind, unchanged) {
            (Kind::Text, true) => {
                for &i in old_chunks.get(key.as_str()).map(|v| v.as_slice()).unwrap_or(&[]) {
                    new.chunks.push(old.chunks[i].clone());
                    new.text_vecs.extend_from_slice(&old.text_vecs[i * TEXT_DIM..(i + 1) * TEXT_DIM]);
                }
                new.meta.files.insert(key, f.state);
            }
            (Kind::Image, true) => {
                if let Some(&i) = old_photos.get(key.as_str()) {
                    new.photos.push(old.photos[i].clone());
                    new.image_vecs.extend_from_slice(&old.image_vecs[i * IMAGE_DIM..(i + 1) * IMAGE_DIM]);
                }
                new.meta.files.insert(key, f.state);
            }
            (Kind::Text, false) => todo_text.push(f),
            (Kind::Image, false) => todo_img.push(f),
        }
    }
    eprintln!(
        "{} files found: {} unchanged, {} text files and {} photos to index",
        found.len(),
        new.meta.files.len(),
        todo_text.len(),
        todo_img.len()
    );
    let engines = if todo_text.is_empty() && todo_img.is_empty() { None } else { Some(Engines::new(a)?) };

    if let (Some(eng), false) = (&engines, todo_text.is_empty()) {
        let mut te = eng.text()?;
        let t0 = Instant::now();
        let mut pending: Vec<Chunk> = Vec::new();
        let total = todo_text.len();
        for (n, f) in todo_text.iter().enumerate() {
            let key = f.path.display().to_string();
            for (line, body) in files::read_chunks(&f.path) {
                pending.push(Chunk { path: key.clone(), line, text: body });
            }
            new.meta.files.insert(key, f.state);
            if pending.len() >= 256 || n + 1 == total {
                let texts: Vec<String> = pending.iter().map(|c| c.text.clone()).collect();
                for v in te.embed(&texts, false)? {
                    new.text_vecs.extend(embed::to_f16(&v));
                }
                new.chunks.append(&mut pending);
                eprintln!("  text: {}/{} files, {:.1} files/s", n + 1, total, (n + 1) as f64 / t0.elapsed().as_secs_f64());
            }
        }
    }

    if let (Some(eng), false) = (&engines, todo_img.is_empty()) {
        let mut ie = eng.image()?;
        let t0 = Instant::now();
        let workers = threads(a).clamp(1, 8);
        let step = ie.batch * 4;
        for (b, group) in todo_img.chunks(step).enumerate() {
            // Decode and resize on several threads; the model call batches them.
            let decoded: Vec<Option<Vec<f32>>> = std::thread::scope(|s| {
                let handles: Vec<_> = group
                    .chunks(group.len().div_ceil(workers).max(1))
                    .map(|part| s.spawn(move || part.iter().map(|f| ImageEmbedder::pixels(&f.path).ok()).collect::<Vec<_>>()))
                    .collect();
                handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
            });
            let ok: Vec<(usize, Vec<f32>)> = decoded.into_iter().enumerate().filter_map(|(i, d)| d.map(|d| (i, d))).collect();
            let pixels: Vec<Vec<f32>> = ok.iter().map(|(_, d)| d.clone()).collect();
            let vecs = if pixels.is_empty() { Vec::new() } else { ie.embed_images(&pixels)? };
            for ((i, _), v) in ok.iter().zip(vecs) {
                new.photos.push(Photo { path: group[*i].path.display().to_string() });
                new.image_vecs.extend(embed::to_f16(&v));
            }
            for f in group {
                new.meta.files.insert(f.path.display().to_string(), f.state);
            }
            let done = ((b + 1) * step).min(todo_img.len());
            eprintln!("  photos: {}/{}, {:.1} photos/s", done, todo_img.len(), done as f64 / t0.elapsed().as_secs_f64());
        }
    }

    new.save(&db)?;
    eprintln!(
        "Indexed {} text chunks and {} photos in {:.1}s -> {}",
        new.chunks.len(),
        new.photos.len(),
        started.elapsed().as_secs_f64(),
        db.display()
    );
    Ok(())
}

struct Hit {
    kind: &'static str,
    path: String,
    line: Option<u32>,
    snippet: String,
    score: f32,
    probability: Option<f32>,
}

fn snippet(text: &str) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > 160 { format!("{}…", flat.chars().take(160).collect::<String>()) } else { flat }
}

struct Searcher<'a> {
    a: &'a Args,
    db: PathBuf,
    store: Store,
    text: Option<TextEmbedder>,
    image: Option<ImageEmbedder>,
}

impl<'a> Searcher<'a> {
    fn new(a: &'a Args) -> Result<Self> {
        let db = db_dir(a);
        let store = Store::load(&db)?;
        if store.chunks.is_empty() && store.photos.is_empty() {
            bail!("nothing indexed yet; run rush-index index <folder>");
        }
        if !["hybrid", "dense", "bm25"].contains(&a.mode.as_str()) {
            bail!("--mode must be hybrid, dense or bm25");
        }
        let eng = Engines::new(a)?;
        let need_text = !store.chunks.is_empty() && !a.photos_only && a.mode != "bm25";
        let text = if need_text { Some(eng.text()?) } else { None };
        let image = if !store.photos.is_empty() && !a.docs_only { Some(eng.image()?) } else { None };
        Ok(Self { a, db, store, text, image })
    }

    fn search(&mut self, query: &str) -> Result<Vec<Hit>> {
        let (a, store) = (self.a, &self.store);
        let mut doc_hits: Vec<Hit> = Vec::new();
        if !store.chunks.is_empty() && !a.photos_only {
            let mut lists = Vec::new();
            if let Some(te) = self.text.as_mut() {
                let q = te.embed(&[query.to_string()], true)?.remove(0);
                lists.push(store::dense_search(&store.text_vecs, TEXT_DIM, &q, 100).into_iter().map(|x| x.0).collect());
            }
            if a.mode != "dense" {
                lists.push(store::bm25_search(&self.db, query, 100)?.into_iter().map(|x| x.0).collect());
            }
            let mut seen = HashSet::new();
            for (i, s) in search::rrf(&lists) {
                let c = &store.chunks[i];
                if seen.insert(c.path.clone()) {
                    doc_hits.push(Hit { kind: "doc", path: c.path.clone(), line: Some(c.line), snippet: snippet(&c.text), score: s, probability: None });
                }
            }
        }
        let mut photo_hits: Vec<Hit> = Vec::new();
        if let Some(ie) = self.image.as_mut() {
            let q = ie.embed_text(query)?;
            for (i, cos) in store::dense_search(&store.image_vecs, IMAGE_DIM, &q, 100) {
                let p = search::photo_probability(cos);
                if p >= a.photo_min {
                    photo_hits.push(Hit { kind: "photo", path: store.photos[i].path.clone(), line: None, snippet: String::new(), score: cos, probability: Some(p) });
                }
            }
        }
        // One list: documents and the photos that match well enough, fused by rank.
        let keys = |hits: &[Hit]| hits.iter().map(|h| (h.kind, h.path.clone())).collect::<Vec<_>>();
        let fused = search::rrf(&[keys(&doc_hits), keys(&photo_hits)]);
        let mut by_key: HashMap<(&str, String), Hit> = doc_hits.into_iter().chain(photo_hits).map(|h| ((h.kind, h.path.clone()), h)).collect();
        Ok(fused
            .into_iter()
            .take(a.k)
            .filter_map(|(k, s)| by_key.remove(&k).map(|mut h| {
                h.score = s;
                h
            }))
            .collect())
    }
}

fn hit_json(h: &Hit) -> serde_json::Value {
    serde_json::json!({
        "type": h.kind, "path": h.path, "line": h.line, "snippet": h.snippet,
        "page": if files::is_pdf(std::path::Path::new(&h.path)) { h.line } else { None },
        "score": h.score, "photo_probability": h.probability,
    })
}

fn search_cmd(a: &Args) -> Result<()> {
    if let Some(file) = &a.batch {
        let mut s = Searcher::new(a)?;
        let text = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let (id, q) = line.split_once('\t').map(|(i, q)| (i.to_string(), q)).unwrap_or((n.to_string(), line));
            let t0 = Instant::now();
            let hits = s.search(q)?;
            let rows: Vec<serde_json::Value> = hits.iter().map(hit_json).collect();
            println!("{}", serde_json::json!({ "id": id, "query": q, "ms": t0.elapsed().as_millis() as u64, "results": rows }));
        }
        return Ok(());
    }
    let query = a.positional.join(" ");
    if query.trim().is_empty() {
        bail!("usage: rush-index search <query>");
    }
    let mut s = Searcher::new(a)?;
    let results = s.search(&query)?;
    print_hits(&query, &results, a.json)
}

fn print_hits(query: &str, hits: &[Hit], json: bool) -> Result<()> {
    if json {
        let rows: Vec<serde_json::Value> = hits.iter().map(hit_json).collect();
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "query": query, "results": rows }))?);
        return Ok(());
    }
    if hits.is_empty() {
        println!("No results.");
    }
    for (i, h) in hits.iter().enumerate() {
        match h.kind {
            "photo" => match h.probability {
                Some(p) => println!("{:>2}. [photo] {}  (match {:.0}%)", i + 1, h.path, p * 100.0),
                None => println!("{:>2}. [photo] {}", i + 1, h.path),
            },
            _ if files::is_pdf(std::path::Path::new(&h.path)) => {
                println!("{:>2}. {} (page {})\n    {}", i + 1, h.path, h.line.unwrap_or(1), h.snippet)
            }
            _ => println!("{:>2}. {}:{}\n    {}", i + 1, h.path, h.line.unwrap_or(1), h.snippet),
        }
    }
    Ok(())
}

fn similar(a: &Args) -> Result<()> {
    let file = a.positional.first().context("usage: rush-index similar <file>")?;
    let path = std::fs::canonicalize(file).with_context(|| format!("file {file}"))?;
    let key = path.display().to_string();
    let store = Store::load(&db_dir(a))?;
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let mut hits = Vec::new();
    if files::IMAGE_EXT.contains(&ext.as_str()) {
        let eng = Engines::new(a)?;
        let v = eng.image()?.embed_images(&[ImageEmbedder::pixels(&path)?])?.remove(0);
        for (i, cos) in store::dense_search(&store.image_vecs, IMAGE_DIM, &v, a.k + 1) {
            if store.photos[i].path != key {
                hits.push(Hit { kind: "photo", path: store.photos[i].path.clone(), line: None, snippet: String::new(), score: cos, probability: None });
            }
        }
    } else {
        let mine: Vec<usize> = store.chunks.iter().enumerate().filter(|(_, c)| c.path == key).map(|(i, _)| i).collect();
        if mine.is_empty() {
            bail!("{} is not in the index; index its folder first", path.display());
        }
        let mut centroid = vec![0f32; TEXT_DIM];
        for &i in &mine {
            for (c, x) in centroid.iter_mut().zip(&store.text_vecs[i * TEXT_DIM..(i + 1) * TEXT_DIM]) {
                *c += x.to_f32();
            }
        }
        let mut seen = HashSet::from([key.clone()]);
        for (i, s) in store::dense_search(&store.text_vecs, TEXT_DIM, &centroid, 200) {
            let c = &store.chunks[i];
            if seen.insert(c.path.clone()) {
                hits.push(Hit { kind: "doc", path: c.path.clone(), line: Some(c.line), snippet: snippet(&c.text), score: s, probability: None });
            }
        }
    }
    hits.truncate(a.k);
    print_hits(&format!("similar to {}", path.display()), &hits, a.json)
}

fn status(a: &Args) -> Result<()> {
    let db = db_dir(a);
    let s = Store::load(&db)?;
    let models = models_dir(a);
    let ready = if models::require(&models, false).is_ok() {
        "CPU models ready"
    } else if models::require(&models, true).is_ok() {
        "GPU models ready"
    } else {
        "not downloaded: run rush-index fetch"
    };
    println!(
        "index:   {}\nroots:   {}\nfiles:   {}\nchunks:  {}\nphotos:  {}\nmodels:  {} ({ready})",
        db.display(),
        s.meta.roots.join(", "),
        s.meta.files.len(),
        s.chunks.len(),
        s.photos.len(),
        models.display()
    );
    Ok(())
}
