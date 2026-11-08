//! The on-disk index: chunk and photo records (JSON lines), their vectors (raw f16), file
//! states for incremental updates, and a tantivy BM25 index over the chunk text.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use half::f16;
use serde::{Deserialize, Serialize};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{FAST, INDEXED, STORED, Schema, TEXT, Value};
use tantivy::{Index as Tantivy, TantivyDocument, doc};

pub const VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Clone)]
pub struct Chunk {
    pub path: String,
    pub line: u32,
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Photo {
    pub path: String,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub struct FileState {
    pub mtime: u64,
    pub size: u64,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Meta {
    pub version: u32,
    pub roots: Vec<String>,
    pub precision: String,
    pub files: BTreeMap<String, FileState>,
}

#[derive(Default)]
pub struct Store {
    pub meta: Meta,
    pub chunks: Vec<Chunk>,
    pub text_vecs: Vec<f16>,
    pub photos: Vec<Photo>,
    pub image_vecs: Vec<f16>,
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    BufReader::new(fs::File::open(path)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect()
}

fn write_jsonl<T: Serialize>(path: &Path, items: &[T]) -> Result<()> {
    let mut w = BufWriter::new(fs::File::create(path)?);
    for it in items {
        serde_json::to_writer(&mut w, it)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

fn read_f16(path: &Path) -> Result<Vec<f16>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = fs::read(path)?;
    let (pairs, _) = bytes.as_chunks::<2>();
    Ok(pairs.iter().map(|b| f16::from_le_bytes(*b)).collect())
}

fn write_f16(path: &Path, v: &[f16]) -> Result<()> {
    let mut w = BufWriter::new(fs::File::create(path)?);
    for x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

impl Store {
    pub fn load(dir: &Path) -> Result<Self> {
        let meta_path = dir.join("meta.json");
        if !meta_path.exists() {
            return Ok(Self::default());
        }
        let meta: Meta = serde_json::from_slice(&fs::read(&meta_path)?).context("reading meta.json")?;
        if meta.version != VERSION {
            return Err(anyhow!("index format {} is not supported by this version; re-index", meta.version));
        }
        Ok(Self {
            meta,
            chunks: read_jsonl(&dir.join("chunks.jsonl"))?,
            text_vecs: read_f16(&dir.join("text.f16"))?,
            photos: read_jsonl(&dir.join("photos.jsonl"))?,
            image_vecs: read_f16(&dir.join("photos.f16"))?,
        })
    }

    /// Write everything to a temporary folder, then swap it in, so a crash never leaves a half index.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let tmp = dir.with_extension("new");
        if tmp.exists() {
            fs::remove_dir_all(&tmp)?;
        }
        fs::create_dir_all(&tmp)?;
        fs::write(tmp.join("meta.json"), serde_json::to_vec_pretty(&self.meta)?)?;
        write_jsonl(&tmp.join("chunks.jsonl"), &self.chunks)?;
        write_f16(&tmp.join("text.f16"), &self.text_vecs)?;
        write_jsonl(&tmp.join("photos.jsonl"), &self.photos)?;
        write_f16(&tmp.join("photos.f16"), &self.image_vecs)?;
        build_bm25(&tmp.join("bm25"), &self.chunks)?;
        let old = dir.with_extension("old");
        if dir.exists() {
            if old.exists() {
                fs::remove_dir_all(&old)?;
            }
            fs::rename(dir, &old)?;
        }
        fs::rename(&tmp, dir)?;
        if old.exists() {
            fs::remove_dir_all(&old)?;
        }
        Ok(())
    }
}

fn schema() -> (Schema, tantivy::schema::Field, tantivy::schema::Field) {
    let mut b = Schema::builder();
    let id = b.add_u64_field("id", INDEXED | STORED | FAST);
    let body = b.add_text_field("body", TEXT);
    (b.build(), id, body)
}

fn build_bm25(dir: &Path, chunks: &[Chunk]) -> Result<()> {
    fs::create_dir_all(dir)?;
    let (schema, id, body) = schema();
    let index = Tantivy::create_in_dir(dir, schema)?;
    let mut w = index.writer(50_000_000)?;
    for (i, c) in chunks.iter().enumerate() {
        // The file name is searchable too: people often remember names.
        let name = Path::new(&c.path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        w.add_document(doc!(id => i as u64, body => format!("{name}\n{}", c.text)))?;
    }
    w.commit()?;
    Ok(())
}

/// BM25 hits as (chunk index, score), best first.
pub fn bm25_search(dir: &Path, query: &str, k: usize) -> Result<Vec<(usize, f32)>> {
    let path: PathBuf = dir.join("bm25");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let index = Tantivy::open_in_dir(&path)?;
    let (_, id, body) = schema();
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let parser = QueryParser::for_index(&index, vec![body]);
    let (q, _errors) = parser.parse_query_lenient(query); // never fail on quotes, colons and the like
    let hits = searcher.search(&q, &TopDocs::with_limit(k).order_by_score())?;
    let mut out = Vec::with_capacity(hits.len());
    for (score, addr) in hits {
        let d: TantivyDocument = searcher.doc(addr)?;
        if let Some(v) = d.get_first(id).and_then(|v| v.as_u64()) {
            out.push((v as usize, score));
        }
    }
    Ok(out)
}

/// Dot products against normalised f16 vectors, best first.
pub fn dense_search(vecs: &[f16], dim: usize, query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = vecs
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, v)| (i, v.iter().zip(query).map(|(a, b)| a.to_f32() * b).sum()))
        .collect();
    let k = k.min(scored.len());
    if k == 0 {
        return Vec::new();
    }
    scored.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}
