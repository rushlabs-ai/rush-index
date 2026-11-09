//! Finding files and cutting text into chunks.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::store::FileState;

pub const TEXT_EXT: &[&str] = &[
    "txt", "md", "markdown", "rst", "org", "tex", "csv", "tsv", "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg",
    "html", "htm", "xml", "log", "py", "rs", "go", "js", "ts", "tsx", "jsx", "java", "kt", "c", "h", "cc", "cpp", "hpp",
    "cs", "rb", "php", "swift", "sh", "sql", "r", "lua", "scala", "dart",
];
/// Documents read through a converter (see docs.rs). PDF needs Poppler's `pdftotext`.
pub const DOC_EXT: &[&str] = &["pdf", "docx"];
pub const IMAGE_EXT: &[&str] = &["jpg", "jpeg", "png", "webp"];
const SKIP_DIRS: &[&str] = &[".git", "node_modules", "target", ".venv", "venv", "__pycache__", ".cache", "dist", "build"];
const MAX_TEXT_BYTES: u64 = 8 << 20;
const MAX_DOC_BYTES: u64 = 64 << 20;
const MAX_IMAGE_BYTES: u64 = 40 << 20;
const CHUNK_CHARS: usize = 1200;
const OVERLAP_LINES: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Text,
    Image,
}

pub struct Found {
    pub path: PathBuf,
    pub kind: Kind,
    pub state: FileState,
}

pub fn ext_of(p: &Path) -> String {
    p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default()
}

pub fn walk(root: &Path) -> Vec<Found> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let path = e.path();
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_symlink() {
                continue; // never follow links out of the folder being indexed
            }
            let name = e.file_name().to_string_lossy().into_owned();
            if ft.is_dir() {
                if !name.starts_with('.') && !SKIP_DIRS.contains(&name.as_str()) {
                    stack.push(path);
                }
                continue;
            }
            let ext = ext_of(&path);
            let kind = if TEXT_EXT.contains(&ext.as_str()) || DOC_EXT.contains(&ext.as_str()) {
                Kind::Text
            } else if IMAGE_EXT.contains(&ext.as_str()) {
                Kind::Image
            } else {
                continue;
            };
            let Ok(md) = e.metadata() else { continue };
            let limit = match kind {
                Kind::Image => MAX_IMAGE_BYTES,
                Kind::Text if DOC_EXT.contains(&ext.as_str()) => MAX_DOC_BYTES,
                Kind::Text => MAX_TEXT_BYTES,
            };
            if md.len() == 0 || md.len() > limit {
                continue;
            }
            let mtime = md.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
            out.push(Found { path, kind, state: FileState { mtime, size: md.len() } });
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Line-based chunks of about CHUNK_CHARS characters, overlapping by a couple of lines.
/// Returns (first line number, text).
pub fn chunk(text: &str) -> Vec<(u32, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        let mut end = start;
        let mut size = 0;
        while end < lines.len() && (size == 0 || size + lines[end].len() < CHUNK_CHARS) {
            size += lines[end].len() + 1;
            end += 1;
        }
        let body = lines[start..end].join("\n");
        if !body.trim().is_empty() {
            // A very long single line becomes one chunk; the embedder truncates it.
            out.push((start as u32 + 1, body));
        }
        if end >= lines.len() {
            break;
        }
        start = end.saturating_sub(OVERLAP_LINES).max(start + 1);
    }
    out
}

/// Chunks of any indexable text file. For PDFs the number is the page; otherwise the first line.
pub fn read_chunks(path: &Path) -> Vec<(u32, String)> {
    match ext_of(path).as_str() {
        "pdf" => crate::docs::pdf_pages(path)
            .unwrap_or_default()
            .iter()
            .enumerate()
            .flat_map(|(page, text)| chunk(text).into_iter().map(move |(_, body)| (page as u32 + 1, body)))
            .collect(),
        "docx" => fs::read(path).ok().and_then(|b| crate::docs::docx_text(&b)).map(|t| chunk(&t)).unwrap_or_default(),
        _ => read_text(path).map(|t| chunk(&t)).unwrap_or_default(),
    }
}

pub fn is_pdf(path: &Path) -> bool {
    ext_of(path) == "pdf"
}

pub fn read_text(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    if bytes.iter().take(4096).any(|&b| b == 0) {
        return None; // binary
    }
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_text_with_overlap() {
        let text: String = (0..200).map(|i| format!("line number {i} with some words\n")).collect();
        let chunks = chunk(&text);
        assert!(chunks.len() > 3);
        assert_eq!(chunks[0].0, 1);
        for w in chunks.windows(2) {
            assert!(w[1].0 > w[0].0, "chunks move forward");
            assert!(w[0].1.len() <= CHUNK_CHARS + 64);
        }
        assert!(chunks.last().unwrap().1.contains("line number 199"));
    }

    #[test]
    fn long_single_line_is_one_chunk() {
        let text = "x".repeat(5000);
        assert_eq!(chunk(&text).len(), 1);
    }

    #[test]
    fn blank_text_has_no_chunks() {
        assert!(chunk("\n\n   \n").is_empty());
    }
}
