//! Text from PDF and Word (.docx) files.
//!
//! - PDF: Poppler's `pdftotext`, run as a separate process with a time limit and an output cap,
//!   so a malformed PDF cannot crash or hang the indexer. Pages come back separated by form feeds.
//! - DOCX: a .docx file is a zip archive; `word/document.xml` holds the body text. A small zip
//!   reader (bounds-checked, decompressed size capped) and a tag scanner extract it, with no
//!   extra dependencies.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Largest decompressed `word/document.xml` we accept, and largest `pdftotext` output.
const MAX_TEXT: u64 = 32 << 20;
const PDF_TIMEOUT: Duration = Duration::from_secs(60);

/// Whether Poppler's `pdftotext` can be started. Checked once.
pub fn pdftotext_available() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        Command::new("pdftotext")
            .arg("-v")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    })
}

/// Text of each page of a PDF, in order. `None` if `pdftotext` failed or took too long.
pub fn pdf_pages(path: &Path) -> Option<Vec<String>> {
    let mut child = Command::new("pdftotext")
        .args(["-q", "-enc", "UTF-8"])
        .arg(path)
        .arg("-")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut out = Vec::new();
        stdout.take(MAX_TEXT + 1).read_to_end(&mut out).map(|_| out)
    });
    let deadline = Instant::now() + PDF_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    let out = reader.join().ok()?.ok()?;
    if !status.success() || out.len() as u64 > MAX_TEXT {
        return None;
    }
    let text = String::from_utf8_lossy(&out);
    let mut pages: Vec<String> = text.split('\u{c}').map(str::to_string).collect();
    if pages.last().is_some_and(|p| p.trim().is_empty()) {
        pages.pop(); // pdftotext ends the last page with a form feed too
    }
    Some(pages)
}

/// Body text of a .docx file: one line per paragraph.
pub fn docx_text(bytes: &[u8]) -> Option<String> {
    let xml = zip_entry(bytes, "word/document.xml")?;
    Some(word_xml_text(&String::from_utf8_lossy(&xml)))
}

fn u16_at(b: &[u8], at: usize) -> Option<usize> {
    b.get(at..at.checked_add(2)?).map(|s| u16::from_le_bytes([s[0], s[1]]) as usize)
}

fn u32_at(b: &[u8], at: usize) -> Option<usize> {
    b.get(at..at.checked_add(4)?).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
}

/// One file from a zip archive, stored or deflated. Zip64 and encrypted entries are not
/// supported (Word does not use them for document.xml).
fn zip_entry(b: &[u8], want: &str) -> Option<Vec<u8>> {
    const EOCD: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
    const CENTRAL: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
    const LOCAL: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];
    // The end-of-central-directory record sits in the last 22 bytes plus an optional comment.
    let last = b.len().checked_sub(22)?;
    let eocd = (last.saturating_sub(0xffff)..=last).rev().find(|&i| b[i..i + 4] == EOCD)?;
    let entries = u16_at(b, eocd + 10)?;
    let mut p = u32_at(b, eocd + 16)?;
    for _ in 0..entries {
        if b.get(p..p.checked_add(4)?)? != CENTRAL {
            return None;
        }
        let flags = u16_at(b, p + 8)?;
        let method = u16_at(b, p + 10)?;
        let compressed = u32_at(b, p + 20)?;
        let size = u32_at(b, p + 24)? as u64;
        let (name_len, extra_len, comment_len) = (u16_at(b, p + 28)?, u16_at(b, p + 30)?, u16_at(b, p + 32)?);
        let local = u32_at(b, p + 42)?;
        let name = b.get(p + 46..(p + 46).checked_add(name_len)?)?;
        if name == want.as_bytes() {
            if flags & 1 != 0 || size > MAX_TEXT || b.get(local..local.checked_add(4)?)? != LOCAL {
                return None;
            }
            let start = local + 30 + u16_at(b, local + 26)? + u16_at(b, local + 28)?;
            let data = b.get(start..start.checked_add(compressed)?)?;
            return match method {
                0 => Some(data.to_vec()),
                8 => {
                    let mut out = Vec::new();
                    flate2::read::DeflateDecoder::new(data).take(MAX_TEXT + 1).read_to_end(&mut out).ok()?;
                    (out.len() as u64 <= MAX_TEXT).then_some(out)
                }
                _ => None,
            };
        }
        p = p.checked_add(46 + name_len + extra_len + comment_len)?;
    }
    None
}

/// Text runs (`<w:t>`), tabs and line breaks of WordprocessingML, one line per paragraph.
/// Deleted text in tracked changes (`<w:delText>`) is left out.
fn word_xml_text(xml: &str) -> String {
    let mut out = String::new();
    let mut rest = xml;
    while let Some(lt) = rest.find('<') {
        let Some(len) = rest[lt..].find('>') else { break };
        let gt = lt + len;
        let tag = &rest[lt + 1..gt];
        let closing = tag.starts_with('/');
        let empty = tag.ends_with('/');
        let name = tag.trim_start_matches('/').split(|c: char| c.is_whitespace() || c == '/').next().unwrap_or("");
        match (name, closing) {
            ("w:t", false) if !empty => {
                let body = &rest[gt + 1..];
                let end = body.find("</w:t>").unwrap_or(body.len());
                out.push_str(&decode_entities(&body[..end]));
                rest = &body[end..];
                continue;
            }
            ("w:tab", false) => out.push('\t'),
            ("w:br", false) | ("w:cr", false) => out.push('\n'),
            ("w:p", true) => out.push('\n'),
            ("w:p", false) if empty => out.push('\n'),
            _ => {}
        }
        rest = &rest[gt + 1..];
    }
    out
}

fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let tail = &rest[amp..];
        let decoded = tail.find(';').filter(|&semi| semi <= 12).and_then(|semi| {
            let ent = &tail[1..semi];
            let ch = match ent {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                _ => ent
                    .strip_prefix("#x")
                    .or_else(|| ent.strip_prefix("#X"))
                    .map(|h| u32::from_str_radix(h, 16).ok())
                    .unwrap_or_else(|| ent.strip_prefix('#').and_then(|d| d.parse().ok()))
                    .and_then(char::from_u32),
            };
            ch.map(|c| (c, semi))
        });
        match decoded {
            Some((c, semi)) => {
                out.push(c);
                rest = &tail[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zip archive with stored (uncompressed) entries, built by hand.
    fn stored_zip(entries: &[(&str, &[u8], Option<u32>)]) -> Vec<u8> {
        let (mut out, mut central) = (Vec::new(), Vec::new());
        for (name, data, claimed_size) in entries {
            let offset = out.len() as u32;
            let size = claimed_size.unwrap_or(data.len() as u32);
            out.extend([0x50, 0x4b, 0x03, 0x04, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend((data.len() as u32).to_le_bytes());
            out.extend(size.to_le_bytes());
            out.extend((name.len() as u16).to_le_bytes());
            out.extend([0, 0]);
            out.extend(name.as_bytes());
            out.extend(*data);
            central.extend([0x50, 0x4b, 0x01, 0x02, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            central.extend((data.len() as u32).to_le_bytes());
            central.extend(size.to_le_bytes());
            central.extend((name.len() as u16).to_le_bytes());
            central.extend([0; 12]);
            central.extend(offset.to_le_bytes());
            central.extend(name.as_bytes());
        }
        let (cd_offset, cd_len) = (out.len() as u32, central.len() as u32);
        out.extend(&central);
        out.extend([0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0]);
        out.extend((entries.len() as u16).to_le_bytes());
        out.extend((entries.len() as u16).to_le_bytes());
        out.extend(cd_len.to_le_bytes());
        out.extend(cd_offset.to_le_bytes());
        out.extend([0, 0]);
        out
    }

    #[test]
    fn word_paragraphs_tabs_breaks_and_entities() {
        let xml = r#"<w:document><w:body><w:p><w:r><w:t>Fish &amp; chips</w:t></w:r><w:r><w:tab/><w:t xml:space="preserve"> at 5 &lt;pm&gt;</w:t></w:r></w:p><w:p/><w:p><w:r><w:t>Caf&#233; &#x2014; line</w:t><w:br/><w:t>two</w:t><w:delText>gone</w:delText></w:r></w:p><w:tbl/></w:body></w:document>"#;
        assert_eq!(word_xml_text(xml), "Fish & chips\t at 5 <pm>\n\nCafé — line\ntwo\n");
    }

    #[test]
    fn unknown_or_broken_entities_are_kept() {
        assert_eq!(decode_entities("a &nbsp; b & c &#xZZ;"), "a &nbsp; b & c &#xZZ;");
    }

    #[test]
    fn docx_from_stored_zip() {
        let xml = b"<w:p><w:r><w:t>Hello docx</w:t></w:r></w:p>";
        let zip = stored_zip(&[("[Content_Types].xml", b"<Types/>", None), ("word/document.xml", xml, None)]);
        assert_eq!(docx_text(&zip).as_deref(), Some("Hello docx\n"));
    }

    #[test]
    fn docx_fixture_written_by_a_real_zip_library() {
        // tests/fixtures/sample.docx: deflate-compressed, as Word writes it.
        let text = docx_text(include_bytes!("../tests/fixtures/sample.docx")).expect("docx text");
        assert!(text.contains("Quarterly report"), "{text}");
        assert!(text.contains("Revenue grew 12% & costs fell"), "{text}");
    }

    #[test]
    fn rejects_oversized_entries_and_garbage() {
        let bomb = stored_zip(&[("word/document.xml", b"<w:t>x</w:t>", Some(u32::MAX))]);
        assert_eq!(docx_text(&bomb), None);
        assert_eq!(docx_text(b"not a zip at all"), None);
        assert_eq!(docx_text(&[]), None);
        let mut cut = stored_zip(&[("word/document.xml", b"<w:t>x</w:t>", None)]);
        cut.truncate(cut.len() - 30);
        assert_eq!(docx_text(&cut), None);
    }

    #[test]
    fn pdf_pages_when_pdftotext_is_installed() {
        if !pdftotext_available() {
            eprintln!("pdftotext not installed: skipping");
            return;
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample.pdf");
        let pages = pdf_pages(&path).expect("pdftotext output");
        assert_eq!(pages.len(), 2, "{pages:?}");
        assert!(pages[0].contains("Router reset guide"), "{pages:?}");
        assert!(pages[1].contains("Hold the button for ten seconds"), "{pages:?}");
    }
}
