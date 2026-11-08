//! Model files and the ONNX Runtime library, pinned by URL and SHA-256.
//!
//! `rush-index fetch` downloads them with the system's `curl` (no TLS stack in this binary)
//! and checks every file's SHA-256 before using it.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

pub struct File {
    pub path: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
}

macro_rules! file {
    ($path:expr, $base:expr, $rel:expr, $sha:expr, $bytes:expr) => {
        File { path: $path, url: concat!($base, "/", $rel), sha256: $sha, bytes: $bytes }
    };
}

/// Full-precision models (CPU). The int8 files were measured too lossy (cosine 0.66-0.91 vs
/// the original models; see tools/verify), so they are not used.
pub const CPU_MODELS: &[File] = &[
    file!("qwen3/model.onnx", "https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/c25a394dd583836952667c12f008335071b3f43d", "onnx/model.onnx",
          "bf27b2f3f9ef9c32ca337d75b361fa99439deaeaefe82e4701b2dbd8439197cc", 307_161_415),
    file!("qwen3/model.onnx_data", "https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/c25a394dd583836952667c12f008335071b3f43d", "onnx/model.onnx_data",
          "f0a61604465929a27e68aa6217c8c89ec6186572f0209fdb7711adda48a9b9a9", 2_093_436_928),
    file!("siglip2/vision_model.onnx", "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/ba1f3b0843f24bc5417d38e19c37b287d719b2f4", "onnx/vision_model.onnx",
          "c0573e3f4140c3a7c4e9cc5912bd6b26a033b46a6a8e8af26cbea262b163bcad", 371_807_752),
    file!("siglip2/text_model.onnx", "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/ba1f3b0843f24bc5417d38e19c37b287d719b2f4", "onnx/text_model.onnx",
          "baf12d941beabafafb14f7b4adb38dc15be18681b964a84410ec53d9d65e6293", 1_129_469_657),
];

/// Half-precision models (NVIDIA GPU).
pub const GPU_MODELS: &[File] = &[
    file!("qwen3/model_fp16.onnx", "https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/c25a394dd583836952667c12f008335071b3f43d", "onnx/model_fp16.onnx",
          "f376cf065d912675dd559270e19361367ebc28c938b9811dd61d5834c78d0834", 583_522),
    file!("qwen3/model_fp16.onnx_data", "https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/c25a394dd583836952667c12f008335071b3f43d", "onnx/model_fp16.onnx_data",
          "06fbfded30166e3a33fe5252d7002ed1c370a4ec1bf45aba778cdf6cf7d31439", 1_199_927_296),
    file!("siglip2/vision_model_fp16.onnx", "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/ba1f3b0843f24bc5417d38e19c37b287d719b2f4", "onnx/vision_model_fp16.onnx",
          "a1959f7bd3993a607e48839f6d01e25b876fe76afda301b028b78eef68aabd95", 186_039_516),
    file!("siglip2/text_model_fp16.onnx", "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/ba1f3b0843f24bc5417d38e19c37b287d719b2f4", "onnx/text_model_fp16.onnx",
          "711da56ada0a4aa11c7dd3320df741081a3cae4f0ae1b5e5c6d5b294738d0eb0", 564_862_230),
];

pub const TOKENIZERS: &[File] = &[
    file!("qwen3/tokenizer.json", "https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/c25a394dd583836952667c12f008335071b3f43d", "tokenizer.json",
          "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a", 11_423_705),
    file!("siglip2/tokenizer.json", "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/ba1f3b0843f24bc5417d38e19c37b287d719b2f4", "tokenizer.json",
          "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322", 34_363_039),
];

/// ONNX Runtime 1.31.1 release archives (github.com/microsoft/onnxruntime).
pub struct Runtime {
    pub archive: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
    pub lib: &'static str,
}

const ORT: &str = "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1";

pub fn runtime(gpu: bool) -> Result<Runtime> {
    let r = match (std::env::consts::OS, std::env::consts::ARCH, gpu) {
        ("linux", "x86_64", false) => Runtime { archive: "onnxruntime-linux-x64-1.31.1.tgz",
            sha256: "c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9", bytes: 11_413_655, lib: "libonnxruntime.so" },
        ("linux", "x86_64", true) => Runtime { archive: "onnxruntime-linux-x64-gpu_cuda12-1.31.1.tgz",
            sha256: "57cd51f677eca19a269cdba31b0c16c15d73a8feea41f170aa8b750447ab1f70", bytes: 450_233_079, lib: "libonnxruntime.so" },
        ("linux", "aarch64", false) => Runtime { archive: "onnxruntime-linux-aarch64-1.31.1.tgz",
            sha256: "5317505e2da99be65f8e2775ba3c2afd6d2961fd78aba1bd8f56159f48820c78", bytes: 10_332_381, lib: "libonnxruntime.so" },
        ("macos", "aarch64", false) => Runtime { archive: "onnxruntime-osx-arm64-1.31.1.tgz",
            sha256: "2e1344a1926ee50eb25b38153cb6992c3567515aef0e677a1a520a1798896d6f", bytes: 36_936_435, lib: "libonnxruntime.dylib" },
        ("windows", "x86_64", false) => Runtime { archive: "onnxruntime-win-x64-1.31.1.zip",
            sha256: "5869e26bb25314375f2d5ef1877e3ea06b1543d6c710e6e4a1ec2f7177ab4b2c", bytes: 82_929_944, lib: "onnxruntime.dll" },
        (os, arch, gpu) => bail!("no pinned ONNX Runtime build for {os}/{arch}{}; set ORT_DYLIB_PATH to your own copy",
                                if gpu { " with CUDA" } else { "" }),
    };
    Ok(r)
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn download(url: &str, dest: &Path, sha256: &str) -> Result<()> {
    if dest.exists() && sha256_file(dest)? == sha256 {
        return Ok(());
    }
    fs::create_dir_all(dest.parent().unwrap())?;
    let part = dest.with_extension("part");
    let status = Command::new("curl")
        .args(["-fL", "--retry", "5", "--progress-bar", "-o"])
        .arg(&part)
        .arg(url)
        .status()
        .context("running curl (is it installed?)")?;
    if !status.success() {
        bail!("download failed: {url}");
    }
    let got = sha256_file(&part)?;
    if got != sha256 {
        let _ = fs::remove_file(&part);
        bail!("checksum mismatch for {url}: expected {sha256}, got {got}");
    }
    fs::rename(&part, dest)?;
    Ok(())
}

/// Download (or verify) the models and runtime for CPU or GPU use.
pub fn fetch(dir: &Path, gpu: bool) -> Result<PathBuf> {
    let models = if gpu { GPU_MODELS } else { CPU_MODELS };
    let total: u64 = models.iter().chain(TOKENIZERS).map(|f| f.bytes).sum();
    eprintln!("Models ({}, {:.1} GB) into {}", if gpu { "GPU, fp16" } else { "CPU, fp32" }, total as f64 / 1e9, dir.display());
    for f in models.iter().chain(TOKENIZERS) {
        eprintln!("  {}", f.path);
        download(f.url, &dir.join(f.path), f.sha256)?;
    }
    let rt = runtime(gpu)?;
    let archive = dir.join("runtime").join(rt.archive);
    eprintln!("  ONNX Runtime {} ({:.0} MB)", rt.archive, rt.bytes as f64 / 1e6);
    download(&format!("{ORT}/{}", rt.archive), &archive, rt.sha256)?;
    let unpack = dir.join("runtime").join(rt.archive.trim_end_matches(".tgz").trim_end_matches(".zip"));
    if !unpack.exists() {
        let st = Command::new("tar").arg("-xf").arg(&archive).arg("-C").arg(dir.join("runtime")).status()?;
        if !st.success() {
            bail!("could not unpack {}", archive.display());
        }
    }
    let lib = unpack.join("lib").join(rt.lib);
    if !lib.exists() {
        bail!("{} not found after unpacking", lib.display());
    }
    Ok(lib)
}

/// The runtime library to load: ORT_DYLIB_PATH, else the one `fetch` unpacked.
pub fn runtime_library(dir: &Path, gpu: bool) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        return Ok(PathBuf::from(p));
    }
    let rt = runtime(gpu)?;
    let lib = dir.join("runtime").join(rt.archive.trim_end_matches(".tgz").trim_end_matches(".zip")).join("lib").join(rt.lib);
    if !lib.exists() {
        bail!("ONNX Runtime not found; run `rush-index fetch{}` first", if gpu { " --gpu" } else { "" });
    }
    Ok(lib)
}

pub fn require(dir: &Path, gpu: bool) -> Result<()> {
    let models = if gpu { GPU_MODELS } else { CPU_MODELS };
    for f in models.iter().chain(TOKENIZERS) {
        if !dir.join(f.path).exists() {
            bail!("missing {}; run `rush-index fetch{}` first", f.path, if gpu { " --gpu" } else { "" });
        }
    }
    Ok(())
}
