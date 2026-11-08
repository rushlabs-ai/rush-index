//! Embedding models on ONNX Runtime, with the preprocessing verified in tools/verify:
//! - Qwen3-Embedding-0.6B: left padding, the tokenizer appends <|endoftext|>, last-token
//!   pooling, L2 normalisation. Queries get the retrieval instruction prefix.
//! - SigLIP2 base/224: images resized to 224x224 (bilinear), scaled to [0,1] and normalised
//!   with mean 0.5 / std 0.5; text padded to 64 tokens. `pooler_output` is the embedding.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use half::f16;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{HasSelectedOutputs, OutputSelector, RunOptions};
use ort::value::Tensor;
use tokenizers::Tokenizer;

pub const TEXT_DIM: usize = 1024;
pub const IMAGE_DIM: usize = 768;
const QWEN_LAYERS: usize = 28;
const QWEN_KV_HEADS: usize = 8;
const QWEN_HEAD_DIM: usize = 128;
const QWEN_MAX_TOKENS: usize = 512;
const SIGLIP_TEXT_LEN: usize = 64;
const SIGLIP_SIZE: u32 = 224;
pub const QUERY_INSTRUCT: &str = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ";

pub fn environment(lib: &Path) -> Result<Environment> {
    let builder = ort::init_from(lib).map_err(|e| anyhow!("loading ONNX Runtime from {}: {e}", lib.display()))?;
    builder.build().map_err(|e| anyhow!("starting ONNX Runtime: {e}"))
}

fn session(env: &Environment, path: &Path, threads: usize, gpu: bool) -> Result<Session> {
    let mut b = Session::builder(env).map_err(|e| anyhow!("{e}"))?;
    if gpu {
        // Fail loudly instead of silently running a "GPU" job on the CPU.
        b = b
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
            .map_err(|e| anyhow!("CUDA could not be enabled ({e}). --gpu needs an NVIDIA driver, the CUDA 12 runtime and cuDNN 9 libraries on the library path"))?;
    }
    b = b.with_optimization_level(GraphOptimizationLevel::Level3).map_err(|e| anyhow!("{e}"))?;
    if threads > 0 {
        b = b.with_intra_threads(threads).map_err(|e| anyhow!("{e}"))?;
    }
    b.commit_from_file(path).map_err(|e| anyhow!("loading {}: {e}", path.display()))
}

fn only(output: &str) -> Result<RunOptions<HasSelectedOutputs>> {
    Ok(RunOptions::new().map_err(|e| anyhow!("{e}"))?.with_outputs(OutputSelector::no_default().with(output)))
}

fn l2(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

/// Inputs and outputs may be f32 (CPU models) or f16 (GPU models).
enum Precision {
    F32,
    F16,
}

fn precision_of(s: &Session, name: &str) -> Precision {
    let is_f16 = s.inputs().iter().find(|i| i.name() == name).map(|i| format!("{:?}", i.dtype()).contains("Float16")).unwrap_or(false);
    if is_f16 { Precision::F16 } else { Precision::F32 }
}

fn extract_f32(value: &ort::value::DynValue) -> Result<(Vec<i64>, Vec<f32>)> {
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return Ok((shape.to_vec(), data.to_vec()));
    }
    let (shape, data) = value.try_extract_tensor::<f16>().map_err(|e| anyhow!("output is neither f32 nor f16: {e}"))?;
    Ok((shape.to_vec(), data.iter().map(|x| x.to_f32()).collect()))
}

pub struct TextEmbedder {
    session: Session,
    tok: Tokenizer,
    kv_f16: bool,
    run: RunOptions<HasSelectedOutputs>,
    pub batch: usize,
}

impl TextEmbedder {
    pub fn load(env: &Environment, dir: &Path, gpu: bool, threads: usize) -> Result<Self> {
        let model = if gpu { "qwen3/model_fp16.onnx" } else { "qwen3/model.onnx" };
        let session = session(env, &dir.join(model), threads, gpu)?;
        let kv = session.inputs().iter().filter(|i| i.name().starts_with("past_key_values")).count();
        if kv != QWEN_LAYERS * 2 {
            bail!("unexpected Qwen3-Embedding export: {kv} cache inputs");
        }
        let kv_f16 = matches!(precision_of(&session, "past_key_values.0.key"), Precision::F16);
        let mut tok = Tokenizer::from_file(dir.join("qwen3/tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        tok.with_truncation(Some(tokenizers::TruncationParams { max_length: QWEN_MAX_TOKENS, ..Default::default() }))
            .map_err(|e| anyhow!("{e}"))?;
        Ok(Self { session, tok, kv_f16, run: only("last_hidden_state")?, batch: if gpu { 64 } else { 8 } })
    }

    /// Embeddings for documents (`query = false`) or search queries, in input order.
    pub fn embed(&mut self, texts: &[String], query: bool) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<String> = texts.iter().map(|t| if query { format!("{QUERY_INSTRUCT}{t}") } else { t.clone() }).collect();
        let enc = self.tok.encode_batch(inputs, true).map_err(|e| anyhow!("{e}"))?;
        // Batch texts of similar length together: less padding, same embeddings.
        let mut order: Vec<usize> = (0..enc.len()).collect();
        order.sort_by_key(|&i| enc[i].get_ids().len());
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); enc.len()];
        for idx in order.chunks(self.batch.max(1)) {
            let batch: Vec<&tokenizers::Encoding> = idx.iter().map(|&i| &enc[i]).collect();
            for (&i, v) in idx.iter().zip(self.embed_batch(&batch)?) {
                out[i] = v;
            }
        }
        Ok(out)
    }

    fn embed_batch(&mut self, enc: &[&tokenizers::Encoding]) -> Result<Vec<Vec<f32>>> {
        let b = enc.len();
        let width = enc.iter().map(|e| e.get_ids().len()).max().unwrap_or(1).max(1);
        let pad = 151_643i64; // <|endoftext|>, the tokenizer's pad token
        let (mut ids, mut mask, mut pos) = (vec![pad; b * width], vec![0i64; b * width], vec![0i64; b * width]);
        for (r, e) in enc.iter().enumerate() {
            let n = e.get_ids().len();
            let off = width - n; // left padding: the last position holds the last token for every row
            for (j, &id) in e.get_ids().iter().enumerate() {
                ids[r * width + off + j] = id as i64;
                mask[r * width + off + j] = 1;
                pos[r * width + off + j] = j as i64;
            }
        }
        let shape = vec![b as i64, width as i64];
        let mut feed: Vec<(String, ort::value::DynValue)> = vec![
            ("input_ids".into(), Tensor::from_array((shape.clone(), ids))?.into_dyn()),
            ("attention_mask".into(), Tensor::from_array((shape.clone(), mask))?.into_dyn()),
            ("position_ids".into(), Tensor::from_array((shape, pos))?.into_dyn()),
        ];
        let kv_shape = vec![b as i64, QWEN_KV_HEADS as i64, 0, QWEN_HEAD_DIM as i64];
        for layer in 0..QWEN_LAYERS {
            for kind in ["key", "value"] {
                let v = if self.kv_f16 {
                    Tensor::from_array((kv_shape.clone(), Vec::<f16>::new()))?.into_dyn()
                } else {
                    Tensor::from_array((kv_shape.clone(), Vec::<f32>::new()))?.into_dyn()
                };
                feed.push((format!("past_key_values.{layer}.{kind}"), v));
            }
        }
        let outputs = self.session.run_with_options(feed, &self.run).map_err(|e| anyhow!("Qwen3-Embedding: {e}"))?;
        let (shape, data) = extract_f32(&outputs["last_hidden_state"])?;
        let (t, d) = (shape[1] as usize, shape[2] as usize);
        Ok((0..b)
            .map(|r| {
                let start = (r * t + (t - 1)) * d;
                let mut v = data[start..start + d].to_vec();
                l2(&mut v);
                v
            })
            .collect())
    }
}

pub struct ImageEmbedder {
    vision: Session,
    text: Session,
    tok: Tokenizer,
    f16_in: bool,
    run: RunOptions<HasSelectedOutputs>,
    pub batch: usize,
}

impl ImageEmbedder {
    pub fn load(env: &Environment, dir: &Path, gpu: bool, threads: usize) -> Result<Self> {
        let (v, t) = if gpu {
            ("siglip2/vision_model_fp16.onnx", "siglip2/text_model_fp16.onnx")
        } else {
            ("siglip2/vision_model.onnx", "siglip2/text_model.onnx")
        };
        let vision = session(env, &dir.join(v), threads, gpu)?;
        let text = session(env, &dir.join(t), threads, gpu)?;
        let f16_in = matches!(precision_of(&vision, "pixel_values"), Precision::F16);
        let mut tok = Tokenizer::from_file(dir.join("siglip2/tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        tok.with_truncation(Some(tokenizers::TruncationParams { max_length: SIGLIP_TEXT_LEN, ..Default::default() }))
            .map_err(|e| anyhow!("{e}"))?;
        tok.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(SIGLIP_TEXT_LEN),
            pad_id: 0,
            pad_token: "<pad>".into(),
            ..Default::default()
        }));
        Ok(Self { vision, text, tok, f16_in, run: only("pooler_output")?, batch: if gpu { 64 } else { 8 } })
    }

    /// Decode and preprocess one image into a CHW float buffer.
    pub fn pixels(path: &Path) -> Result<Vec<f32>> {
        let img = image::open(path).with_context(|| format!("reading {}", path.display()))?.to_rgb8();
        let img = image::imageops::resize(&img, SIGLIP_SIZE, SIGLIP_SIZE, image::imageops::FilterType::Triangle);
        let n = (SIGLIP_SIZE * SIGLIP_SIZE) as usize;
        let mut chw = vec![0f32; 3 * n];
        for (i, p) in img.pixels().enumerate() {
            for c in 0..3 {
                chw[c * n + i] = (p[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        Ok(chw)
    }

    pub fn embed_images(&mut self, pixels: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks(self.batch.max(1)) {
            let shape = vec![chunk.len() as i64, 3, SIGLIP_SIZE as i64, SIGLIP_SIZE as i64];
            let flat: Vec<f32> = chunk.iter().flatten().copied().collect();
            let input = if self.f16_in {
                Tensor::from_array((shape, flat.into_iter().map(f16::from_f32).collect::<Vec<_>>()))?.into_dyn()
            } else {
                Tensor::from_array((shape, flat))?.into_dyn()
            };
            let outputs = self.vision.run_with_options(vec![("pixel_values".to_string(), input)], &self.run).map_err(|e| anyhow!("SigLIP2 vision: {e}"))?;
            let (shape, data) = extract_f32(&outputs["pooler_output"])?;
            let d = shape[1] as usize;
            out.extend(data.chunks(d).map(|v| {
                let mut v = v.to_vec();
                l2(&mut v);
                v
            }));
        }
        Ok(out)
    }

    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let enc = self.tok.encode(text, true).map_err(|e| anyhow!("{e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let input = Tensor::from_array((vec![1i64, ids.len() as i64], ids))?.into_dyn();
        let outputs = self.text.run_with_options(vec![("input_ids".to_string(), input)], &self.run).map_err(|e| anyhow!("SigLIP2 text: {e}"))?;
        let (_, data) = extract_f32(&outputs["pooler_output"])?;
        let mut v = data;
        l2(&mut v);
        Ok(v)
    }
}

pub fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}
