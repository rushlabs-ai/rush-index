"""Check that the ONNX models Rush Index uses match the original models, on Modal (CPU).

    modal run tools/verify/verify_onnx.py

For Qwen3-Embedding-0.6B (documents) and SigLIP2 base/224 (photos) it prints the ONNX inputs
and outputs, implements the preprocessing the Rust code will use (left padding and last-token
pooling; resize to 224, scale, normalise), and compares embeddings with the reference
PyTorch models. It also times CPU throughput for the int8 and fp32 files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
REV = {
    "onnx-community/Qwen3-Embedding-0.6B-ONNX": "c25a394dd583836952667c12f008335071b3f43d",
    "onnx-community/siglip2-base-patch16-224-ONNX": "ba1f3b0843f24bc5417d38e19c37b287d719b2f4",
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "google/siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
}
IMAGES = {  # COCO val2017 (CC BY 4.0), the usual Hugging Face example photos
    "cats": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "living room": "http://images.cocodataset.org/val2017/000000000139.jpg",
    "bear": "http://images.cocodataset.org/val2017/000000000285.jpg",
}
TEXTS = [
    "How do I reset my router to factory settings?",
    "Hold the reset button for 10 seconds until the lights flash.",
    "Bananas are a good source of potassium.",
    "def read_lines(path):\n    with open(path) as f:\n        return f.readlines()",
]
QUERY_INSTRUCT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

app = modal.App("rush-index-verify")
volume = modal.Volume.from_name("rush-index", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .env({"HF_HOME": "/vol/hf"})
)


def cos(a, b):
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def describe(sess):
    def short(items):
        main = [(i.name, i.type, i.shape) for i in items if not i.name.startswith(("past_key_values", "present"))]
        cache = sum(1 for i in items if i.name.startswith(("past_key_values", "present")))
        return main + ([f"... plus {cache} KV-cache tensors"] if cache else [])
    return {"inputs": short(sess.get_inputs()), "outputs": short(sess.get_outputs())}


@app.function(image=image, cpu=4, memory=16384, volumes={"/vol": volume}, timeout=3600)
def verify() -> dict:
    import io
    import urllib.request

    import numpy as np
    import onnxruntime as ort
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    torch.set_grad_enabled(False)
    report: dict = {"onnxruntime": ort.__version__}
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4

    # -- Qwen3-Embedding --------------------------------------------------------
    onnx_dir = snapshot_download("onnx-community/Qwen3-Embedding-0.6B-ONNX", revision=REV["onnx-community/Qwen3-Embedding-0.6B-ONNX"], local_dir="/vol/models/Qwen3-Embedding-0.6B-ONNX",
                                 allow_patterns=["*.json", "onnx/model_int8.onnx", "onnx/model.onnx", "onnx/model.onnx_data"])
    ref_dir = snapshot_download("Qwen/Qwen3-Embedding-0.6B", revision=REV["Qwen/Qwen3-Embedding-0.6B"], local_dir="/vol/models/Qwen3-Embedding-0.6B",
                                allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(onnx_dir, padding_side="left")
    texts = [QUERY_INSTRUCT + TEXTS[0]] + TEXTS[1:]
    batch = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    ref = AutoModel.from_pretrained(ref_dir, dtype=torch.float32).eval()
    out = ref(input_ids=torch.tensor(batch["input_ids"]), attention_mask=torch.tensor(batch["attention_mask"]))
    ref_emb = out.last_hidden_state[:, -1].numpy()  # left padding: the last position is the last token
    report["qwen3_tokenizer"] = {"padding_side": tok.padding_side, "eos_appended": int(batch["input_ids"][0, -1]),
                                 "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id}
    for variant in ("model_int8.onnx", "model.onnx"):
        sess = ort.InferenceSession(str(Path(onnx_dir) / "onnx" / variant), opts, providers=["CPUExecutionProvider"])
        feed = {"input_ids": batch["input_ids"].astype(np.int64), "attention_mask": batch["attention_mask"].astype(np.int64)}
        names = {i.name for i in sess.get_inputs()}
        if "position_ids" in names:
            pos = np.cumsum(batch["attention_mask"], axis=1) - 1
            feed["position_ids"] = np.clip(pos, 0, None).astype(np.int64)
        feed = {k: v for k, v in feed.items() if k in names}
        # Decoder export with a KV cache: feed empty caches (sequence length 0) for every layer.
        cfg = json.loads((Path(onnx_dir) / "config.json").read_text())
        kv_heads, head_dim = cfg["num_key_value_heads"], cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        for i in sess.get_inputs():
            if i.name.startswith("past_key_values"):
                dtype = np.float16 if "float16" in i.type else np.float32
                feed[i.name] = np.zeros((len(texts), kv_heads, 0, head_dim), dtype=dtype)
        out_names = [o.name for o in sess.get_outputs()]
        t0 = time.time()
        res = sess.run(["last_hidden_state"] if "last_hidden_state" in out_names else None, feed)
        dt = time.time() - t0
        hidden = res[0]
        emb = hidden[:, -1] if hidden.ndim == 3 else hidden
        sims = [round(cos(emb[i], ref_emb[i]), 5) for i in range(len(texts))]
        q, d_good, d_bad = emb[0], emb[1], emb[2]
        report[f"qwen3/{variant}"] = {**describe(sess), "cosine_vs_reference": sims,
                                      "query_doc_similarity": {"relevant": round(cos(q, d_good), 3), "irrelevant": round(cos(q, d_bad), 3)},
                                      "seconds_for_4_texts": round(dt, 3)}

    # -- SigLIP2 ----------------------------------------------------------------
    s_onnx = snapshot_download("onnx-community/siglip2-base-patch16-224-ONNX", revision=REV["onnx-community/siglip2-base-patch16-224-ONNX"], local_dir="/vol/models/siglip2-base-patch16-224-ONNX",
                               allow_patterns=["*.json", "*.model", "onnx/text_model_int8.onnx", "onnx/vision_model_int8.onnx",
                                               "onnx/text_model.onnx", "onnx/vision_model.onnx"])
    s_ref = snapshot_download("google/siglip2-base-patch16-224", revision=REV["google/siglip2-base-patch16-224"], local_dir="/vol/models/siglip2-base-patch16-224",
                              allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
    proc = AutoProcessor.from_pretrained(s_ref)
    model = AutoModel.from_pretrained(s_ref, dtype=torch.float32).eval()
    pics = {k: Image.open(io.BytesIO(urllib.request.urlopen(u).read())).convert("RGB") for k, u in IMAGES.items()}
    labels = ["a photo of two cats", "a photo of a living room", "a photo of a bear"]

    # Preprocessing as Rust will do it: resize to 224x224 (bilinear), scale to [0,1], normalise with 0.5/0.5.
    def manual(img):
        a = np.asarray(img.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return ((a - 0.5) / 0.5).transpose(2, 0, 1)

    pv_ref = proc(images=list(pics.values()), return_tensors="pt")["pixel_values"]
    pv_manual = np.stack([manual(im) for im in pics.values()])
    report["siglip2_preprocessing_max_abs_diff"] = float(np.abs(pv_ref.numpy() - pv_manual).max())
    text_in = proc(text=labels, padding="max_length", max_length=64, return_tensors="pt")
    report["siglip2_text_tokens"] = {"first": text_in["input_ids"][0][:12].tolist(), "keys": list(text_in.keys())}
    ref_img = model.get_image_features(pixel_values=pv_ref)
    ref_img = (ref_img.pooler_output if hasattr(ref_img, "pooler_output") else ref_img).numpy()
    ref_txt = model.get_text_features(input_ids=text_in["input_ids"])
    ref_txt = (ref_txt.pooler_output if hasattr(ref_txt, "pooler_output") else ref_txt).numpy()

    for suffix in ("_int8", ""):
        v = ort.InferenceSession(str(Path(s_onnx) / "onnx" / f"vision_model{suffix}.onnx"), opts, providers=["CPUExecutionProvider"])
        t = ort.InferenceSession(str(Path(s_onnx) / "onnx" / f"text_model{suffix}.onnx"), opts, providers=["CPUExecutionProvider"])
        t0 = time.time()
        vout = v.run(None, {v.get_inputs()[0].name: pv_manual.astype(np.float32)})
        v_dt = time.time() - t0
        tfeed = {i.name: text_in[i.name].numpy().astype(np.int64) for i in t.get_inputs() if i.name in text_in}
        tout = t.run(None, tfeed)
        vnames, tnames = [o.name for o in v.get_outputs()], [o.name for o in t.get_outputs()]
        img_emb = vout[vnames.index("pooler_output")] if "pooler_output" in vnames else vout[-1]
        txt_emb = tout[tnames.index("pooler_output")] if "pooler_output" in tnames else tout[-1]
        sim = (txt_emb / np.linalg.norm(txt_emb, axis=1, keepdims=True)) @ (img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)).T
        report[f"siglip2{suffix or '_fp32'}"] = {
            "vision": describe(v), "text": describe(t),
            "image_cosine_vs_reference": [round(cos(img_emb[i], ref_img[i]), 5) for i in range(len(pics))],
            "text_cosine_vs_reference": [round(cos(txt_emb[i], ref_txt[i]), 5) for i in range(len(labels))],
            "zero_shot_correct": int(sum(int(np.argmax(sim[:, j]) == j) for j in range(len(labels)))),
            "seconds_for_3_images": round(v_dt, 3),
        }
    return report


@app.local_entrypoint()
def main() -> None:
    res = verify.remote()
    out = HERE / "results.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))
