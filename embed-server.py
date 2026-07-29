#!/usr/bin/env python3
"""Embedding server — provides text embeddings via ONNX Runtime.
Replaces the Go/ORT embedder to avoid SIGSEGV conflicts.
"""
import json
import os
import sys
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from tokenizers import Tokenizer

app = FastAPI()

class EmbedRequest(BaseModel):
    texts: list[str]
    model: str = "jina-v2-base-code-int8"

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    model: str

models = {}

def load_model(model_id: str) -> tuple:
    """Load a model and tokenizer from ROUTER_ONNX_ASSETS_DIR."""
    assets_root = os.environ.get("ROUTER_ONNX_ASSETS_DIR", "/opt/router/assets")
    model_dir = os.path.join(assets_root, model_id)
    
    # Try legacy flat layout
    model_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_path):
        model_path = os.path.join(assets_root, "model.onnx")
    
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Model files not found for {model_id} at {model_dir}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Load ONNX model
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 2
    
    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=['CPUExecutionProvider']
    )
    
    return session, tokenizer

@app.on_event("startup")
async def startup():
    models["jina-v2-base-code-int8"] = load_model("jina-v2-base-code-int8")

@app.get("/health")
async def health():
    return {"status": "ok", "loaded_models": list(models.keys())}

@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if req.model not in models:
        raise HTTPException(404, f"Model {req.model} not loaded")
    
    session, tokenizer = models[req.model]
    
    # Tokenize
    input_ids = []
    attention_mask = []
    for text in req.texts:
        enc = tokenizer.encode(text, add_special_tokens=True)
        ids = enc.ids
        input_ids.append(ids + [0] * (512 - len(ids)) if len(ids) < 512 else ids[:512])
        mask = [1] * len(enc.ids) + [0] * (512 - len(enc.ids)) if len(enc.ids) < 512 else [1] * 512
        attention_mask.append(mask)
    
    input_ids = np.array(input_ids, dtype=np.int64)
    attention_mask = np.array(attention_mask, dtype=np.int64)
    
    # Run inference
    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    
    # Mean pooling
    token_embeddings = outputs[0]
    mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.sum(mask_expanded, axis=1)
    mean_embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)
    
    # L2 normalize
    norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
    normalized = mean_embeddings / np.maximum(norms, 1e-9)
    
    return EmbedResponse(
        embeddings=normalized.tolist(),
        dim=normalized.shape[1],
        model=req.model
    )

if __name__ == "__main__":
    port = int(os.environ.get("EMBED_PORT", "8089"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
