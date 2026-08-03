"""Export SentenceTransformer to ONNX from LOCAL cache (no network required)."""

import os, sys, time

MODEL_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "all-MiniLM-L6-v2-int8")

def main():
    print(f"Source: {MODEL_SNAPSHOT}")
    print(f"Output: {OUTPUT_DIR}")
    t0 = time.time()

    # Load from local snapshot
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        MODEL_SNAPSHOT,
        device="cpu",
        cache_folder=os.path.dirname(MODEL_SNAPSHOT),
    )

    # Export to ONNX via optimum
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_SNAPSHOT)
    onnx_model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_SNAPSHOT,
        export=True,
        provider="CPUExecutionProvider",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    onnx_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s")
    print(f"Files: {sorted(os.listdir(OUTPUT_DIR))}")


if __name__ == "__main__":
    main()
