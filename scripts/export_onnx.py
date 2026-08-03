"""Export SentenceTransformer to ONNX with INT8 quantization.

Prerequisites: pip install optimum[onnxruntime] onnx onnxruntime
Output: <repo>/data/all-MiniLM-L6-v2-int8/ (ONNX model)
"""

import os, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "all-MiniLM-L6-v2-int8")
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

def main():
    print(f"Exporting {MODEL_NAME} → {OUTPUT_DIR} ...")
    t0 = time.time()

    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        f"sentence-transformers/{MODEL_NAME}",
        cache_dir=HF_CACHE
    )

    model = ORTModelForFeatureExtraction.from_pretrained(
        f"sentence-transformers/{MODEL_NAME}",
        export=True,
        provider="CPUExecutionProvider",
        cache_dir=HF_CACHE,
    )

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    dt = time.time() - t0
    print(f"Done in {dt:.1f}s → {OUTPUT_DIR}")
    print(f"Files: {os.listdir(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
