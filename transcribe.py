#!/usr/bin/env python3
"""Transcription service — converts audio to text via multiple backends.
Installed as part of the full multimodal capability stack.
"""
import base64
import json
import os
import subprocess
import sys
import urllib.request

def convert_to_mp3(input_path: str, output_path: str = "/tmp/audio_for_asr.mp3") -> str:
    """Convert any audio to 16kHz mono mp3 for API submission."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "32k",
         output_path],
        capture_output=True, check=True
    )
    return output_path


def transcribe_dashscope(audio_path: str, api_key: str = None) -> str:
    """Transcribe via DashScope qwen3.5-omni-plus using base64 audio."""
    if api_key is None:
        api_key = os.environ.get("QWEN_API_KEY", os.environ.get("DASHSCOPE_API_KEY", ""))
    if not api_key:
        api_key = "sk-REDACTED"

    mp3_path = convert_to_mp3(audio_path)

    with open(mp3_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    # 正确的 DashScope 原生多模态 API 格式
    body = {
        "model": "qwen3.5-omni-plus",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "请完整转写这段音频的内容。"},
                        {"audio": f"data:audio/mp3;base64,{audio_b64}"}
                    ]
                }
            ]
        },
        "parameters": {
            "result_format": "message"
        }
    }

    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read())
        # Native API response format: output.choices[0].message.content[0].text
        try:
            return result["output"]["choices"][0]["message"]["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return str(result)


def transcribe_whisper(audio_path: str, model_size: str = "tiny") -> str:
    """Transcribe via local Whisper model (if installed)."""
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch
        import soundfile as sf

        # Convert to 16kHz wav
        wav_path = "/tmp/audio_for_whisper.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             wav_path],
            capture_output=True, check=True
        )

        model_id = f"openai/whisper-{model_size}"
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)

        audio_input, sr = sf.read(wav_path)
        input_features = processor(
            audio_input, sampling_rate=sr,
            return_tensors="pt"
        ).input_features

        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    except ImportError:
        return "Whisper not installed. Install: pip install transformers torch soundfile"
    except Exception as e:
        return f"Whisper error: {e}"


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/admin/.hermes/cache/documents/doc_ed61b6665be4_阳神修炼与天心合一.m4a"

    print(f"🎤 转写: {path}")
    print()

    # Try DashScope API first
    print("📡 尝试 DashScope API...")
    try:
        result = transcribe_dashscope(path)
        if result and len(result) > 10 and "error" not in result.lower():
            print(f"✅ 结果:\n{result}")
            sys.exit(0)
        else:
            print(f"  ❌ {result[:100]}")
    except Exception as e:
        print(f"  ❌ {e}")

    print()
    print("📡 尝试本地 Whisper...")
    result = transcribe_whisper(path)
    print(f"✅ 结果:\n{result}")
