#!/usr/bin/env python3
"""Codex ↔ Deepseek bridge v4 — supports SSE streaming"""
import json, subprocess, sys, uuid, time
from http.server import HTTPServer, BaseHTTPRequestHandler

DS_KEY = "sk-<REDACTED>"

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        
        if self.path not in ("/v1/responses", "/v1/chat/completions"):
            self.send_error(404); return

        streaming = body.get("stream", False)
        model_name = body.get("model", "deepseek-chat").replace("deepseek/deepseek-", "deepseek-")
        model_name = model_name if model_name.startswith("deepseek-") else "deepseek-chat"

        messages = []
        if body.get("instructions"):
            messages.append({"role": "system", "content": body["instructions"]})
        inp = body.get("input", body.get("messages", ""))
        if isinstance(inp, str):
            messages.append({"role": "user", "content": inp})
        elif isinstance(inp, list):
            for item in inp:
                if isinstance(item, dict):
                    messages.append({"role": item.get("role","user"),
                                    "content": item.get("content", str(item))})
                else:
                    messages.append({"role": "user", "content": str(item)})
        if not messages:
            messages = [{"role": "user", "content": "hello"}]

        ds_body = json.dumps({
            "model": model_name,
            "messages": messages,
            "max_tokens": body.get("max_output_tokens", body.get("max_tokens", 4096)),
            "temperature": body.get("temperature", 0.3),
            "stream": streaming,
        })

        try:
            result = subprocess.run(
                ["curl", "--socks5-hostname", "127.0.0.1:1081", "-s", "--max-time", "60",
                 "https://api.deepseek.com/v1/chat/completions",
                 "-H", "Content-Type: application/json",
                 "-H", f"Authorization: Bearer {DS_KEY}",
                 "-d", ds_body],
                capture_output=True, text=True, timeout=65
            )
            if result.returncode != 0:
                raise Exception(f"curl: {result.stderr[:200]}")
            
            ds_resp = json.loads(result.stdout)
            if "choices" not in ds_resp:
                err = ds_resp.get("error", {}).get("message", str(ds_resp)[:200])
                raise Exception(f"DS error: {err}")
            
            text = ds_resp["choices"][0]["message"]["content"]
        except Exception as e:
            text = f"[Bridge error: {e}]"

        if streaming:
            # SSE streaming response
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            
            # Send a single chunk
            chunk = {
                "choices": [{"delta": {"content": text}, "index": 0, "finish_reason": None}]
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            # Send done signal
            done = {
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
            }
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
            self.wfile.write("data: [DONE]\n\n".encode())
        else:
            # Non-streaming response (Responses API format)
            resp_out = {
                "id": f"resp_{uuid.uuid4().hex[:8]}",
                "object": "response",
                "status": "completed",
                "output": [{"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": text}]}],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_out).encode())

    def log_message(self, *a): pass

port = int(sys.argv[1]) if len(sys.argv) > 1 else 3457
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
