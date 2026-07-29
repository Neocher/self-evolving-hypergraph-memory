#!/usr/bin/env python3
"""Codex ↔ Deepseek bridge (v3 - robust)"""
import json, subprocess, sys
from http.server import HTTPServer, BaseHTTPRequestHandler

DS_KEY = "sk-<REDACTED>"

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        
        if self.path != "/v1/responses":
            self.send_error(404); return

        # 提取消息
        messages = []
        if body.get("instructions"):
            messages.append({"role": "system", "content": body["instructions"]})
        inp = body.get("input", "")
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

        # 处理模型名
        model = body.get("model", "deepseek-chat").replace("deepseek/deepseek-", "deepseek-")
        
        ds_body = json.dumps({
            "model": model if model.startswith("deepseek-") else "deepseek-chat",
            "messages": messages,
            "max_tokens": body.get("max_output_tokens", 4096),
            "temperature": body.get("temperature", 0.3),
            "stream": False,
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
                raise Exception(f"curl failed: {result.stderr[:200]}")
            
            ds_resp = json.loads(result.stdout)
            
            if "choices" not in ds_resp:
                # Deepseek返回了错误
                err_msg = ds_resp.get("error", {}).get("message", str(ds_resp)[:200])
                raise Exception(f"Deepseek error: {err_msg}")
                
            text = ds_resp["choices"][0]["message"]["content"]
        except Exception as e:
            # 回退：直接返回简单的文本响应（让Codex不崩溃）
            text = f"I processed your request but encountered: {e}"

        resp_out = {
            "id": "resp_1", "object": "response", "status": "completed",
            "output": [{"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": text}]}],
        }
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_out).encode())
        except BrokenPipeError:
            pass

    def log_message(self, *a): pass

port = int(sys.argv[1]) if len(sys.argv) > 1 else 3457
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
