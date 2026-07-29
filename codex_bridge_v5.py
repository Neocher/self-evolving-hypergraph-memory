#!/usr/bin/env python3
"""
Codex ↔ Deepseek 桥接 v5
—— 支持 Responses API SSE 流式格式
"""
import json, subprocess, sys, uuid, socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler

DS_KEY = "sk-<REDACTED>"
DS_URL = "https://api.deepseek.com/v1/chat/completions"
PROXY = "http://127.0.0.1:8083"
SOCKS_PROXY = "socks5://127.0.0.1:1081"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        if self.path not in ("/v1/responses", "/v1/chat/completions"):
            self.send_error(404)
            return

        streaming = body.get("stream", False)

        # ---- 提取消息 ----
        messages = []
        if body.get("instructions"):
            messages.append({"role": "system", "content": body["instructions"]})
        inp = body.get("input", body.get("messages", ""))
        if isinstance(inp, str):
            messages.append({"role": "user", "content": inp})
        elif isinstance(inp, list):
            for item in inp:
                if isinstance(item, dict):
                    messages.append({
                        "role": item.get("role", "user"),
                        "content": item.get("content", str(item)),
                    })
                else:
                    messages.append({"role": "user", "content": str(item)})
        if not messages:
            messages = [{"role": "user", "content": "hello"}]

        model = body.get("model", "deepseek-chat")
        if "/" in model:
            model = model.split("/")[-1]
        if not model.startswith("deepseek-"):
            model = "deepseek-chat"

        ds_body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": body.get("max_output_tokens", body.get("max_tokens", 4096)),
            "temperature": body.get("temperature", 0.3),
            "stream": False,  # 始终非流式调Deepseek，由本桥模拟SSE
        })

        # ---- 调用 Deepseek ----
        try:
            result = subprocess.run(
                ["curl", "-x", PROXY, "-s", "--max-time", "90",
                 DS_URL,
                 "-H", "Content-Type: application/json",
                 "-H", f"Authorization: Bearer {DS_KEY}",
                 "-d", ds_body],
                capture_output=True, text=True, timeout=95,
            )
            if result.returncode != 0:
                raise Exception(f"curl exit={result.returncode}: {result.stderr[:200]}")
            ds_resp = json.loads(result.stdout)
            if "choices" not in ds_resp:
                err = ds_resp.get("error", {}).get("message", str(ds_resp)[:200])
                raise Exception(f"DS error: {err}")
            text = ds_resp["choices"][0]["message"]["content"]
        except Exception as e:
            text = f"[Bridge error: {e}]"

        resp_id = f"resp_{uuid.uuid4().hex[:12]}"
        item_id = f"item_{uuid.uuid4().hex[:12]}"
        part_id = f"part_{uuid.uuid4().hex[:12]}"

        # ---- 构建 Responses API 完整响应 ----
        full_response = {
            "id": resp_id,
            "object": "response",
            "status": "completed",
            "model": model,
            "output": [{
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{
                    "id": part_id,
                    "type": "output_text",
                    "text": text,
                }],
            }],
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        }

        if streaming:
            # SSE 流式输出（OpenAI Responses API 标准事件序列）
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            def _sse(event, data_dict):
                self.wfile.write(f"event: {event}\ndata: {json.dumps(data_dict)}\n\n".encode())
                self.wfile.flush()

            resp_id = f"resp_{uuid.uuid4().hex[:12]}"

            # 1) response.created — 最小化，不含output
            created_resp = dict(full_response)
            created_resp["output"] = []
            _sse("response.created", {
                "type": "response.created",
                "response": created_resp,
            })

            # 2) response.output_item.added
            output_item = full_response["output"][0]
            _sse("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": 0,
                "item_id": item_id,
                "item": output_item,
            })

            # 3) response.content_part.added
            content_part = output_item["content"][0]
            _sse("response.content_part.added", {
                "type": "response.content_part.added",
                "part_index": 0,
                "item_id": item_id,
                "part": content_part,
            })

            # 4) response.output_text.delta
            _sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "delta": text,
                "item_id": item_id,
                "part_id": part_id,
                "output_index": 0,
            })

            # 5) response.output_text.done
            _sse("response.output_text.done", {
                "type": "response.output_text.done",
                "text": text,
                "item_id": item_id,
                "part_id": part_id,
                "output_index": 0,
            })

            # 6) response.content_part.done
            cp = full_response["output"][0]["content"][0]
            _sse("response.content_part.done", {
                "type": "response.content_part.done",
                "index": 0,
                "item_id": item_id,
                "part": cp,
            })

            # 7) response.output_item.done
            oi = full_response["output"][0]
            _sse("response.output_item.done", {
                "type": "response.output_item.done",
                "index": 0,
                "item_id": item_id,
                "item": oi,
            })

            # 8) response.completed
            _sse("response.completed", {
                "type": "response.completed",
                "response": full_response,
            })

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            # 非流式：标准 JSON
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(full_response).encode())

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    import socketserver
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3457
    server = socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
