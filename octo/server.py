# -*- coding: utf-8 -*-
"""Octo HTTP API — chat with Octo from any app. stdlib only, loopback by default.

Usage:
    python -m octo.server                    # 127.0.0.1:8600
    python -m octo.server --port 9000

Endpoints:
    GET  /health -> {"status": "ok", "device": ...}
    POST /chat   -> body {"prompt": "...", "tokens": 200, "temperature": 0.8}
                    -> {"response": "..."}
    GET  /       -> minimal web chat page
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHECKPOINT = "octo_checkpoints/octo.pt"
_state = {"model": None, "tokenizer": None, "device": "cpu", "history": []}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Octo</title>
<style>body{font-family:sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
#out{white-space:pre-wrap;border:1px solid #ddd;border-radius:8px;padding:12px;min-height:120px}
input{width:70%;padding:8px}button{padding:8px 16px}</style></head>
<body><h2>🐙 Octo — offline LLM</h2><div id="out">loading model...</div><br>
<input id="q" placeholder="প্রশ্ন লিখুন..."><button onclick="go()">Send</button>
<script>
async function go(){const q=document.getElementById('q').value;if(!q)return;
document.getElementById('out').textContent='thinking...';
const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({prompt:q})});
const j=await r.json();document.getElementById('out').textContent=j.response||j.error;}
</script></body></html>"""


def load(checkpoint=CHECKPOINT):
    import torch
    from octo.chat import load_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _state["model"], _state["tokenizer"] = load_model(checkpoint, device)
    _state["device"] = device
    return _state


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            ready = _state["model"] is not None
            self._send(200, json.dumps({"status": "ok" if ready else "loading",
                                        "device": _state["device"]}))
        elif self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/chat":
            self._send(404, json.dumps({"error": "not found"}))
            return
        if _state["model"] is None:
            self._send(503, json.dumps({"error": "model still loading"}))
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "invalid json"}))
            return
        prompt = str(req.get("prompt", "")).strip()
        if not prompt:
            self._send(400, json.dumps({"error": "prompt required"}))
            return
        from octo.chat import complete
        response = complete(_state["model"], _state["tokenizer"], prompt,
                            int(req.get("tokens", 200)),
                            float(req.get("temperature", 0.8)),
                            int(req.get("top_k", 40)),
                            _state["device"],
                            history=_state["history"][-3:])
        _state["history"].append((prompt, response))
        self._send(200, json.dumps({"response": response}, ensure_ascii=False))

    def log_message(self, fmt, *args):
        pass


def make_server(checkpoint=CHECKPOINT, host="127.0.0.1", port=8600):
    if _state["model"] is None:
        load(checkpoint)
    return ThreadingHTTPServer((host, port), Handler)


def main():
    ap = argparse.ArgumentParser(description="Octo HTTP API server.")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"no checkpoint at {args.checkpoint}; run: python -m octo.train")
    print(f"loading model from {args.checkpoint} ...")
    server = make_server(args.checkpoint, args.host, args.port)
    print(f"Octo API: http://{args.host}:{args.port}  (web chat at /)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
