#!/usr/bin/env python3
"""RLCD 0.6B 决策模型的 HTTP 服务 —— 与 Jev 的 POST /v1/systemone 契约兼容。

用途：替换 outputs/jev-duplex/server.py 里的上游 Jev API，
      让 demo 页指向本地模型（延迟从 ~1.1s 降到 ~25ms）。

启动：
    CKPT=/tmp/laya_rlcd/ckpt_20e PORT=8790 python3 api_server.py
    浏览器/demo 把请求发到 http://127.0.0.1:8790/v1/systemone

同时提供：
    GET  /health          健康检查 + 模型信息
    POST /v1/systemone    与 Jev 同形状
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_rlcd import DecisionModel  # noqa

CKPT = os.environ.get("CKPT", "/tmp/laya_rlcd/ckpt_20e")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8790"))

MODEL = None
BOOT_S = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        if self.path in ("/health", "/"):
            body = json.dumps({
                "ok": True,
                "model": "rlcd-qwen3-0.6b",
                "checkpoint": CKPT,
                "device": str(MODEL.device) if MODEL else None,
                "boot_seconds": round(BOOT_S, 2) if BOOT_S else None,
                "endpoint": "POST /v1/systemone",
            }, ensure_ascii=False).encode()
            return self._send(200, body)
        return self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path not in ("/v1/systemone", "/api/systemone"):
            return self._send(404, b'{"error":"not found"}')
        if MODEL is None:
            return self._send(503, b'{"error":"model not loaded"}')

        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw)
        except Exception as e:
            return self._send(400, json.dumps(
                {"detail": {"error_type": "api_usage_error", "message": "invalid json: %s" % e}}
            ).encode())

        state = req.get("state")
        questions = req.get("questions")
        if not state or not isinstance(questions, dict) or not questions:
            return self._send(400, json.dumps(
                {"detail": {"error_type": "api_usage_error", "message": "need state + questions"}}
            ).encode())

        t0 = time.perf_counter()
        try:
            resp = MODEL.predict(state, questions)
        except Exception as e:
            return self._send(500, json.dumps(
                {"detail": {"error_type": "internal", "message": "%s: %s" % (type(e).__name__, e)}}
            ).encode())
        resp["usage"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return self._send(200, json.dumps(resp, ensure_ascii=False).encode())


def main():
    global MODEL, BOOT_S
    t0 = time.time()
    print("加载 checkpoint: %s" % CKPT, flush=True)
    MODEL = DecisionModel(CKPT)
    BOOT_S = time.time() - t0
    print("✓ 模型就绪 (%.1fs) device=%s" % (BOOT_S, MODEL.device), flush=True)

    srv = None
    for port in range(PORT, PORT + 20):
        try:
            srv = ThreadingHTTPServer((HOST, port), Handler)
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit("端口 %d-%d 都不可用" % (PORT, PORT + 19))
    print("→ http://%s:%d   (POST /v1/systemone)" % (HOST, port), flush=True)
    print("  Ctrl-C 停止", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
