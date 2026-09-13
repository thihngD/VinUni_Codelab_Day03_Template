"""
Web UI demo cho Lab #3: mô phỏng Chatbot Baseline vs ReAct Agent
(cả bản rule-based lẫn bản gọi Gemini API thật), và hiển thị dữ liệu
gốc của dự án (flight_data.json, weather_data.json, customer_queries.json).

Chạy: python webapp/app.py   (từ thư mục starter-code/)
Mở trình duyệt: http://127.0.0.1:5000
"""

import json
import os
import sys
import time

from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../starter-code/webapp
STARTER_DIR = os.path.dirname(BASE_DIR)                         # .../starter-code
PROJECT_ROOT = os.path.dirname(STARTER_DIR)                     # project root
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw-data")

if STARTER_DIR not in sys.path:
    sys.path.insert(0, STARTER_DIR)

import template as rule_engine   # noqa: E402  (ChatbotBaseline / ReActAgent rule-based)

try:
    import templateAI as ai_engine   # noqa: E402  (bản gọi Gemini API thật)
    AI_AVAILABLE = True
    AI_IMPORT_ERROR = None
except Exception as exc:  # thiếu package (openai/dotenv) hoặc thiếu API key khi import
    ai_engine = None
    AI_AVAILABLE = False
    AI_IMPORT_ERROR = str(exc)

app = Flask(__name__)


def load_json(filename: str):
    path = os.path.join(RAW_DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    return jsonify({
        "flights": load_json("flight_data.json"),
        "weather": load_json("weather_data.json"),
        "sample_queries": load_json("customer_queries.json"),
        "ai_available": AI_AVAILABLE,
        "ai_import_error": AI_IMPORT_ERROR,
        "react_model_chain": getattr(ai_engine, "DEFAULT_MODEL_CHAIN", None),
        "baseline_model_chain": getattr(ai_engine, "DEFAULT_MINI_MODEL_CHAIN", None),
    })


@app.route("/api/system-prompt")
def api_system_prompt():
    mode = request.args.get("mode", "react_ai")
    if not AI_AVAILABLE:
        return jsonify({"error": f"Chưa sẵn sàng gọi Gemini API: {AI_IMPORT_ERROR}"}), 503
    try:
        prompt = ai_engine.get_default_system_prompt(mode)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"mode": mode, "system_prompt": prompt})


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True, silent=True) or {}
    mode = payload.get("mode")
    query = (payload.get("query") or "").strip()
    custom_system_prompt = (payload.get("system_prompt") or "").strip() or None
    try:
        max_iterations = int(payload.get("max_iterations") or 5)
    except (TypeError, ValueError):
        max_iterations = 5
    max_iterations = max(1, min(max_iterations, 10))

    if not query:
        return jsonify({"error": "Vui lòng nhập câu hỏi của khách hàng."}), 400

    if mode in ("baseline_ai", "react_ai") and not AI_AVAILABLE:
        return jsonify({
            "error": f"Chưa sẵn sàng gọi Gemini API: {AI_IMPORT_ERROR}"
        }), 503

    started = time.time()
    try:
        if mode == "baseline_rule":
            result = rule_engine.ChatbotBaseline().query(query)
        elif mode == "baseline_ai":
            result = ai_engine.ChatbotBaseline(system_prompt=custom_system_prompt).query(query)
        elif mode == "react_rule":
            result = rule_engine.ReActAgent(max_iterations=max_iterations).run(query)
        elif mode == "react_ai":
            result = ai_engine.ReActAgent(
                max_iterations=max_iterations, system_prompt=custom_system_prompt
            ).run(query)
        else:
            return jsonify({"error": f"mode không hợp lệ: {mode}"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    result["_elapsed_ms"] = int((time.time() - started) * 1000)
    result["_mode"] = mode
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
