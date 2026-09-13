"""
Lab #3 (bản AI thật): Baseline Chatbot vs ReAct Agent gọi Gemini API.

Khác với template.py (rule-based, dùng để pass autograder ổn định), file này
gọi thật một LLM (Gemini, qua endpoint tương thích OpenAI) để LLM tự sinh
Thought/Action theo đúng SYSTEM_PROMPT, rồi code tự parse Action JSON, thực
thi tool trong TOOL_MAP, đưa Observation quay lại cho LLM, lặp lại tới khi có
Final Answer hoặc chạm max_iterations.

Vì câu trả lời do LLM tự sinh, output KHÔNG đảm bảo cố định 100% giữa các lần
chạy -> không dùng file này để autograde (autograder vẫn trỏ vào template.py).
"""

import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_DEFINITIONS, TOOL_MAP

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# .env thật (chứa OPENAI_API_KEY / OPENAI_BASE_URL / LAB_MODEL) nằm ở thư mục gốc
# project, không phải trong starter-code/.
_ROOT_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_ROOT_ENV)
load_dotenv()  # fallback: .env cục bộ nếu có (không override biến đã load ở trên)

# Free tier của Gemini API giới hạn quota theo TỪNG model/ngày (VD: 20 request/ngày/model),
# nên khi 1 model bị 429 thì đổi sang model khác là cách fix hợp lý nhất (không phải retry
# cùng model). Chain dưới đây được thử lần lượt từ mạnh nhất tới nhẹ nhất.
PRO_MODEL = os.environ.get("LAB_PRO_MODEL", "gemini-pro-latest")
FLASH_MODEL = os.environ.get("LAB_MODEL", "gemini-flash-latest")
FLASH_LITE_MODEL = os.environ.get("LAB_FLASH_LITE_MODEL", "gemini-flash-lite-latest")
GEMMA_MODEL = os.environ.get("LAB_GEMMA_MODEL", "gemma-4-26b-a4b-it")

DEFAULT_MODEL_CHAIN = [PRO_MODEL, FLASH_MODEL, FLASH_LITE_MODEL, GEMMA_MODEL]
DEFAULT_MINI_MODEL_CHAIN = [FLASH_MODEL, FLASH_LITE_MODEL, GEMMA_MODEL]

# Giữ lại 2 tên biến cũ để tương thích ngược (VD: webapp/app.py đọc DEFAULT_MODEL/FALLBACK_MODEL).
DEFAULT_MODEL = PRO_MODEL
DEFAULT_MINI_MODEL = FLASH_MODEL
FALLBACK_MODEL = FLASH_MODEL


def _default_client() -> OpenAI:
    # max_retries=0: tự quản lý retry/fallback ở chat_completion_with_chain(),
    # tránh chồng thêm retry mặc định của SDK khiến 1 lượt gọi bị treo rất lâu.
    return OpenAI(max_retries=0)


def _error_kind(exc: Exception) -> str:
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
        return "quota"
    if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower():
        return "overloaded"
    return "other"


def _extract_retry_seconds(exc: Exception, default: float = 2.0) -> float:
    match = re.search(r"retry in\s*([\d.]+)\s*s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"retrydelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else default


def _clean_model_output(text: str) -> str:
    """Một vài model (VD: Gemma) leak khối suy luận <thought>...</thought> ra ngoài câu trả lời."""
    return re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()


def chat_completion_with_chain(client: OpenAI, messages, model_chain,
                                temperature: float = 0, overload_retries: int = 1,
                                overload_wait_cap: float = 6.0):
    """
    Thử lần lượt từng model trong `model_chain` (VD: [Pro, Flash, Flash-Lite, Gemma]).
    - Lỗi quota (429 / RESOURCE_EXHAUSTED): bỏ qua ngay, thử model tiếp theo trong chain
      (vì quota free-tier tính riêng theo từng model/ngày, không phải đợi rồi retry cùng model).
    - Lỗi quá tải (503 / UNAVAILABLE): thử lại đúng model đó tối đa `overload_retries` lần
      với backoff ngắn (đọc "retry in Xs" nếu server trả về), rồi mới chuyển model kế tiếp.
    Trả về (nội_dung_trả_lời, model_thực_sự_đã_dùng, index_của_model_trong_chain).
    Raise RuntimeError với thông báo rõ ràng nếu TẤT CẢ model trong chain đều thất bại.
    """
    last_exc = None
    for idx, model in enumerate(model_chain):
        attempt = 0
        while True:
            try:
                kwargs = {"model": model, "messages": messages, "temperature": temperature,
                          "max_tokens": 1024}
                if "gemini" in model.lower():
                    # Không giới hạn "thinking" thì model có thể tiêu hết token budget vào suy
                    # luận nội bộ và trả về content rỗng (finish_reason=None) — đặc biệt với
                    # các model flash-lite khi hội thoại đã dài (nhiều lượt Observation).
                    kwargs["reasoning_effort"] = "low"
                response = client.chat.completions.create(**kwargs)
                content = _clean_model_output(response.choices[0].message.content or "")
                if not content and response.choices[0].finish_reason is None:
                    raise RuntimeError(f"Model '{model}' trả về nội dung rỗng (hết token budget cho 'thinking').")
                return content, model, idx
            except Exception as exc:
                last_exc = exc
                if _error_kind(exc) == "overloaded" and attempt < overload_retries:
                    time.sleep(min(_extract_retry_seconds(exc), overload_wait_cap))
                    attempt += 1
                    continue
                break  # quota hết hoặc hết lượt retry overloaded -> sang model kế tiếp

    raise RuntimeError(
        f"Tất cả model đều lỗi (đã thử: {', '.join(model_chain)}). "
        f"Lỗi gần nhất: {last_exc}"
    )


# Placeholder {tools} được thay bằng str.replace() (không dùng .format()) để người dùng có thể
# tự sửa prompt trong UI mà không phải lo escape dấu {{ }} trong các ví dụ JSON bên dưới.
SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc, PHẢI theo đúng định dạng dưới đây (không thêm markdown, không thêm giải thích ngoài format):
Thought: <Suy nghĩ bước tiếp theo>
Action: {"name": "<tên tool>", "args": {<tham số>}}

Hoặc khi đã đủ dữ liệu để trả lời khách hàng:
Thought: <Suy nghĩ bước tiếp theo>
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>

Quy tắc:
- Mỗi lượt chỉ được chọn MỘT trong hai: hoặc "Action", hoặc "Final Answer". Không viết cả hai trong cùng một lượt.
- "Action" phải là một chuỗi JSON hợp lệ duy nhất trên các dòng theo sau, đúng khoá "name" và "args".
- Nếu một tool trả về lỗi (VD: {"error": ...}) 2 lần liên tiếp, đừng gọi lại tool đó nữa — hãy đưa ra Final Answer báo lỗi rõ ràng cho khách hàng.
- Không tự bịa dữ liệu chuyến bay/thời tiết; mọi số liệu phải lấy từ Observation của tool.
"""


def get_default_system_prompt(mode: str) -> str:
    """Trả về system prompt mặc định (đã điền sẵn {tools}) cho 1 chế độ AI, để UI hiển thị/tải lại."""
    if mode == "react_ai":
        return SYSTEM_PROMPT.replace("{tools}", format_tools_for_prompt(TOOL_DEFINITIONS))
    if mode == "baseline_ai":
        return BASELINE_SYSTEM_PROMPT
    raise ValueError(f"Không có system prompt mặc định cho mode '{mode}'.")

BASELINE_SYSTEM_PROMPT = (
    "Bạn là một trợ lý ảo hỗ trợ khách hàng Vingroup. Bạn KHÔNG có quyền truy cập "
    "bất kỳ cơ sở dữ liệu chuyến bay hay thời tiết thực tế nào — chỉ được trả lời "
    "dựa trên kiến thức chung, không được gọi tool."
)


def format_tools_for_prompt(tool_definitions) -> str:
    lines = []
    for tool in tool_definitions:
        params = ", ".join(tool["parameters"].keys())
        lines.append(f"- {tool['name']}({params}): {tool['description']}")
    return "\n".join(lines)


def extract_after_marker(text: str, marker: str, stop_markers=None):
    """Lấy phần text ngay sau `marker`, cắt tại marker dừng gần nhất nếu có."""
    idx = text.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    end = len(text)
    for stop in stop_markers or []:
        pos = text.find(stop, start)
        if pos != -1:
            end = min(end, pos)
    return text[start:end].strip()


def extract_action(text: str):
    """
    Tìm cụm 'Action:' và bóc tách JSON object theo sau bằng cách đếm dấu ngoặc
    (bracket matching) thay vì regex, vì JSON có thể chứa dấu ngoặc lồng nhau.
    Trả về (raw_json_str, parsed_dict_hoặc_None).
    """
    idx = text.find("Action:")
    if idx == -1:
        return None, None

    brace_start = text.find("{", idx)
    if brace_start == -1:
        return None, None

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                raw = text[brace_start:i + 1]
                try:
                    return raw, json.loads(raw)
                except json.JSONDecodeError:
                    return raw, None
    # Không tìm thấy dấu đóng ngoặc khớp -> JSON không hợp lệ/bị cắt.
    return text[brace_start:], None


class ChatbotBaseline:
    """Baseline: gọi LLM một lượt duy nhất, KHÔNG dùng ReAct loop hay tool."""

    def __init__(self, model_chain=None, client: OpenAI = None, system_prompt: str = None):
        self.model_chain = list(model_chain) if model_chain else list(DEFAULT_MINI_MODEL_CHAIN)
        self.client = client or _default_client()
        self.system_prompt = system_prompt if system_prompt else BASELINE_SYSTEM_PROMPT

    def query(self, user_input: str) -> dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        answer, model_used, idx = chat_completion_with_chain(
            self.client, messages, self.model_chain
        )
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],
            "model": model_used,
            "model_fallback": idx > 0,
        }


class ReActAgent:
    """ReAct Agent gọi LLM thật để tự sinh Thought/Action theo SYSTEM_PROMPT."""

    def __init__(self, max_iterations: int = 5, model_chain=None, client: OpenAI = None,
                 system_prompt: str = None):
        self.max_iterations = max_iterations
        self.model_chain = list(model_chain) if model_chain else list(DEFAULT_MODEL_CHAIN)
        self.client = client or _default_client()
        # Template có thể chứa hoặc không chứa placeholder {tools} (người dùng có thể xoá nó
        # trong UI nếu muốn tự liệt kê tool theo cách khác).
        self.system_prompt_template = system_prompt if system_prompt else SYSTEM_PROMPT
        self.trace = []
        self.used_fallback = False
        self.last_model_used = self.model_chain[0]

    def _call_llm(self, messages) -> str:
        content, model_used, idx = chat_completion_with_chain(
            self.client, messages, self.model_chain
        )
        self.last_model_used = model_used
        self.used_fallback = self.used_fallback or idx > 0
        return content

    def run(self, user_input: str) -> dict:
        self.trace = []
        self.used_fallback = False
        self.last_model_used = self.model_chain[0]

        tools_desc = format_tools_for_prompt(TOOL_DEFINITIONS)
        system_content = self.system_prompt_template.replace("{tools}", tools_desc)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input},
        ]

        consecutive_tool_errors = 0
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            raw_text = self._call_llm(messages)
            messages.append({"role": "assistant", "content": raw_text})

            final_pos = raw_text.find("Final Answer:")
            action_pos = raw_text.find("Action:")
            has_final = final_pos != -1
            has_action = action_pos != -1 and (not has_final or action_pos < final_pos)

            thought = extract_after_marker(
                raw_text, "Thought:", stop_markers=["Action:", "Final Answer:"]
            )

            if has_action:
                action_raw, action_obj = extract_action(raw_text)

                # Trap 2: Action không phải JSON hợp lệ.
                if action_obj is None:
                    observation_text = "Invalid JSON format. Action phải là JSON hợp lệ dạng {\"name\": ..., \"args\": {...}}."
                    self.trace.append({
                        "iteration": iteration,
                        "thought": thought,
                        "action_raw": action_raw,
                        "action": None,
                        "observation": observation_text,
                    })
                    messages.append({"role": "user", "content": f"Observation: {observation_text}"})
                    continue

                # Trap 1: chuẩn hoá tên tool (khoảng trắng / viết hoa).
                tool_name = str(action_obj.get("name", "")).strip().lower()
                tool_args = action_obj.get("args", {}) or {}
                tool_fn = TOOL_MAP.get(tool_name)

                if tool_fn is None:
                    observation = {"error": f"Không tìm thấy tool '{tool_name}' trong TOOL_MAP."}
                else:
                    try:
                        observation = tool_fn(**tool_args)
                    except Exception as exc:  # tool raise lỗi bất ngờ (sai tham số...)
                        observation = {"error": str(exc)}

                is_error = isinstance(observation, dict) and "error" in observation
                consecutive_tool_errors = consecutive_tool_errors + 1 if is_error else 0

                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": {"name": tool_name, "args": tool_args},
                    "observation": observation,
                })

                observation_text = (
                    observation if isinstance(observation, str)
                    else json.dumps(observation, ensure_ascii=False)
                )

                # Trap 3: lỗi 2 lần liên tiếp -> nhắc LLM dừng lại và trả lời Final Answer báo lỗi.
                nudge = ""
                if consecutive_tool_errors >= 2:
                    nudge = (
                        " (Đã lỗi 2 lần liên tiếp với tool này — theo quy tắc, hãy đưa ra "
                        "Final Answer báo lỗi cho khách hàng ngay, KHÔNG gọi lại tool này nữa.)"
                    )

                messages.append({"role": "user", "content": f"Observation: {observation_text}{nudge}"})
                continue

            if has_final:
                answer = extract_after_marker(raw_text, "Final Answer:") or raw_text.strip()
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": None,
                    "observation": None,
                })
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "trace": self.trace,
                    "answer": answer,
                    "model": self.last_model_used,
                    "model_fallback": self.used_fallback,
                }

            # LLM trả về sai định dạng (không có cả Action lẫn Final Answer) -> nhắc lại format.
            self.trace.append({
                "iteration": iteration,
                "thought": raw_text.strip(),
                "action": None,
                "observation": "format_error",
            })
            messages.append({
                "role": "user",
                "content": (
                    "Observation: Định dạng phản hồi không hợp lệ. Vui lòng chỉ trả lời theo "
                    "đúng format Thought/Action hoặc Thought/Final Answer như hướng dẫn."
                ),
            })

        return {
            "status": "max_iterations_reached",
            "iterations": iteration,
            "trace": self.trace,
            "answer": "Không thể hoàn thành trong số bước tối đa cho phép.",
            "model": self.last_model_used,
            "model_fallback": self.used_fallback,
        }


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Thiếu OPENAI_API_KEY trong biến môi trường (kiểm tra file .env ở thư mục gốc project)."
        )
        return

    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE (Gemini, không dùng tool) ===")
    try:
        chatbot = ChatbotBaseline()
        print(json.dumps(chatbot.query(user_query), indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"[Lỗi khi gọi LLM baseline]: {exc}")

    print("\n=== RUNNING REACT AGENT (Gemini tự sinh Thought/Action) ===")
    try:
        agent = ReActAgent(max_iterations=5)
        result = agent.run(user_query)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"[Lỗi khi gọi LLM ReAct Agent]: {exc}")


if __name__ == "__main__":
    main()
