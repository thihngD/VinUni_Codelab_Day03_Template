"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import re
import sys
from tools import TOOL_MAP

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

# Các mã sân bay/thành phố mà hệ thống hỗ trợ tra cứu.
KNOWN_CITY_CODES = ["HAN", "SGN", "DAD"]

FLIGHT_KEYWORDS = ["chuyến bay", "vé máy bay", "vé", "bay từ"]
WEATHER_KEYWORDS = ["thời tiết", "mặc gì", "nhiệt độ"]


def extract_airport_codes(text: str):
    """Trả về danh sách mã sân bay xuất hiện trong câu, theo đúng thứ tự xuất hiện."""
    pattern = r"\b(" + "|".join(KNOWN_CITY_CODES) + r")\b"
    return re.findall(pattern, text)


def extract_max_price(text: str):
    """Trích xuất giá tối đa (VND) từ cụm 'dưới X triệu'. Trả về None nếu không tìm thấy."""
    match = re.search(r"dưới\s+([\d]+(?:[.,]\d+)?)\s*tri[eệ]u", text, re.IGNORECASE)
    if not match:
        return None
    number_str = match.group(1).replace(",", ".")
    return int(float(number_str) * 1_000_000)


def detect_intents(text: str):
    """Phát hiện nhu cầu sử dụng tool dựa trên từ khóa trong câu hỏi."""
    lowered = text.lower()
    needs_flight = any(keyword in lowered for keyword in FLIGHT_KEYWORDS)
    needs_weather = any(keyword in lowered for keyword in WEATHER_KEYWORDS)
    return {"flight": needs_flight, "weather": needs_weather}


class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def query(self, user_input: str) -> dict:
        # TODO: Trả về câu trả lời tĩnh hoặc gọi LLM 1 lượt (không dùng tool)
        answer = (
            f"Đây là câu trả lời chung chung cho câu hỏi: '{user_input}'. "
            "Chatbot này không tra cứu dữ liệu thực tế (không dùng tool) nên thông tin "
            "có thể không chính xác hoặc bị bịa ra (hallucination)."
        )
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": []
        }


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    def _faq_answer(self, user_input: str) -> str:
        return (
            "Theo chính sách của Vinpearl, khách hàng có thể đổi hoặc trả vé máy bay tùy theo "
            "hạng vé đã mua, thường sẽ phát sinh phí đổi/trả và cần thực hiện trước giờ khởi hành "
            "tối thiểu 24 giờ. Quý khách vui lòng liên hệ tổng đài chăm sóc khách hàng của Vinpearl "
            "để được hỗ trợ chi tiết theo từng loại vé."
        )

    def _build_answer(self, collected: dict) -> str:
        parts = []

        if "flight" in collected:
            flights = collected["flight"]
            if flights:
                flight_desc = "; ".join(
                    f"{fl['flight_number']} ({fl['airline']}, khởi hành {fl['departure_time']}, "
                    f"giá {fl['price_vnd']:,} VND)"
                    for fl in flights
                )
                parts.append(f"Các chuyến bay phù hợp: {flight_desc}.")
            else:
                parts.append("Không tìm thấy chuyến bay nào phù hợp với yêu cầu của bạn.")

        if "weather" in collected:
            weather = collected["weather"]
            if "error" in weather:
                parts.append(f"Không tìm thấy thông tin thời tiết: {weather['error']}.")
            else:
                parts.append(
                    f"Thời tiết tại {weather['city']}: {weather['temperature_c']}°C, "
                    f"{weather['condition']}. Gợi ý trang phục: {weather['recommendation']}"
                )

        return " ".join(parts)

    def run(self, user_input: str) -> dict:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []

        codes = extract_airport_codes(user_input)
        intents = detect_intents(user_input)

        needs_flight = intents["flight"] and len(codes) >= 1
        needs_weather = intents["weather"] and len(codes) >= 1

        required_steps = []
        if needs_flight:
            required_steps.append("flight")
        if needs_weather:
            required_steps.append("weather")

        total_steps = len(required_steps)

        # Câu hỏi không cần tra cứu tool (VD: FAQ chính sách) -> trả lời trực tiếp.
        if total_steps == 0:
            answer = self._faq_answer(user_input)
            self.trace.append({
                "iteration": 1,
                "thought": "Câu hỏi này không cần dữ liệu chuyến bay/thời tiết, có thể trả lời trực tiếp.",
                "action": None,
                "observation": None
            })
            return {
                "status": "completed",
                "iterations": 1,
                "trace": self.trace,
                "answer": answer
            }

        origin = codes[0] if len(codes) > 0 else None
        destination = codes[1] if len(codes) > 1 else None
        weather_city = codes[-1] if codes else None
        max_price = extract_max_price(user_input) or 5_000_000

        collected = {}
        pending = list(required_steps)
        iteration = 0

        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        while iteration < self.max_iterations:
            iteration += 1

            # TODO 3 & TODO 4: Phân tích Thought/Action và thực thi Tool trong TOOL_MAP nếu có Action
            if pending:
                step = pending.pop(0)

                if step == "flight":
                    thought = (
                        f"Cần tìm chuyến bay từ {origin} đến {destination} "
                        f"với giá dưới {max_price:,} VND."
                    )
                    action = {
                        "name": "get_flight_info",
                        "args": {"origin": origin, "destination": destination, "max_price": max_price}
                    }
                    observation = TOOL_MAP["get_flight_info"](**action["args"])
                    collected["flight"] = observation
                else:  # step == "weather"
                    thought = f"Cần tra cứu thời tiết cho mã sân bay {weather_city}."
                    action = {
                        "name": "get_weather_forecast",
                        "args": {"city_code": weather_city}
                    }
                    observation = TOOL_MAP["get_weather_forecast"](**action["args"])
                    collected["weather"] = observation

                # TODO 5: Ghi lại Observation và lặp lại cho tới khi ra Final Answer
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "observation": observation
                })

                # Nếu chỉ cần đúng 1 tool để trả lời, trả về Final Answer ngay trong bước này.
                if not pending and total_steps == 1:
                    return {
                        "status": "completed",
                        "iterations": iteration,
                        "trace": self.trace,
                        "answer": self._build_answer(collected)
                    }

                continue

            # Đã thu thập đủ dữ liệu từ nhiều tool -> dùng thêm 1 bước để tổng hợp Final Answer.
            self.trace.append({
                "iteration": iteration,
                "thought": "Đã thu thập đủ thông tin từ các công cụ, tổng hợp câu trả lời cuối cùng.",
                "action": None,
                "observation": None
            })
            return {
                "status": "completed",
                "iterations": iteration,
                "trace": self.trace,
                "answer": self._build_answer(collected)
            }

        # Safeguard: đạt số vòng lặp tối đa mà chưa có Final Answer.
        return {
            "status": "max_iterations_reached",
            "iterations": iteration,
            "trace": self.trace,
            "answer": "Không thể hoàn thành yêu cầu trong số bước tối đa cho phép."
        }


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(json.dumps(chatbot.query(user_query), indent=2, ensure_ascii=False))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
