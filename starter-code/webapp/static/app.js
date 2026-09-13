const state = {
  sampleQueries: [],
  systemPromptCache: {},   // mode -> nội dung system prompt hiện tại (kể cả đã chỉnh sửa) của mode đó
  currentMode: "baseline_rule",
};

function $(sel) { return document.querySelector(sel); }
function $all(sel) { return Array.from(document.querySelectorAll(sel)); }
function isAiMode(mode) { return mode === "baseline_ai" || mode === "react_ai"; }

function fmtPrice(n) {
  return new Intl.NumberFormat("vi-VN").format(n);
}

// ---------- Tabs ----------
$all(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    $all(".tab-btn").forEach(b => b.classList.remove("active"));
    $all(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ---------- Load project data ----------
async function loadData() {
  const res = await fetch("/api/data");
  const data = await res.json();
  state.sampleQueries = data.sample_queries;

  const flightsBody = $("#flights-table tbody");
  flightsBody.innerHTML = data.flights.map(f => `
    <tr>
      <td><strong>${f.flight_number}</strong></td>
      <td>${f.airline}</td>
      <td>${f.origin} → ${f.destination}</td>
      <td>${f.departure_time}</td>
      <td>${fmtPrice(f.price_vnd)}</td>
    </tr>
  `).join("");

  const weatherBody = $("#weather-table tbody");
  weatherBody.innerHTML = Object.entries(data.weather).map(([code, w]) => `
    <tr>
      <td><strong>${code}</strong></td>
      <td>${w.city}</td>
      <td>${w.temperature_c}°C</td>
      <td>${w.condition}</td>
      <td>${w.humidity_pct}%</td>
      <td>${w.recommendation}</td>
    </tr>
  `).join("");

  $("#queries-list").innerHTML = data.sample_queries.map(q => `
    <div class="query-item">
      <div><span class="qid">${q.id}</span>${q.query}</div>
      <div class="qtags">${q.category} · tools: ${q.expected_tools.join(", ") || "none"}</div>
    </div>
  `).join("");

  $("#sample-chips").innerHTML = data.sample_queries.map(q =>
    `<button class="chip" data-query="${encodeURIComponent(q.query)}">${q.id}: ${q.query.slice(0, 28)}...</button>`
  ).join("");

  $all(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      $("#query-input").value = decodeURIComponent(chip.dataset.query);
    });
  });

  if (!data.ai_available) {
    $all('input[value="baseline_ai"], input[value="react_ai"]').forEach(input => {
      input.disabled = true;
      input.closest(".mode-card").title = "Chưa cấu hình được Gemini API: " + data.ai_import_error;
      input.closest(".mode-card").style.opacity = 0.5;
    });
  } else {
    const proNote = document.createElement("p");
    proNote.style.cssText = "color:var(--text-dim);font-size:12px;margin-top:-8px;";
    proNote.textContent = `Thứ tự model ReAct: ${data.react_model_chain.join(" → ")} (tự chuyển sang model kế tiếp nếu model trước hết quota/lỗi)`;
    $("#tab-agent .card").insertBefore(proNote, $("#max-iter-wrap"));
  }
}

// ---------- System prompt: tải mặc định / cho phép chỉnh sửa theo từng mode ----------
async function fetchDefaultPrompt(mode) {
  if (!isAiMode(mode)) return;
  const textarea = $("#system-prompt-input");
  try {
    const res = await fetch(`/api/system-prompt?mode=${encodeURIComponent(mode)}`);
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || "Không tải được system prompt mặc định.");
      return;
    }
    state.systemPromptCache[mode] = data.system_prompt;
    textarea.value = data.system_prompt;
  } catch (err) {
    alert("Lỗi kết nối khi tải system prompt: " + err.message);
  }
}

function updateSystemPromptUI(mode) {
  const textarea = $("#system-prompt-input");
  const hint = $("#system-prompt-hint");
  const loadBtn = $("#load-default-prompt");

  if (!isAiMode(mode)) {
    textarea.disabled = true;
    loadBtn.disabled = true;
    hint.textContent = "(không áp dụng cho chế độ rule-based)";
    return;
  }

  textarea.disabled = false;
  loadBtn.disabled = false;
  hint.textContent = "(có thể chỉnh sửa trước khi Chạy)";

  if (state.systemPromptCache[mode] !== undefined) {
    textarea.value = state.systemPromptCache[mode];
  } else {
    textarea.value = "";
    fetchDefaultPrompt(mode);
  }
}

$("#load-default-prompt").addEventListener("click", () => fetchDefaultPrompt(state.currentMode));

// ---------- Mode toggle: show max_iterations only for ReAct modes ----------
$all('input[name="mode"]').forEach(radio => {
  radio.addEventListener("change", () => {
    if (!radio.checked) return;

    // Lưu lại nội dung đang chỉnh sửa của mode trước đó trước khi chuyển sang mode mới.
    if (isAiMode(state.currentMode)) {
      state.systemPromptCache[state.currentMode] = $("#system-prompt-input").value;
    }
    state.currentMode = radio.value;

    $("#max-iter-wrap").hidden = !radio.value.startsWith("react_");
    updateSystemPromptUI(radio.value);
  });
});

updateSystemPromptUI(state.currentMode);

// ---------- Run agent ----------
function statusTag(status) {
  return `<span class="tag status-${status}">${status}</span>`;
}

function renderTrace(trace) {
  if (!trace || !trace.length) {
    $("#trace-title").hidden = true;
    $("#trace-list").innerHTML = "";
    return;
  }
  $("#trace-title").hidden = false;
  $("#trace-list").innerHTML = trace.map(step => {
    const actionLine = step.action
      ? `<div class="action-line">Action: ${step.action.name}(${JSON.stringify(step.action.args)})</div>`
      : "";
    const obsValue = step.observation === null || step.observation === undefined
      ? ""
      : (typeof step.observation === "string" ? step.observation : JSON.stringify(step.observation, null, 2));
    const obsLine = obsValue ? `<div class="observation">Observation: ${obsValue}</div>` : "";
    return `
      <div class="trace-step">
        <div class="step-head"><span>Iteration ${step.iteration}</span></div>
        ${step.thought ? `<div class="thought">💭 ${step.thought}</div>` : ""}
        ${actionLine}
        ${obsLine}
      </div>
    `;
  }).join("");
}

async function runAgent() {
  const mode = $('input[name="mode"]:checked').value;
  const query = $("#query-input").value.trim();
  const maxIterations = parseInt($("#max-iter").value || "5", 10);

  if (!query) {
    alert("Vui lòng nhập câu hỏi.");
    return;
  }

  const btn = $("#run-btn");
  btn.disabled = true;
  $("#run-status").textContent = "Đang xử lý...";
  $("#result-card").hidden = false;
  $("#result-error").hidden = true;
  $("#result-answer").textContent = "";
  $("#trace-list").innerHTML = "";
  $("#trace-title").hidden = true;
  $("#result-meta").innerHTML = "";

  const systemPrompt = isAiMode(mode) ? $("#system-prompt-input").value : undefined;

  const started = performance.now();
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, query, max_iterations: maxIterations, system_prompt: systemPrompt }),
    });
    const data = await res.json();

    if (!res.ok) {
      $("#result-error").hidden = false;
      $("#result-error").textContent = data.error || "Đã có lỗi xảy ra.";
      return;
    }

    const metaTags = [];
    if (data.status) metaTags.push(statusTag(data.status));
    if (data.iterations !== undefined) metaTags.push(`<span class="tag">iterations: ${data.iterations}</span>`);
    if (data.model) metaTags.push(`<span class="tag">model: ${data.model}${data.model_fallback ? " (fallback)" : ""}</span>`);
    if (data._elapsed_ms !== undefined) metaTags.push(`<span class="tag">${data._elapsed_ms} ms</span>`);
    $("#result-meta").innerHTML = metaTags.join("");

    $("#result-answer").textContent = data.answer || "(không có câu trả lời)";
    renderTrace(data.trace);
  } catch (err) {
    $("#result-error").hidden = false;
    $("#result-error").textContent = "Lỗi kết nối tới server: " + err.message;
  } finally {
    btn.disabled = false;
    $("#run-status").textContent = "";
  }
}

$("#run-btn").addEventListener("click", runAgent);

loadData();
