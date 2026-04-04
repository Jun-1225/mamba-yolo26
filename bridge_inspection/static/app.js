/**
 * 桥梁病害检测前端 · 全息控制台
 */
const $ = (sel) => document.querySelector(sel);

/** 与 app.py 中 _DEFECT_NAME_EN_TO_ZH 一致，用于分布区展示与兜底 */
const DEFECT_NAME_EN_TO_ZH = {
  "alligator crack": "龟裂（网状裂缝）",
  bearing: "支座",
  cavity: "孔洞",
  crack: "裂缝",
  drainage: "排水设施",
  efflorescence: "泛碱（白华）",
  "expansion joint": "伸缩缝",
  "exposed rebars": "钢筋外露",
  graffiti: "涂鸦",
  hollowareas: "空洞区",
  "hollow areas": "空洞区",
  "joint tape": "接缝胶带",
  "protective equipment": "防护设施",
  restformwork: "模板残留",
  "rest formwork": "模板残留",
  rockpocket: "石子窝槽（蜂窝）",
  "rock pocket": "石子窝槽（蜂窝）",
  rust: "锈蚀",
  spalling: "剥落",
  "washouts/concrete corrosion": "冲刷与混凝土腐蚀",
  washouts: "冲刷",
  "concrete corrosion": "混凝土腐蚀",
  weathering: "风化",
  wetspot: "渗水湿渍",
  "wet spot": "渗水湿渍",
};

const DONUT_COLORS = ["#00f5ff", "#ff2bd6", "#ffd60a", "#00c4ee", "#ff55e0", "#a78bfa", "#34d399"];

function defectLabelZh(raw) {
  const s = String(raw ?? "").trim();
  if (!s) return s;
  const k = s.toLowerCase().replace(/_/g, " ").replace(/-/g, " ");
  if (DEFECT_NAME_EN_TO_ZH[k] != null) return DEFECT_NAME_EN_TO_ZH[k];
  const compact = k.replace(/\s+/g, "");
  for (const [en, zh] of Object.entries(DEFECT_NAME_EN_TO_ZH)) {
    if (en.replace(/\s+/g, "") === compact) return zh;
  }
  return s;
}

function escapeHtml(text) {
  const s = String(text);
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function donutWedge(cx, cy, r0, r1, a0, a1) {
  const x0o = cx + r1 * Math.cos(a0);
  const y0o = cy + r1 * Math.sin(a0);
  const x1o = cx + r1 * Math.cos(a1);
  const y1o = cy + r1 * Math.sin(a1);
  const x0i = cx + r0 * Math.cos(a1);
  const y0i = cy + r0 * Math.sin(a1);
  const x1i = cx + r0 * Math.cos(a0);
  const y1i = cy + r0 * Math.sin(a0);
  const large = a1 - a0 > Math.PI ? 1 : 0;
  return `M ${x0o} ${y0o} A ${r1} ${r1} 0 ${large} 1 ${x1o} ${y1o} L ${x0i} ${y0i} A ${r0} ${r0} 0 ${large} 0 ${x1i} ${y1i} Z`;
}

/** 霓虹环形图 SVG */
function buildNeonDonutSvg(entries, total) {
  const cx = 70;
  const cy = 70;
  const rOuter = 46;
  const rInner = 26;
  let angle = -Math.PI / 2;
  const paths = [];
  for (let i = 0; i < entries.length; i++) {
    const n = Number(entries[i][1]);
    const frac = total > 0 && Number.isFinite(n) ? n / total : 0;
    if (frac <= 0) continue;
    const da = frac * 2 * Math.PI;
    const a1 = angle + da;
    const d = donutWedge(cx, cy, rInner, rOuter, angle, a1);
    const col = DONUT_COLORS[i % DONUT_COLORS.length];
    paths.push(
      `<path d="${d}" fill="${col}" fill-opacity="0.88" stroke="rgba(255,255,255,0.25)" stroke-width="0.5" />`
    );
    angle = a1;
  }
  if (!paths.length) return "";
  return `<svg class="neon-donut" viewBox="0 0 140 140" width="140" height="140" aria-hidden="true">${paths.join("")}</svg>`;
}

/** 病害类型分布：霓虹环 + 列表 */
function renderClassDistribution(byClass, el) {
  if (!el) return;
  if (!byClass || !Object.keys(byClass).length) {
    el.className = "donut-placeholder class-dist-empty";
    el.innerHTML = "";
    el.textContent = "等待推理";
    return;
  }
  const entries = Object.entries(byClass).sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1];
    return defectLabelZh(a[0]).localeCompare(defectLabelZh(b[0]), "zh-CN");
  });
  const total = entries.reduce((s, [, v]) => s + Number(v), 0);
  const svg = buildNeonDonutSvg(entries, total);
  const lis = entries
    .map(([k, v]) => {
      const label = escapeHtml(defectLabelZh(k));
      const n = Number(v);
      return `<li><span class="class-dist-name" title="${label}">${label}</span><span class="class-dist-count">${Number.isFinite(n) ? n : v}</span></li>`;
    })
    .join("");
  el.className = "class-dist-wrap";
  el.innerHTML = `<div class="dist-chart-inner">${svg}<ul class="class-dist" role="list">${lis}</ul></div>`;
}

function renderHealthGauge(score) {
  const arc = document.getElementById("healthGaugeArc");
  if (!arc) return;
  const n = Number(score);
  if (score === "—" || !Number.isFinite(n)) {
    arc.style.strokeDashoffset = "100";
    return;
  }
  const s = Math.max(0, Math.min(100, n));
  arc.style.strokeDashoffset = String(100 - s);
}

let lastSegB64 = "";
let lastBoxesB64 = "";

function mountViewportMode(wrap, body, mode) {
  const s = lastSegB64;
  const b = lastBoxesB64;
  if (!body) return;
  if (!s || !b) {
    body.innerHTML = "";
    return;
  }
  if (mode === "split") {
    body.innerHTML = `
      <div class="viewport-dual">
        <figure class="viewport-pane viewport-pane--seg">
          <figcaption>分割掩膜</figcaption>
          <img src="data:image/png;base64,${s}" alt="segmentation" />
        </figure>
        <figure class="viewport-pane viewport-pane--det">
          <figcaption>检测框 · 置信度</figcaption>
          <img src="data:image/png;base64,${b}" alt="detection" />
        </figure>
      </div>`;
  } else if (mode === "overlay") {
    body.innerHTML = `
      <div class="vp-stage">
        <div class="vp-overlay-stack">
          <img class="img-base" src="data:image/png;base64,${s}" alt="segmentation" />
          <img class="img-top" src="data:image/png;base64,${b}" alt="detection" />
        </div>
      </div>`;
    const blend = wrap.querySelector("#vpBlend");
    const v = blend ? Number(blend.value) / 100 : 0.55;
    wrap.style.setProperty("--vp-blend", String(v));
  } else if (mode === "wipe") {
    body.innerHTML = `
      <div class="vp-stage">
        <div class="vp-wipe-wrap">
          <img class="img-base" src="data:image/png;base64,${s}" alt="segmentation" />
          <img class="img-wipe" src="data:image/png;base64,${b}" alt="detection" />
        </div>
      </div>`;
    const w = wrap.querySelector("#vpWipe");
    const pct = w ? Number(w.value) : 50;
    wrap.style.setProperty("--vp-wipe", `${100 - pct}%`);
  }
}

function wireViewportChrome(wrap) {
  const body = wrap.querySelector(".vp-body");
  wrap.querySelectorAll("[data-vp-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      wrap.querySelectorAll("[data-vp-mode]").forEach((x) => x.classList.toggle("active", x === btn));
      const mode = btn.getAttribute("data-vp-mode");
      wrap.querySelectorAll(".vp-ctl").forEach((el) => el.classList.add("vp-ctl-hidden"));
      if (mode === "overlay") wrap.querySelector(".vp-ctl-overlay")?.classList.remove("vp-ctl-hidden");
      if (mode === "wipe") wrap.querySelector(".vp-ctl-wipe")?.classList.remove("vp-ctl-hidden");
      mountViewportMode(wrap, body, mode);
    });
  });
  wrap.querySelector("#vpBlend")?.addEventListener("input", (e) => {
    wrap.style.setProperty("--vp-blend", String(Number(e.target.value) / 100));
  });
  wrap.querySelector("#vpWipe")?.addEventListener("input", (e) => {
    const pct = Number(e.target.value);
    wrap.style.setProperty("--vp-wipe", `${100 - pct}%`);
  });
}

/** 并排 / 叠加 / 相位扫描 */
function showDualResult(segB64, boxesB64) {
  lastSegB64 = segB64 || "";
  lastBoxesB64 = boxesB64 || "";
  const wrap = $("#viewport");
  const s = lastSegB64;
  const b = lastBoxesB64;
  wrap.className = "viewport-outer glass-frame viewport-chrome";
  wrap.style.setProperty("--vp-blend", "0.55");
  wrap.style.setProperty("--vp-wipe", "50%");
  wrap.innerHTML = `
    <div class="compare-toolbar">
      <span class="chrome-label">VIEW</span>
      <button type="button" class="vp-mode active" data-vp-mode="split">双轨分屏</button>
      <button type="button" class="vp-mode" data-vp-mode="overlay">全息叠加</button>
      <button type="button" class="vp-mode" data-vp-mode="wipe">相位扫描</button>
      <label class="instrument-inline vp-ctl vp-ctl-overlay vp-ctl-hidden">融合<input type="range" id="vpBlend" min="0" max="100" value="55" /></label>
      <label class="instrument-inline vp-ctl vp-ctl-wipe vp-ctl-hidden">切面<input type="range" id="vpWipe" min="0" max="100" value="50" /></label>
    </div>
    <div class="vp-body"></div>`;
  const body = wrap.querySelector(".vp-body");
  mountViewportMode(wrap, body, "split");
  wireViewportChrome(wrap);
}

let lastDetections = [];
let lastSummary = null;
let lastMmPerPixel = 0.5;
/** @type {Record<string, unknown> | null} 最近一次展示的 timing 对象（含 inference_ms 等） */
let lastTimingSnapshot = null;
let hasCompletedInference = false;
let lastAssistantReply = "";
let agentBusy = false;
/** @type {{ role: string; content: string; ts: number }[]} */
const chatTranscript = [];

function setExpertLed(mode) {
  const el = $("#aiExpertLed");
  if (!el) return;
  el.classList.remove("expert-led--idle", "expert-led--busy");
  if (mode === "busy") {
    el.classList.add("expert-led--busy");
    el.title = "分析中";
  } else {
    el.classList.add("expert-led--idle");
    el.title = "待命";
  }
}

function scrollChatToBottom() {
  const stream = $("#chatMessages");
  if (stream) stream.scrollTop = stream.scrollHeight;
}

function severityMeta(score) {
  if (!Number.isFinite(score)) return { label: "—", cls: "" };
  if (score >= 80) return { label: "一级 · 优", cls: "sev-1" };
  if (score >= 60) return { label: "二级 · 良", cls: "sev-2" };
  if (score >= 40) return { label: "三级 · 注意", cls: "sev-3" };
  return { label: "四级 · 警戒", cls: "sev-4" };
}

/** 健康指数旁白：说明分数含义（与初筛严重等级四档一致） */
function explainHealthMeaning(score, total) {
  if (!hasCompletedInference) {
    return "完成推理后：根据检出实例总数与高置信（≥60%）数量折算为 0–100 分示意指数；分数越高表示在当前阈值下算法判定的表观病害压力越低。与上方初筛「严重等级」四档一一对应，仅供辅助判读，不能替代规范检测或定级。";
  }
  if (total === 0) {
    return "本帧未检出达到阈值的病害，示意分通常较高：表示「算法未触发明显告警」，不等于结构绝对安全或无损伤，请务必结合原始影像与现场复核。";
  }
  if (score >= 80) {
    return `得分 ${score}（一级 · 优）：检出规模相对可控，高置信实例较少。算法视角下表观状况偏向有利，仍应按制度巡检并保留复核记录。`;
  }
  if (score >= 60) {
    return `得分 ${score}（二级 · 良）：存在一定病害或中等置信实例。建议按表格分类建档，安排现场复核与加密巡检，关注发展趋势。`;
  }
  if (score >= 40) {
    return `得分 ${score}（三级 · 注意）：检出偏多或高置信占比较高，病害压力上升。宜结合专项检测评估耐久与承载，并酌情研究荷载或交通管控。`;
  }
  return `得分 ${score}（四级 · 警戒）：示意指数偏低，算法判定当前帧病害压力较大。应优先组织现场核查、成因分析与应急处置或限载等措施研究。`;
}

function updateBriefing(summary, dets) {
  const elC = $("#briefConclusion");
  const elS = $("#briefSeverity");
  const elOne = $("#briefingOneLiner");
  if (!elC || !elS) return;

  const total = summary?.total ?? dets?.length ?? 0;
  const high = (dets || []).filter((d) => d.confidence >= 0.6).length;
  let score = 100;
  score -= Math.min(40, total * 4);
  score -= Math.min(30, high * 5);
  score = Math.max(0, Math.round(score));
  const meta = severityMeta(score);

  if (!hasCompletedInference) {
    elC.textContent = "等待上传并完成推理后，将在此生成初筛结论文案。";
    elS.textContent = "—";
    elS.className = "severity-pill";
    if (elOne) elOne.textContent = "";
    return;
  }

  let text;
  if (total === 0) {
    text =
      "本轮推理未检出达到阈值的病害实例；若为净空或净空极好场景，仍建议对照原始影像进行人工复核。结构健康指数（示意）仍供参考。";
  } else {
    text = `当前数据：检出 ${total} 处病害实例，其中高置信（≥60%）${high} 处；结构健康指数（示意）${score} 分。本结论为算法初筛，非规范评定。`;
  }
  elC.textContent = text;
  elS.textContent = meta.label;
  elS.className = "severity-pill " + meta.cls;
  if (elOne) elOne.textContent = `${meta.label} · ${text.length > 88 ? text.slice(0, 88) + "…" : text}`;
}

function wireBriefingToggle() {
  const root = $("#reportBriefing");
  const btn = $("#briefingToggle");
  if (!root || !btn) return;
  btn.addEventListener("click", () => {
    const collapsed = root.classList.toggle("report-briefing--collapsed");
    btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
  });
}

function appendUserMessage(text) {
  const stream = $("#chatMessages");
  if (!stream) return;
  const wrap = document.createElement("div");
  wrap.className = "msg msg--user";
  const meta = document.createElement("div");
  meta.className = "msg__meta";
  meta.textContent = "操作员";
  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  bubble.textContent = text;
  wrap.appendChild(meta);
  wrap.appendChild(bubble);
  stream.appendChild(wrap);
  chatTranscript.push({ role: "user", content: text, ts: Date.now() });
  scrollChatToBottom();
}

/** @returns {HTMLDivElement} 内层文本节点容器 */
function appendAssistantShell() {
  const stream = $("#chatMessages");
  const wrap = document.createElement("div");
  wrap.className = "msg msg--assistant";
  const meta = document.createElement("div");
  meta.className = "msg__meta";
  meta.textContent = "AI 专家";
  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  wrap.appendChild(meta);
  wrap.appendChild(bubble);
  stream.appendChild(wrap);
  scrollChatToBottom();
  return bubble;
}

function typewriterReveal(el, fullText, opts = {}) {
  const speed = opts.speed ?? 10;
  const full = String(fullText ?? "");
  const sym = Symbol();
  el._twSym = sym;
  el.textContent = "";
  let i = 0;
  function tick() {
    if (el._twSym !== sym) return;
    if (i >= full.length) {
      el.textContent = full;
      opts.onDone?.();
      return;
    }
    el.textContent = full.slice(0, i + 1) + "▌";
    i += 1;
    scrollChatToBottom();
    setTimeout(tick, speed);
  }
  tick();
}

function releaseAgentUi() {
  agentBusy = false;
  setExpertLed("idle");
  const b = $("#btnSendChat");
  if (b) b.disabled = false;
}

function exportChatTranscript() {
  const lines = [];
  lines.push("# 桥梁病害专家对话导出");
  lines.push("");
  lines.push("## 初筛概览");
  lines.push($("#briefConclusion")?.textContent?.trim() || "—");
  lines.push("");
  lines.push("**严重等级** " + ($("#briefSeverity")?.textContent?.trim() || "—"));
  lines.push("");
  lines.push("## 对话记录");
  for (const m of chatTranscript) {
    const t = new Date(m.ts).toLocaleString("zh-CN");
    lines.push(`### ${m.role} · ${t}`);
    lines.push(m.content);
    lines.push("");
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `bridge-copilot-${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/**
 * @param {string} [rawText]
 * @param {{ skipUserBubble?: boolean }} [opt]
 */
async function sendAgentMessage(rawText, opt = {}) {
  const userQuestion =
    (rawText && String(rawText).trim()) || "请根据当前检测结果，按系统要求的四段标题格式输出完整分析。";
  if (agentBusy) return;
  if (!hasCompletedInference) {
    alert("请先在左侧完成图像或视频推理，再发起专家对话。");
    return;
  }

  agentBusy = true;
  setExpertLed("busy");
  const btn = $("#btnSendChat");
  if (btn) btn.disabled = true;

  if (!opt.skipUserBubble) appendUserMessage(userQuestion);

  const bubbleEl = appendAssistantShell();
  scrollChatToBottom();

  const body = {
    detections: lastDetections,
    summary: lastSummary,
    mm_per_pixel: lastMmPerPixel,
    extra_context: $("#agentCtx").value.trim() || null,
    user_question: userQuestion,
    prior_assistant_reply: lastAssistantReply ? lastAssistantReply.slice(0, 1200) : null,
  };

  try {
    const r = await fetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let j;
    try {
      j = await r.json();
    } catch {
      bubbleEl.textContent = "响应不是 JSON，HTTP " + r.status;
      chatTranscript.push({ role: "assistant", content: bubbleEl.textContent, ts: Date.now() });
      releaseAgentUi();
      return;
    }
    if (!r.ok) {
      const d =
        j.detail != null ? (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)) : JSON.stringify(j);
      bubbleEl.textContent = "请求失败 (" + r.status + "): " + d;
      chatTranscript.push({ role: "assistant", content: bubbleEl.textContent, ts: Date.now() });
      releaseAgentUi();
      return;
    }
    const analysis = j.analysis || "";
    const tm = j.timing_ms != null ? `\n\n（请求耗时 ${j.timing_ms} ms）` : "";
    const full = analysis + tm;
    typewriterReveal(bubbleEl, full, {
      speed: 9,
      onDone: () => {
        lastAssistantReply = full;
        chatTranscript.push({ role: "assistant", content: full, ts: Date.now() });
        releaseAgentUi();
      },
    });
  } catch (e) {
    bubbleEl.textContent = "请求失败: " + e;
    chatTranscript.push({ role: "assistant", content: bubbleEl.textContent, ts: Date.now() });
    releaseAgentUi();
  }
}

function wireChatComposer() {
  const input = $("#chatInput");
  const send = $("#btnSendChat");
  if (!input || !send) return;
  send.addEventListener("click", () => {
    const t = input.value.trim();
    input.value = "";
    sendAgentMessage(t || undefined);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.shiftKey) return;
    e.preventDefault();
    const t = input.value.trim();
    input.value = "";
    sendAgentMessage(t || undefined);
  });
}

function wireActionChips() {
  const root = $("#actionChips");
  if (!root) return;
  root.querySelectorAll("[data-chip]").forEach((chip) => {
    chip.addEventListener("click", () => {
      const kind = chip.getAttribute("data-chip");
      if (kind === "export") {
        exportChatTranscript();
        return;
      }
      if (kind === "repair") {
        sendAgentMessage(
          "请单独强化【建议】部分，给出更具体的修复与养护措施（仍为概括性意见，非设计文件）。"
        );
        return;
      }
      if (kind === "risk") {
        sendAgentMessage(
          "请从结构安全与运营风险角度，基于当前检出结果做简要风险评估；禁止编造未在 detections 中出现的病害。"
        );
      }
    });
  });
}

function formatInferenceTiming(t) {
  if (!t || typeof t !== "object") return "—";
  const parts = [];
  if (t.preprocess_ms != null) parts.push(`预处理 ${t.preprocess_ms} ms`);
  if (t.inference_ms != null) parts.push(`核心推理 ${t.inference_ms} ms`);
  if (t.postprocess_ms != null) parts.push(`后处理 ${t.postprocess_ms} ms`);
  if (t.server_total_ms != null) parts.push(`服务端总计 ${t.server_total_ms} ms`);
  return parts.length ? parts.join(" · ") : "—";
}

function setTimingBar(text) {
  const el = $("#timingBar");
  if (el) el.textContent = text;
}

function setStatus(text, type = "ok") {
  const el = $("#modelStatus");
  el.textContent = text;
  el.className = "status-pill" + (type === "warn" ? " warn" : type === "err" ? " err" : "");
}

async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    let llmHint = "";
    if (j.llm) {
      const prov =
        j.llm.provider === "deepseek"
          ? "DeepSeek"
          : j.llm.provider === "openai"
            ? "OpenAI 兼容"
            : "";
      const tag = prov ? `（${prov}）` : "";
      llmHint = j.llm.configured ? " · 大模型已配置" + tag : " · 大模型未配置（仅规则摘要）";
      if (!j.llm.configured && Array.isArray(j.llm.hints) && j.llm.hints.length) {
        const one = j.llm.hints[0];
        if (one && one.length < 120) llmHint += " — " + one;
      }
    }
    if (j.model_exists) {
      setStatus("模型已就绪" + llmHint);
    } else {
      setStatus("未检测到 ONNX，请放置 weights/best.onnx" + llmHint, "warn");
    }
  } catch {
    setStatus("无法连接后端", "err");
  }
}

function cellPlaceholderOrNumber(val, letter, { asInt = false } = {}) {
  const td = document.createElement("td");
  const n = Number(val);
  const bad = val == null || val === "" || Number.isNaN(n) || n <= 0;
  if (bad) {
    const sp = document.createElement("span");
    sp.className = "sym-placeholder";
    sp.textContent = letter;
    td.appendChild(sp);
  } else {
    td.textContent = asInt ? String(Math.round(n)) : String(val);
    td.className = "mono-digits";
  }
  return td;
}

function renderTable(dets) {
  const tb = $("#detTableBody");
  tb.innerHTML = "";
  if (!dets || !dets.length) {
    tb.innerHTML = `<tr><td colspan="7" class="table-empty">暂无检出</td></tr>`;
    return;
  }
  for (const d of dets) {
    const tr = document.createElement("tr");
    const tdType = document.createElement("td");
    tdType.textContent = d.class_name ?? "";
    const tdConf = document.createElement("td");
    tdConf.className = "mono-digits";
    tdConf.textContent = `${((d.confidence ?? 0) * 100).toFixed(1)}%`;
    const tdId = document.createElement("td");
    tdId.className = "mono-digits";
    tdId.textContent = String(d.id ?? "");
    tr.appendChild(tdType);
    tr.appendChild(tdConf);
    tr.appendChild(cellPlaceholderOrNumber(d.length_mm, "L"));
    tr.appendChild(cellPlaceholderOrNumber(d.max_width_mm, "W"));
    tr.appendChild(cellPlaceholderOrNumber(d.area_px, "A", { asInt: true }));
    const tdSm = document.createElement("td");
    tdSm.className = "cell-sm mono-digits";
    if (d.area_mm2_expr && String(d.area_mm2_expr).trim()) {
      tdSm.textContent = String(d.area_mm2_expr);
      tdSm.title = d.area_source
        ? `A 来源: ${d.area_source}（mask_raster=分割栅格像素；polygon=轮廓填充；bbox=检测框）；ρ²·A=${d.area_mm2_partial ?? "—"}`
        : "";
    } else {
      const sp = document.createElement("span");
      sp.className = "sym-placeholder";
      sp.textContent = "α·ρ²·A";
      tdSm.appendChild(sp);
    }
    tr.appendChild(tdSm);
    tr.appendChild(tdId);
    tb.appendChild(tr);
  }
}

function updateStats(summary, dets) {
  const total = summary?.total ?? dets?.length ?? 0;
  $("#statTotal").textContent = total;
  const high = (dets || []).filter((d) => d.confidence >= 0.6).length;
  $("#statHigh").textContent = high;
  const donut = $("#donutText");
  if (summary?.by_class && Object.keys(summary.by_class).length) {
    renderClassDistribution(summary.by_class, donut);
  } else {
    donut.className = "donut-placeholder";
    donut.innerHTML = "";
    donut.textContent = total ? "详见下方表格" : "等待推理";
  }
  let score = 100;
  score -= Math.min(40, total * 4);
  score -= Math.min(30, high * 5);
  score = Math.max(0, Math.round(score));

  const scoreEl = $("#healthScore");
  const tierEl = $("#healthTier");
  const subEl = $("#healthSub");
  if (!hasCompletedInference) {
    if (scoreEl) scoreEl.textContent = "—";
    if (tierEl) {
      tierEl.textContent = "未评定";
      tierEl.className = "severity-pill health-tier-pill";
    }
    if (subEl) subEl.textContent = explainHealthMeaning(null, total);
    renderHealthGauge("—");
  } else {
    if (scoreEl) scoreEl.textContent = String(score);
    const meta = severityMeta(score);
    if (tierEl) {
      tierEl.textContent = meta.label;
      tierEl.className = "severity-pill health-tier-pill " + meta.cls;
    }
    if (subEl) subEl.textContent = explainHealthMeaning(score, total);
    renderHealthGauge(score);
  }

  updateBriefing(summary, dets);
  updateDerivedMetricsPanel(summary, dets);
  renderRiskPanel(dets);
}

/** 侧栏：由当前检出与 timing 推算的指标 */
function renderRiskPanel(dets) {
  const el = $("#riskPanelBody");
  if (!el) return;
  if (!hasCompletedInference) {
    el.innerHTML =
      '<p class="risk-placeholder">完成推理后，在此列出高置信（≥60%）实例，便于优先复核。</p>';
    return;
  }
  const list = (dets || []).filter((d) => Number(d.confidence) >= 0.6);
  list.sort((a, b) => (Number(b.confidence) || 0) - (Number(a.confidence) || 0));
  if (!list.length) {
    el.innerHTML =
      '<p class="risk-placeholder">当前无置信度 ≥60% 的检出；可调低置信度阈值后重新推理以检视边缘实例。</p>';
    return;
  }
  const rows = list
    .slice(0, 14)
    .map((d) => {
      const name = defectLabelZh(d.class_name ?? "");
      const conf = ((Number(d.confidence) || 0) * 100).toFixed(1);
      const id = d.id != null ? `#${d.id}` : "—";
      return `<li class="risk-list__item"><span class="risk-list__name">${escapeHtml(name)}</span><span class="risk-list__meta mono-digits">${conf}%</span><span class="risk-list__id mono-digits">${escapeHtml(
        id
      )}</span></li>`;
    })
    .join("");
  el.innerHTML = `<ul class="risk-list" role="list">${rows}</ul>`;
}

function updateDerivedMetricsPanel(summary, dets) {
  const elMean = $("#statMeanConf");
  const elKinds = $("#statClassKinds");
  const elRho = $("#statRhoDisplay");
  const elInfer = $("#statInferCore");
  if (!elMean || !elKinds || !elRho || !elInfer) return;

  if (!hasCompletedInference) {
    elMean.textContent = "—";
    elKinds.textContent = "—";
    elRho.textContent = "—";
    elInfer.textContent = "—";
    return;
  }

  const list = dets || [];
  if (list.length) {
    let s = 0;
    for (const d of list) s += Number(d.confidence) || 0;
    elMean.textContent = `${((s / list.length) * 100).toFixed(1)}%`;
  } else {
    elMean.textContent = "—";
  }

  const byClass = summary?.by_class && typeof summary.by_class === "object" ? summary.by_class : null;
  if (byClass && Object.keys(byClass).length) {
    elKinds.textContent = `${Object.keys(byClass).length} 种`;
  } else if (list.length) {
    const names = new Set(list.map((d) => String(d.class_name ?? "").trim()).filter(Boolean));
    elKinds.textContent = `${names.size} 种`;
  } else {
    elKinds.textContent = "0 种";
  }

  const rho = Number(lastMmPerPixel);
  elRho.textContent = Number.isFinite(rho) ? `${rho.toFixed(3)} mm/px` : "—";

  const t = lastTimingSnapshot;
  const inf = t && t.inference_ms != null ? Number(t.inference_ms) : NaN;
  elInfer.textContent = Number.isFinite(inf) ? `${Math.round(inf * 100) / 100} ms` : "—";
}

function syncConfFromRange() {
  const r = $("#confRange");
  const h = $("#conf");
  const out = $("#confReadout");
  if (!r || !h) return;
  const v = Number(r.value) / 100;
  h.value = String(v);
  if (out) out.textContent = v.toFixed(2);
}

function syncIouFromRange() {
  const r = $("#iouRange");
  const h = $("#iou");
  const out = $("#iouReadout");
  if (!r || !h) return;
  const v = Number(r.value) / 100;
  h.value = String(v);
  if (out) out.textContent = v.toFixed(2);
}

async function predictImage(file) {
  const fd = new FormData();
  fd.append("file", file);
  syncConfFromRange();
  syncIouFromRange();
  fd.append("conf", $("#conf").value || "0.25");
  fd.append("iou", $("#iou").value || "0.7");
  fd.append("mm_per_pixel", $("#mmPx").value || "0.5");
  setStatus("推理中…", "warn");
  const r = await fetch("/api/predict_image", { method: "POST", body: fd });
  if (!r.ok) {
    let t = await r.text();
    try {
      const j = JSON.parse(t);
      if (j.detail != null) {
        t = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      }
    } catch {
      /* 非 JSON 时沿用原文 */
    }
    setStatus("推理失败", "err");
    alert(t.slice(0, 800));
    return;
  }
  const j = await r.json();
  showDualResult(j.image_seg_png_base64, j.image_boxes_png_base64);
  lastDetections = j.detections || [];
  lastSummary = j.summary;
  lastMmPerPixel = j.mm_per_pixel;
  lastTimingSnapshot = j.timing && typeof j.timing === "object" ? { ...j.timing } : null;
  hasCompletedInference = true;
  renderTable(lastDetections);
  updateStats(lastSummary, lastDetections);
  setTimingBar("推理耗时（图像）：" + formatInferenceTiming(j.timing));
  setStatus("完成");
}

async function predictVideo(file) {
  const fd = new FormData();
  fd.append("file", file);
  syncConfFromRange();
  syncIouFromRange();
  fd.append("conf", $("#conf").value || "0.25");
  fd.append("iou", $("#iou").value || "0.7");
  fd.append("mm_per_pixel", $("#mmPx").value || "0.5");
  fd.append("max_frames", $("#maxFrames").value || "24");
  setStatus("视频采样推理中…", "warn");
  const r = await fetch("/api/predict_video", { method: "POST", body: fd });
  if (!r.ok) {
    let t = await r.text();
    try {
      const j = JSON.parse(t);
      if (j.detail != null) {
        t = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      }
    } catch {
      /* 非 JSON */
    }
    setStatus("视频处理失败", "err");
    alert(t.slice(0, 800));
    return;
  }
  const j = await r.json();
  const frames = j.frames || [];
  const thumbs = $("#thumbs");
  thumbs.innerHTML = "";
  if (!frames.length) {
    setStatus("未读到有效帧", "warn");
    return;
  }
  hasCompletedInference = true;
  frames.forEach((fr, idx) => {
    const img = document.createElement("img");
    img.src = `data:image/png;base64,${fr.image_seg_png_base64 || ""}`;
    const tline = formatInferenceTiming(fr.timing);
    img.title = `帧 ${fr.frame_index}（缩略图为分割图）` + (tline !== "—" ? ` · ${tline}` : "");
    img.className = idx === 0 ? "active" : "";
    img.onclick = () => {
      thumbs.querySelectorAll("img").forEach((x) => x.classList.remove("active"));
      img.classList.add("active");
      showDualResult(fr.image_seg_png_base64, fr.image_boxes_png_base64);
      lastDetections = fr.detections || [];
      lastSummary = fr.summary;
      lastTimingSnapshot = fr.timing && typeof fr.timing === "object" ? { ...fr.timing } : null;
      renderTable(lastDetections);
      updateStats(fr.summary, lastDetections);
      setTimingBar("当前帧耗时：" + formatInferenceTiming(fr.timing));
    };
    thumbs.appendChild(img);
  });
  const first = frames[0];
  showDualResult(first.image_seg_png_base64, first.image_boxes_png_base64);
  lastDetections = first.detections || [];
  lastSummary = first.summary;
  lastMmPerPixel = j.mm_per_pixel;
  lastTimingSnapshot = first.timing && typeof first.timing === "object" ? { ...first.timing } : null;
  renderTable(lastDetections);
  updateStats(first.summary, lastDetections);
  const vbc = j.video_summary?.by_class;
  if (vbc && Object.keys(vbc).length) {
    renderClassDistribution(vbc, $("#donutText"));
  } else {
    const dn = $("#donutText");
    dn.className = "donut-placeholder";
    dn.innerHTML = "";
    dn.textContent = frames.length ? "本视频采样帧未检出" : "等待推理";
  }
  const ta = j.timing || {};
  let aggLine = "";
  if (ta.mean_inference_ms != null) {
    aggLine = `平均核心推理 ${ta.mean_inference_ms} ms / 帧`;
    if (ta.sum_inference_ms != null) aggLine += ` · 累计核心 ${ta.sum_inference_ms} ms`;
  }
  if (ta.mean_server_ms != null) {
    aggLine += (aggLine ? " · " : "") + `平均服务端 ${ta.mean_server_ms} ms / 帧`;
  }
  const firstT = first.timing ? " · 首帧 " + formatInferenceTiming(first.timing) : "";
  setTimingBar(
    aggLine
      ? `视频 ${frames.length} 帧：` + aggLine + firstT + "（点击缩略图切换单帧耗时）"
      : "视频推理：暂无有效帧耗时统计"
  );
  setStatus(`已处理 ${frames.length} 个采样帧（缩略图切换可查看各帧）`);
}

function wireUpload(inputId, handler) {
  $(inputId).addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) handler(f);
    e.target.value = "";
  });
}

function wireNav() {
  document.querySelectorAll(".sidebar .nav-item[data-nav]").forEach((item) => {
    item.addEventListener("click", () => {
      document.querySelectorAll(".sidebar .nav-item[data-nav]").forEach((x) => x.classList.remove("active"));
      item.classList.add("active");
      const id = item.getAttribute("data-nav-section");
      if (!id) return;
      const target = document.getElementById(`section-${id}`);
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      document.querySelectorAll(".nav-scroll-target.section-flash").forEach((x) => x.classList.remove("section-flash"));
      target.classList.add("section-flash");
      window.setTimeout(() => target.classList.remove("section-flash"), 1200);
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  wireUpload("#fileImage", predictImage);
  wireUpload("#fileVideo", predictVideo);
  $("#btnPickImg").onclick = () => $("#fileImage").click();
  $("#btnPickVid").onclick = () => $("#fileVideo").click();
  $("#confRange")?.addEventListener("input", syncConfFromRange);
  $("#iouRange")?.addEventListener("input", syncIouFromRange);
  syncConfFromRange();
  syncIouFromRange();
  wireNav();
  wireBriefingToggle();
  wireChatComposer();
  wireActionChips();
  renderHealthGauge("—");
  updateBriefing(null, []);
  updateDerivedMetricsPanel(null, []);
  renderRiskPanel([]);
});
