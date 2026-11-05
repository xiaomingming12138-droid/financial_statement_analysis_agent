"use strict";

const $ = (id) => document.getElementById(id);
let documents = [];
let selected = new Set();
let lastResult = null;
let lastQuestion = "";
let busy = false;
let toastTimer;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4500);
}

async function api(url, options = {}) {
  let response;
  try { response = await fetch(url, options); }
  catch { throw new Error("无法连接本地服务，请确认服务仍在运行。"); }
  const data = await response.json();
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "请求参数不符合要求，请检查问题和所选文档。";
    throw new Error(detail);
  }
  return data;
}

function renderDocuments() {
  $("documents").replaceChildren();
  $("document-count").textContent = String(documents.length).padStart(2, "0");
  for (const doc of documents) {
    const label = element("label", "document-item");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selected.has(doc.id);
    checkbox.disabled = busy;
    checkbox.addEventListener("change", () => checkbox.checked ? selected.add(doc.id) : selected.delete(doc.id));
    const title = element("span", "doc-title", doc.title);
    title.append(element("small", "doc-details", `${doc.pages} ${/\.txt$|\.md$/i.test(doc.title) ? "个文本段" : "页"} · ${doc.is_demo ? "虚构示例" : "本次上传"}`));
    label.append(checkbox, title);
    $("documents").append(label);
  }
}

function setBusy(value) {
  busy = value;
  $("submit").disabled = value;
  $("submit").firstElementChild.textContent = value ? "研究中…" : "开始研究";
  $("question").disabled = value;
  $("mode").disabled = value;
  $("upload").disabled = value;
  document.querySelectorAll("[data-question]").forEach(button => { button.disabled = value; });
  renderDocuments();
}

function appendCitedText(parent, text, sources) {
  const expression = /\[([\w:-]+)\]/g;
  let last = 0;
  for (const match of text.matchAll(expression)) {
    parent.append(document.createTextNode(text.slice(last, match.index)));
    const index = sources.findIndex(source => source.id === match[1]);
    if (index >= 0) {
      const button = element("button", "citation", `[${index + 1}]`);
      button.type = "button";
      button.title = `${sources[index].title} · 第 ${sources[index].page} 页`;
      button.setAttribute("aria-label", `查看引用 ${index + 1} 的原文`);
      button.addEventListener("click", () => {
        const card = $(sources[index].id);
        card.scrollIntoView({ behavior: "smooth", block: "center" });
        card.classList.add("highlight");
        card.focus({ preventScroll: true });
        setTimeout(() => card.classList.remove("highlight"), 1800);
      });
      parent.append(button);
    } else {
      parent.append(document.createTextNode(match[0]));
    }
    last = match.index + match[0].length;
  }
  parent.append(document.createTextNode(text.slice(last)));
}

function renderResult(result, elapsed) {
  $("results").hidden = false;
  $("empty-state").hidden = true;
  $("answer-mode").textContent = result.mode === "live" ? "真实模型" : "离线演示 / 非模型生成";
  $("result-meta").textContent = `${result.mode === "live" ? "模型工具流程" : "离线流程"} · ${result.trace.length} 次工具调用 · ${result.sources.length} 段证据 · ${(elapsed / 1000).toFixed(1)} 秒`;
  $("warnings").replaceChildren(...result.warnings.map(warning => element("p", "warning-banner", warning)));
  $("calculations").replaceChildren();
  for (const calculation of result.calculations) {
    const card = element("article", "calculation");
    const value = element("p", "calculation-value", calculation.value);
    value.append(element("small", "", calculation.unit));
    const details = document.createElement("details");
    details.append(element("summary", "", "查看计算依据"), element("p", "formula", calculation.formula));
    for (const [label, operand] of Object.entries(calculation.inputs)) {
      details.append(element("p", "input-quote", `${label === "numerator" ? "输入 A" : "输入 B"}：${operand.value} ${operand.unit}\n原文：${operand.quote}`));
    }
    card.append(element("p", "calculation-label", calculation.label), value, details);
    $("calculations").append(card);
  }
  $("answer").replaceChildren();
  for (const line of result.answer.split("\n")) {
    if (!line.trim()) continue;
    const paragraph = document.createElement("p");
    appendCitedText(paragraph, line, result.sources);
    $("answer").append(paragraph);
  }
  $("sources").replaceChildren();
  $("source-count").textContent = `${result.sources.length} 段原文`;
  result.sources.forEach((source, index) => {
    const card = element("article", "source-card");
    card.id = source.id;
    card.tabIndex = -1;
    const heading = element("div", "source-heading");
    heading.append(element("strong", "", `[${index + 1}] ${source.title}`), element("span", "", /\.txt$|\.md$/i.test(source.title) ? "文本段" : `第 ${source.page} 页`));
    card.append(heading, element("p", "source-text", source.text), element("code", "source-id", source.id));
    $("sources").append(card);
  });
  $("trace-count").textContent = `${result.trace.length} STEPS`;
  $("trace").replaceChildren();
  for (const event of result.trace) {
    const step = element("div", `trace-step${event.status === "error" ? " failed" : ""}`);
    const name = element("div", "tool-name", event.tool === "search_documents" ? "原文检索" : event.tool === "financial_calculator" ? "财务计算" : "未知工具");
    name.append(element("span", "duration", `${event.duration_ms.toFixed(1)} ms`));
    const details = document.createElement("details");
    details.append(element("summary", "", "查看调用参数"), element("pre", "", JSON.stringify(event.arguments, null, 2)));
    step.append(name, element("p", "", `${event.status === "error" ? "失败 · " : ""}${event.summary}`), details);
    $("trace").append(step);
  }
}

$("research-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (busy) return;
  const question = $("question").value.trim();
  $("error").hidden = true;
  if (question.length < 2) { toast("请先输入研究问题。"); $("question").focus(); return; }
  if (!selected.size) { toast("请至少选择一份研究资料。"); return; }
  const payload = { question, document_ids: [...selected], mode: $("mode").value };
  setBusy(true);
  lastResult = null;
  $("results").hidden = true;
  $("empty-state").hidden = true;
  $("loading").hidden = false;
  $("trace-count").textContent = "0 STEPS";
  const waiting = element("div", "trace-placeholder");
  waiting.append(element("span", "trace-placeholder-dot"), element("p", "", "等待本次研究完成"));
  $("trace").replaceChildren(waiting);
  const started = performance.now();
  try {
    const result = await api("/api/research", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    lastResult = result;
    lastQuestion = question;
    renderResult(result, performance.now() - started);
  } catch (error) {
    $("error").textContent = error.message;
    $("error").hidden = false;
    $("empty-state").hidden = true;
  } finally {
    setBusy(false);
    $("loading").hidden = true;
  }
});

document.querySelectorAll("[data-question]").forEach(button => {
  button.addEventListener("click", () => { $("question").value = button.dataset.question; $("question").focus(); });
});

$("mode").addEventListener("change", () => {
  $("mode-note").textContent = $("mode").value === "live"
    ? "真实模型模式：所选文档的检索片段将发送至已配置的模型服务，由模型自主选择工具。"
    : "离线模式：虚构财报使用固定流程；上传文档仅做检索预览，不调用大模型。";
});

$("upload").addEventListener("change", async () => {
  const file = $("upload").files[0];
  if (!file) return;
  if (file.size > 5 * 1024 * 1024) { toast("文件超过 5 MB，请上传相关章节。"); $("upload").value = ""; return; }
  const form = new FormData();
  form.append("file", file);
  setBusy(true);
  try {
    const result = await api("/api/documents", { method: "POST", body: form });
    documents.push(result.document);
    selected = new Set([result.document.id]);
    toast("已导入并选中该文档。离线模式可预览检索结果。");
  } catch (error) { toast(error.message); }
  finally { $("upload").value = ""; setBusy(false); }
});

$("export").addEventListener("click", () => {
  if (!lastResult) return;
  const lines = ["# 研迹研究简报", "", `研究问题：${lastQuestion}`, `运行模式：${lastResult.mode === "live" ? "真实模型" : "离线演示（非模型生成）"}`, "", ...lastResult.warnings.map(w => `注意：${w}`), "", lastResult.answer, "", "## 原文证据", ""];
  lastResult.sources.forEach(source => { lines.push(`### ${source.title} · 页 ${source.page} · ${source.id}`, "", source.text, ""); });
  lines.push("## 计算记录", "", "```json", JSON.stringify(lastResult.calculations, null, 2), "```", "", "## 校验范围", "", lastResult.checks.note);
  const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" }));
  const link = element("a");
  link.href = url;
  link.download = "研迹-研究简报.md";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

document.querySelectorAll("[data-view]").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-view]").forEach(nav => nav.classList.toggle("active", nav === button));
    const workspace = button.dataset.view === "workspace";
    $("workspace-view").hidden = !workspace;
    $("architecture-view").hidden = workspace;
    $("page-title").textContent = workspace ? "研究工作台" : "项目架构";
  });
});

async function initialize() {
  try {
    const [config, response] = await Promise.all([api("/api/config"), api("/api/documents")]);
    documents = response.documents;
    selected = new Set(documents.length ? [documents[0].id] : []);
    renderDocuments();
    const live = $("mode").querySelector('[value="live"]');
    live.disabled = !config.llm.available;
    live.textContent = config.llm.available ? `真实模型 · ${config.llm.model}` : "真实模型 · 未配置";
  } catch (error) {
    $("error").textContent = error.message;
    $("error").hidden = false;
    $("submit").disabled = true;
  }
}
initialize();
