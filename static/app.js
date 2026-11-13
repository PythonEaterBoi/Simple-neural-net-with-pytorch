/* OmniTrain frontend: dynamic forms from /api/modalities, job polling, prompting. */
const $ = (id) => document.getElementById(id);
let REGISTRY = {};
let curModality = "text", curTask = "generation";
let activeJob = null, pollTimer = null, logCursor = 0;
let checkpoints = [];

/* ---------------- tabs ---------------- */
document.querySelectorAll(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    document.querySelectorAll(".panel").forEach((p) => (p.hidden = true));
    $("tab-" + b.dataset.tab).hidden = false;
    if (b.dataset.tab === "prompt") loadCheckpoints();
    if (b.dataset.tab === "models") renderModels();
  })
);

/* ---------------- health ---------------- */
async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    const el = $("health");
    el.textContent = h.torch ? `torch ${h.torch} · ${h.cuda ? "cuda ✅" : "cpu"}` : "no torch ⚠️";
    el.classList.toggle("ok", !!h.torch);
  } catch {
    $("health").textContent = "offline ⚠️";
  }
}

/* ---------------- registry / train form ---------------- */
async function loadRegistry() {
  const r = await fetch("/api/modalities");
  REGISTRY = await r.json();
  curModality = Object.keys(REGISTRY)[0];
  renderModalities();
  renderTasks();
}

function renderModalities() {
  const box = $("modality-cards");
  box.innerHTML = "";
  for (const [key, m] of Object.entries(REGISTRY)) {
    const d = document.createElement("div");
    d.className = "mcard" + (key === curModality ? " active" : "");
    d.textContent = `${m.icon} ${m.label}`;
    d.title = m.description;
    d.onclick = () => {
      curModality = key;
      curTask = Object.keys(m.tasks)[0];
      renderModalities();
      renderTasks();
    };
    box.appendChild(d);
  }
}

function renderTasks() {
  const tasks = REGISTRY[curModality].tasks;
  if (!tasks[curTask]) curTask = Object.keys(tasks)[0];
  const sel = $("task-select");
  sel.innerHTML = "";
  for (const [key, t] of Object.entries(tasks)) {
    const o = document.createElement("option");
    o.value = key;
    o.textContent = t.label;
    sel.appendChild(o);
  }
  sel.value = curTask;
  sel.onchange = () => {
    curTask = sel.value;
    renderTasks();
  };
  const t = tasks[curTask];
  $("task-desc").textContent = t.description;
  $("data-help").textContent = "📁 " + t.data_help;
  $("train-files").accept = t.accept || "";
  renderHyperparams(t.hyperparams || []);
}

function renderHyperparams(params) {
  const box = $("hyperparams");
  box.innerHTML = "";
  for (const h of params) {
    const wrap = document.createElement("div");
    const lab = document.createElement("label");
    lab.textContent = h.label;
    lab.title = h.help || "";
    wrap.appendChild(lab);
    let input;
    if (h.type === "select") {
      input = document.createElement("select");
      for (const o of h.options) {
        const op = document.createElement("option");
        op.value = o;
        op.textContent = o;
        input.appendChild(op);
      }
      input.value = String(h.default);
    } else if (h.type === "checkbox") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = !!h.default;
      input.style.width = "20px";
    } else {
      input = document.createElement("input");
      input.type = h.type === "int" || h.type === "float" ? "number" : "text";
      if (h.type === "float") input.step = h.step || 0.0001;
      if (h.min !== undefined) input.min = h.min;
      if (h.max !== undefined) input.max = h.max;
      input.value = h.default;
    }
    input.dataset.name = h.name;
    input.dataset.type = h.type;
    wrap.appendChild(input);
    box.appendChild(wrap);
  }
}

function readHyperparams() {
  const cfg = {};
  document.querySelectorAll("#hyperparams [data-name]").forEach((el) => {
    const t = el.dataset.type;
    let v;
    if (t === "checkbox") v = el.checked;
    else if (t === "int") v = parseInt(el.value, 10);
    else if (t === "float") v = parseFloat(el.value);
    else if (!isNaN(Number(el.value)) && el.tagName === "SELECT" && !isNaN(Number(el.value)))
      v = Number(el.value);
    else v = el.value;
    cfg[el.dataset.name] = v;
  });
  return cfg;
}

$("train-files").addEventListener("change", (e) => {
  const files = [...e.target.files];
  $("file-list").textContent = files.length
    ? files.map((f) => `${f.name} (${(f.size / 1024).toFixed(1)} KB)`).join(", ")
    : "";
});

$("start-train").addEventListener("click", async () => {
  const msg = $("train-msg");
  msg.className = "msg";
  msg.textContent = "";
  const files = $("train-files").files;
  if (!files.length) {
    msg.className = "msg err";
    msg.textContent = "Please choose training data files first.";
    return;
  }
  const fd = new FormData();
  fd.append("modality", curModality);
  fd.append("task", curTask);
  fd.append("config", JSON.stringify(readHyperparams()));
  for (const f of files) fd.append("files", f);
  $("start-train").disabled = true;
  try {
    const r = await fetch("/api/train", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const { job_id } = await r.json();
    msg.className = "msg ok";
    msg.textContent = `Training started (job ${job_id}).`;
    $("train-files").value = "";
    $("file-list").textContent = "";
    await refreshJobs();
    selectJob(job_id);
  } catch (e) {
    msg.className = "msg err";
    msg.textContent = "Error: " + e.message;
  } finally {
    $("start-train").disabled = false;
  }
});

/* ---------------- jobs ---------------- */
$("refresh-jobs").addEventListener("click", refreshJobs);

async function refreshJobs() {
  const r = await fetch("/api/jobs");
  const list = await r.json();
  const box = $("job-list");
  box.innerHTML = list.length ? "" : '<span class="muted">No jobs yet.</span>';
  for (const j of list) {
    const d = document.createElement("div");
    d.className = "job" + (j.id === activeJob ? " active" : "");
    const pct = j.epochs ? Math.round((100 * (j.last?.epoch || 0)) / j.epochs) : 0;
    d.innerHTML =
      `<div class="jt">${j.title} <span class="badge ${j.status}">${j.status}</span></div>` +
      `<div class="js">#${j.id} · ${j.created_at}` +
      (j.last ? ` · epoch ${j.last.epoch}/${j.epochs} · loss ${j.last.loss.toFixed(4)}` : "") +
      `</div><div class="progress"><div style="width:${pct}%"></div></div>`;
    d.onclick = () => selectJob(j.id);
    box.appendChild(d);
  }
}

function selectJob(id) {
  activeJob = id;
  logCursor = 0;
  $("job-logs").textContent = "";
  $("job-detail-card").hidden = false;
  refreshJobs();
  pollJob();
  clearInterval(pollTimer);
  pollTimer = setInterval(pollJob, 2000);
}

async function pollJob() {
  if (!activeJob) return;
  try {
    const r = await fetch(`/api/jobs/${activeJob}/logs?since=${logCursor}`);
    if (!r.ok) return;
    const d = await r.json();
    logCursor = d.next;
    if (d.logs.length) {
      const pre = $("job-logs");
      pre.textContent += d.logs.join("\n") + "\n";
      pre.scrollTop = pre.scrollHeight;
    }
    drawChart(d.metrics || []);
    const jr = await fetch(`/api/jobs/${activeJob}`);
    if (jr.ok) {
      const j = await jr.json();
      $("job-title").textContent = `${j.title} (#${j.id})`;
      $("job-status").textContent = `status: ${j.status}` +
        (j.checkpoint ? ` · checkpoint: ${j.checkpoint}` : "") +
        (j.error ? ` · ${j.error}` : "");
      if (["done", "error", "cancelled"].includes(j.status)) {
        clearInterval(pollTimer);
        refreshJobs();
      }
    }
  } catch { /* transient */ }
}

$("cancel-job").addEventListener("click", async () => {
  if (!activeJob) return;
  await fetch(`/api/jobs/${activeJob}/cancel`, { method: "POST" });
  pollJob();
});
$("delete-job").addEventListener("click", async () => {
  if (!activeJob) return;
  const r = await fetch(`/api/jobs/${activeJob}`, { method: "DELETE" });
  if (r.ok) {
    activeJob = null;
    clearInterval(pollTimer);
    $("job-detail-card").hidden = true;
    refreshJobs();
  } else {
    alert("Cancel the job before deleting it.");
  }
});

function drawChart(metrics) {
  const c = $("loss-chart");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (!metrics.length) {
    ctx.fillStyle = "#9aa0c3";
    ctx.font = "13px sans-serif";
    ctx.fillText("Loss curve appears here once training starts…", 12, 24);
    return;
  }
  const series = [
    { key: "loss", color: "#7c6cf0" },
    { key: "val_loss", color: "#3ddc97" },
  ].filter((s) => metrics.some((m) => m[s.key] !== undefined));
  const vals = metrics.flatMap((m) => series.map((s) => m[s.key]).filter((v) => v !== undefined));
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-9) hi = lo + 1;
  const pad = 28;
  const X = (i) => pad + (i / Math.max(metrics.length - 1, 1)) * (c.width - pad - 8);
  const Y = (v) => c.height - pad - ((v - lo) / (hi - lo)) * (c.height - pad - 8);
  ctx.strokeStyle = "#2c3260";
  ctx.strokeRect(pad, 4, c.width - pad - 8, c.height - pad - 8);
  ctx.fillStyle = "#9aa0c3";
  ctx.font = "10px sans-serif";
  ctx.fillText(hi.toFixed(3), 2, 14);
  ctx.fillText(lo.toFixed(3), 2, c.height - pad + 3);
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    metrics.forEach((m, i) => {
      if (m[s.key] === undefined) return;
      const x = X(i), y = Y(m[s.key]);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.font = "11px sans-serif";
  series.forEach((s, i) => {
    ctx.fillStyle = s.color;
    ctx.fillText("━ " + s.key, pad + 8 + i * 90, 16);
  });
}

/* ---------------- prompt ---------------- */
$("refresh-ckpts").addEventListener("click", loadCheckpoints);
$("ckpt-select").addEventListener("change", renderPromptInputs);

async function loadCheckpoints() {
  const r = await fetch("/api/checkpoints");
  checkpoints = await r.json();
  const sel = $("ckpt-select");
  const prev = sel.value;
  sel.innerHTML = "";
  if (!checkpoints.length) {
    const o = document.createElement("option");
    o.textContent = "— no trained models yet —";
    sel.appendChild(o);
  }
  for (const c of checkpoints) {
    const o = document.createElement("option");
    o.value = c.name;
    o.textContent = `${c.name} (${c.modality}/${c.task})`;
    sel.appendChild(o);
  }
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  renderPromptInputs();
}

function selectedCheckpoint() {
  return checkpoints.find((c) => c.name === $("ckpt-select").value);
}

function promptKind(ckpt) {
  if (!ckpt || !REGISTRY[ckpt.modality]) return null;
  const task = REGISTRY[ckpt.modality].tasks[ckpt.task];
  return task ? task.prompt : null;
}

function renderPromptInputs() {
  const box = $("prompt-inputs");
  box.innerHTML = "";
  const ckpt = selectedCheckpoint();
  const spec = promptKind(ckpt);
  $("ckpt-info").textContent = ckpt
    ? `${ckpt.modality}/${ckpt.task} · ${ckpt.size_mb} MB` +
      (ckpt.meta?.labels ? ` · classes: ${ckpt.meta.labels.join(", ")}` : "") +
      (ckpt.meta?.problem_type ? ` · ${ckpt.meta.problem_type}` : "")
    : "";
  if (!ckpt || !spec) {
    box.innerHTML = '<p class="muted">Train a model first, then come back to prompt it.</p>';
    return;
  }
  const kind = spec.kind;
  if (kind === "text") {
    box.innerHTML = `<label class="flabel">Prompt</label>
      <textarea id="p-text" placeholder="${spec.placeholder || ""}"></textarea>`;
  } else if (kind === "image" || kind === "video") {
    box.innerHTML = `<label class="flabel">Upload ${kind}</label>
      <input type="file" id="p-file" accept="${kind}/*,.zip">`;
    const prev = document.createElement("div");
    box.appendChild(prev);
    box.querySelector("#p-file").addEventListener("change", (e) => {
      const f = e.target.files[0];
      if (f && f.type.startsWith("image/")) {
        prev.innerHTML = `<img class="preview" src="${URL.createObjectURL(f)}">`;
      } else prev.innerHTML = f ? `<div class="muted small">${f.name}</div>` : "";
    });
  } else if (kind === "class") {
    const labels = ckpt.meta?.labels || [];
    box.innerHTML = `<label class="flabel">Class to generate</label><select id="p-text"></select>`;
    const sel = box.querySelector("#p-text");
    labels.forEach((l) => {
      const o = document.createElement("option");
      o.value = l; o.textContent = l;
      sel.appendChild(o);
    });
  } else if (kind === "form") {
    const feats = ckpt.meta?.feature_names || [];
    box.innerHTML = `<label class="flabel">Feature values</label>`;
    const grid = document.createElement("div");
    grid.className = "hyper";
    feats.forEach((f) => {
      const w = document.createElement("div");
      w.innerHTML = `<label>${f}</label><input type="number" step="any" data-feat="${f}">`;
      grid.appendChild(w);
    });
    box.appendChild(grid);
  }
  for (const o of spec.options || []) {
    const w = document.createElement("div");
    w.innerHTML = `<label class="flabel">${o.label}</label>`;
    const inp = document.createElement("input");
    inp.type = "number";
    inp.value = o.default;
    if (o.min !== undefined) inp.min = o.min;
    if (o.max !== undefined) inp.max = o.max;
    if (o.type === "float") inp.step = 0.05;
    inp.dataset.opt = o.name;
    inp.dataset.type = o.type;
    w.appendChild(inp);
    box.appendChild(w);
  }
}

function readPromptOptions() {
  const o = {};
  document.querySelectorAll("#prompt-inputs [data-opt]").forEach((el) => {
    o[el.dataset.opt] = el.dataset.type === "float" ? parseFloat(el.value) : parseInt(el.value, 10);
  });
  return o;
}

$("run-prompt").addEventListener("click", async () => {
  const msg = $("prompt-msg");
  msg.className = "msg";
  msg.textContent = "";
  const ckpt = selectedCheckpoint();
  if (!ckpt) {
    msg.className = "msg err";
    msg.textContent = "No checkpoint selected.";
    return;
  }
  const spec = promptKind(ckpt);
  const fd = new FormData();
  fd.append("checkpoint", ckpt.name);
  fd.append("options", JSON.stringify(readPromptOptions()));
  if (spec.kind === "text" || spec.kind === "class") {
    fd.append("prompt_text", $("p-text").value);
  } else if (spec.kind === "image" || spec.kind === "video") {
    const f = $("p-file").files[0];
    if (!f) {
      msg.className = "msg err";
      msg.textContent = `Please upload a ${spec.kind} file.`;
      return;
    }
    fd.append("file", f);
  } else if (spec.kind === "form") {
    const vals = {};
    document.querySelectorAll("#prompt-inputs [data-feat]").forEach((el) => {
      vals[el.dataset.feat] = el.value;
    });
    fd.append("values", JSON.stringify(vals));
  }
  $("run-prompt").disabled = true;
  $("prompt-result").innerHTML = '<span class="muted">Running…</span>';
  try {
    const r = await fetch("/api/prompt", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    renderResult(j.result);
  } catch (e) {
    msg.className = "msg err";
    msg.textContent = "Error: " + e.message;
    $("prompt-result").innerHTML = '<span class="muted">Failed.</span>';
  } finally {
    $("run-prompt").disabled = false;
  }
});

function renderResult(res) {
  const box = $("prompt-result");
  box.innerHTML = "";
  if (res.type === "text") {
    const d = document.createElement("div");
    d.className = "gen-text";
    d.textContent = res.text;
    box.appendChild(d);
  } else if (res.type === "label") {
    const h = document.createElement("h3");
    h.textContent = `🏷️ ${res.label} (${(res.confidence * 100).toFixed(1)}%)`;
    box.appendChild(h);
    for (const r of res.ranking || []) {
      const d = document.createElement("div");
      d.className = "rankbar";
      d.innerHTML = `${r.label} — ${(r.prob * 100).toFixed(1)}%
        <div class="bar"><div style="width:${r.prob * 100}%"></div></div>`;
      box.appendChild(d);
    }
  } else if (res.type === "number") {
    box.innerHTML = `<h3>🔢 Predicted value: ${res.value}</h3>`;
  } else if (res.type === "images") {
    const h = document.createElement("h3");
    h.textContent = `🎨 Generated "${res.label}"`;
    box.appendChild(h);
    for (const b64 of res.images) {
      const img = document.createElement("img");
      img.className = "gen-img";
      img.src = "data:image/png;base64," + b64;
      box.appendChild(img);
    }
  }
}

/* ---------------- models ---------------- */
$("refresh-models").addEventListener("click", renderModels);

async function renderModels() {
  const r = await fetch("/api/checkpoints");
  checkpoints = await r.json();
  const box = $("model-list");
  if (!checkpoints.length) {
    box.innerHTML = '<span class="muted">No checkpoints yet. Train something!</span>';
    return;
  }
  let html = `<table class="models"><tr><th>Checkpoint</th><th>Modality/Task</th>
    <th>Details</th><th>Size</th><th></th></tr>`;
  for (const c of checkpoints) {
    const det = [
      c.meta?.labels ? c.meta.labels.join(", ") : "",
      c.meta?.problem_type || "",
      c.created || "",
    ].filter(Boolean).join(" · ");
    html += `<tr><td>${c.name}</td><td>${c.modality}/${c.task}</td>
      <td class="muted">${det}</td><td>${c.size_mb} MB</td>
      <td><button class="danger" data-del="${c.name}">Delete</button></td></tr>`;
  }
  box.innerHTML = html + "</table>";
  box.querySelectorAll("[data-del]").forEach((b) =>
    b.addEventListener("click", async () => {
      if (!confirm(`Delete ${b.dataset.del}?`)) return;
      await fetch(`/api/checkpoints/${b.dataset.del}`, { method: "DELETE" });
      renderModels();
    })
  );
}

/* ---------------- init ---------------- */
checkHealth();
loadRegistry().then(() => loadCheckpoints());
refreshJobs();
setInterval(checkHealth, 15000);
