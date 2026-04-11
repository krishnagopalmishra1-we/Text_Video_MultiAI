const form = document.getElementById("job-form");
const submitBtn = document.getElementById("submit-btn");
const refreshBtn = document.getElementById("refresh-btn");
const statusPill = document.getElementById("status-pill");
const progressBar = document.getElementById("progress-inner");
const progressText = document.getElementById("progress-text");
const jobIdText = document.getElementById("job-id");
const scenesText = document.getElementById("scenes");
const errorText = document.getElementById("error");
const outputPathText = document.getElementById("output-path");
const downloadWrap = document.getElementById("download-wrap");
const downloadLink = document.getElementById("download-link");
const logEl = document.getElementById("log");

let currentJobId = "";
let pollTimer = null;

function addLog(msg) {
  const t = new Date().toLocaleTimeString();
  logEl.innerText += `[${t}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(status) {
  statusPill.className = "status-pill";
  if (status === "done") {
    statusPill.classList.add("done");
  }
  if (status === "failed") {
    statusPill.classList.add("failed");
  }
  statusPill.textContent = status || "idle";
}

function updateUI(data) {
  setStatus(data.status);
  const pct = Math.max(0, Math.min(100, Number(data.progress || 0)));
  progressBar.style.width = `${pct}%`;
  progressText.textContent = `${pct.toFixed(1)}%`;
  jobIdText.textContent = data.job_id || "-";
  scenesText.textContent = `${data.completed_scenes || 0} / ${data.total_scenes || 0}`;
  errorText.textContent = data.error || "-";
  outputPathText.textContent = data.output_path || "-";

  if (data.status === "done") {
    downloadWrap.style.display = "block";
    downloadLink.href = `/download/${data.job_id}`;
    addLog("Job finished. Download is ready.");
    stopPolling();
    submitBtn.disabled = false;
  } else if (data.status === "failed") {
    addLog(`Job failed: ${data.error || "unknown error"}`);
    stopPolling();
    submitBtn.disabled = false;
  }
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function fetchStatus(jobId, silent = false) {
  try {
    const res = await fetch(`/status/${jobId}`);
    if (!res.ok) {
      throw new Error(`status HTTP ${res.status}`);
    }
    const data = await res.json();
    updateUI(data);
    if (!silent) {
      addLog(`Status: ${data.status}, progress ${Number(data.progress || 0).toFixed(1)}%`);
    }
  } catch (err) {
    addLog(`Status check failed: ${err.message}`);
  }
}

function startPolling(jobId) {
  stopPolling();
  fetchStatus(jobId, true);
  pollTimer = setInterval(() => fetchStatus(jobId, true), 3000);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  downloadWrap.style.display = "none";

  const script = document.getElementById("script").value.trim();
  if (script.length < 10) {
    addLog("Script must be at least 10 characters.");
    return;
  }

  const payload = {
    script,
    strategy: document.getElementById("strategy").value,
    style: document.getElementById("style").value,
    quality: document.getElementById("quality").value,
    preferred_model: document.getElementById("preferred_model").value,
    api_fallback: document.getElementById("api_fallback").checked,
    pacing: document.getElementById("pacing").value,
    transition: document.getElementById("transition").value,
    min_clip: Number(document.getElementById("min_clip").value),
    max_clip: Number(document.getElementById("max_clip").value),
    upscale_4k: document.getElementById("upscale_4k").checked,
    resume: document.getElementById("resume").checked
  };

  submitBtn.disabled = true;
  setStatus("submitting");
  addLog("Submitting job...");

  try {
    const res = await fetch("/generate_video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`submit HTTP ${res.status}: ${txt}`);
    }

    const data = await res.json();
    currentJobId = data.job_id;
    addLog(`Job submitted: ${currentJobId}`);
    updateUI(data);
    startPolling(currentJobId);
  } catch (err) {
    setStatus("failed");
    addLog(`Submit failed: ${err.message}`);
    submitBtn.disabled = false;
  }
});

refreshBtn.addEventListener("click", async () => {
  const inputId = document.getElementById("job_id_input").value.trim();
  const id = inputId || currentJobId;
  if (!id) {
    addLog("No job id available. Submit a job or enter one manually.");
    return;
  }
  currentJobId = id;
  addLog(`Refreshing status for ${id}`);
  await fetchStatus(id);
  startPolling(id);
});
