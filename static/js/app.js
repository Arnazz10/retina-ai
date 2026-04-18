// ── Prediction logic ──
const form = document.getElementById("predict-form");
const fileInput = document.getElementById("file");
const previewImage = document.getElementById("preview-image");
const previewPlaceholder = document.getElementById("preview-placeholder");
const resultTitle = document.getElementById("result-title");
const resultBadge = document.getElementById("result-badge");
const resultSummary = document.getElementById("result-summary");
const resultConfidence = document.getElementById("result-confidence");
const resultRisk = document.getElementById("result-risk");
const resultMode = document.getElementById("result-mode");
const resultNote = document.getElementById("result-note");
const probabilityList = document.getElementById("probability-list");
const config = window.RETINA_CONFIG || {};
const API_BASE = String(config.apiBase || "").replace(/\/$/, "");
const FRONTEND_ONLY = Boolean(config.frontendOnly);

const DR_CLASSES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"];
const DR_COLORS = ["#22c55e", "#84cc16", "#f59e0b", "#f97316", "#ef4444"];

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function backendUnavailableMessage() {
  return API_BASE
    ? "The backend is unavailable right now. Check the deployed API URL and try again."
    : "This Vercel deployment is frontend-only right now. Add your backend URL in retina-config.js to enable live analysis.";
}

function setIdleState() {
  resultTitle.textContent = "Awaiting image";
  resultBadge.textContent = "Idle";
  resultBadge.style.background = "rgba(84, 168, 255, 0.12)";
  resultBadge.style.color = "#1769e7";
  resultSummary.textContent = "Upload a retinal fundus image to generate a grade, confidence score, and follow-up recommendation.";
  resultConfidence.textContent = "--";
  resultRisk.textContent = "--";
  resultMode.textContent = "--";
  resultNote.textContent = "No analysis run yet.";
  probabilityList.innerHTML = "";
}

function setLoadingState() {
  resultTitle.textContent = "Analyzing retinal image";
  resultBadge.textContent = "Running";
  resultBadge.style.background = "rgba(255, 180, 70, 0.18)";
  resultBadge.style.color = "#b86a00";
  resultSummary.textContent = "The upload is being processed. This usually takes a moment.";
  resultConfidence.textContent = "...";
  resultRisk.textContent = "...";
  resultMode.textContent = "Processing";
  resultNote.textContent = "Receiving model output...";
  probabilityList.innerHTML = "";
}

function setErrorState(message) {
  resultTitle.textContent = "Analysis failed";
  resultBadge.textContent = "Error";
  resultBadge.style.background = "rgba(239, 68, 68, 0.12)";
  resultBadge.style.color = "#c62828";
  resultSummary.textContent = message;
  resultConfidence.textContent = "--";
  resultRisk.textContent = "--";
  resultMode.textContent = "--";
  resultNote.textContent = "Fix the issue and try another file.";
  probabilityList.innerHTML = "";
}

function updatePreview(file) {
  if (!file) {
    previewImage.hidden = true;
    previewImage.removeAttribute("src");
    previewPlaceholder.hidden = false;
    previewPlaceholder.textContent = "Image preview will appear here before analysis.";
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  previewImage.hidden = false;
  previewPlaceholder.hidden = true;
  previewImage.onload = () => URL.revokeObjectURL(objectUrl);
}

function renderProbabilities(probabilities) {
  probabilityList.innerHTML = "";

  Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, value]) => {
      const row = document.createElement("div");
      row.className = "probability-row";

      const meta = document.createElement("div");
      meta.className = "probability-meta";

      const name = document.createElement("span");
      name.textContent = label;

      const score = document.createElement("strong");
      score.textContent = `${value.toFixed(2)}%`;

      meta.append(name, score);

      const bar = document.createElement("div");
      bar.className = "probability-bar";

      const fill = document.createElement("div");
      fill.className = "probability-fill";
      fill.style.width = `${Math.max(2, Math.min(100, value))}%`;

      bar.appendChild(fill);
      row.append(meta, bar);
      probabilityList.appendChild(row);
    });
}

fileInput.addEventListener("change", () => {
  updatePreview(fileInput.files[0]);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    setErrorState("Choose an image before running the analysis.");
    return;
  }

  setLoadingState();
  form.classList.add("is-loading");

  try {
    const formData = new FormData();
    formData.append("file", file);

    if (FRONTEND_ONLY && !API_BASE) {
      throw new Error(backendUnavailableMessage());
    }

    const response = await fetch(apiUrl("/api/predict"), {
      method: "POST",
      body: formData
    });
    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "The server could not process this image.");
    }

    resultTitle.textContent = data.predicted_class;
    resultBadge.textContent = data.risk_level.toUpperCase();
    resultBadge.style.background = `${data.color}22`;
    resultBadge.style.color = data.color;
    resultSummary.textContent = data.description;
    resultConfidence.textContent = `${Number(data.confidence).toFixed(2)}%`;
    resultRisk.textContent = data.risk_level;
    resultMode.textContent = data.demo_mode ? "Demo mode" : "Model mode";
    resultNote.textContent = data.recommendation;
    renderProbabilities(data.probabilities || {});
  } catch (error) {
    setErrorState(error.message);
  } finally {
    form.classList.remove("is-loading");
  }
});


// ── Tab Management ──
const navLinks = document.querySelectorAll("#main-nav a");
const tabContents = document.querySelectorAll(".tab-content");

navLinks.forEach(link => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const tabId = link.getAttribute("data-tab");
    
    navLinks.forEach(l => l.classList.remove("active"));
    link.classList.add("active");
    
    tabContents.forEach(content => {
      content.classList.remove("active");
      if (content.id === `${tabId}-tab`) {
        content.classList.add("active");
      }
    });

    if (tabId === "dataset") loadDataset();
    if (tabId === "training") pollTrainingStatus();
  });
});


// ── Dataset Logic ──
let currentSplit = "train";
let currentPage = 1;

const datasetGrid = document.getElementById("dataset-grid");
const prevBtn = document.getElementById("prev-page");
const nextBtn = document.getElementById("next-page");
const pageInfo = document.getElementById("page-info");
const splitToggles = document.querySelectorAll(".split-toggle");

splitToggles.forEach(btn => {
  btn.addEventListener("click", () => {
    splitToggles.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentSplit = btn.getAttribute("data-split");
    currentPage = 1;
    loadDataset();
  });
});

async function loadDataset() {
  if (FRONTEND_ONLY && !API_BASE) {
    datasetGrid.innerHTML = `<div class='error'>${backendUnavailableMessage()}</div>`;
    return;
  }

  datasetGrid.innerHTML = "<div class='loading'>Loading dataset images...</div>";
  try {
    const response = await fetch(apiUrl(`/api/dataset?split=${currentSplit}&page=${currentPage}&limit=20`));
    const data = await response.json();
    
    if (data.success) {
      datasetGrid.innerHTML = "";
      data.data.forEach(item => {
        const card = document.createElement("div");
        card.className = "dataset-item";
        card.innerHTML = `
          <img src="${item.image_url || '/static/img/placeholder.png'}" loading="lazy">
          <div class="dataset-info">
            <strong>ID: ${item.id_code}</strong>
            ${item.diagnosis !== null ? 
              `<span class="label-pill" style="background: ${DR_COLORS[item.diagnosis]}22; color: ${DR_COLORS[item.diagnosis]}">
                Grade ${item.diagnosis}: ${DR_CLASSES[item.diagnosis]}
               </span>` : 
              `<span class="label-pill" style="background: #eee; color: #666">Unlabeled</span>`
            }
          </div>
        `;
        datasetGrid.appendChild(card);
      });
      
      pageInfo.textContent = `Page ${data.page} of ${Math.ceil(data.total / data.limit)}`;
      prevBtn.disabled = data.page <= 1;
      nextBtn.disabled = data.page >= Math.ceil(data.total / data.limit);
    }
  } catch (error) {
    datasetGrid.innerHTML = `<div class='error'>Failed to load data: ${error.message}</div>`;
  }
}

prevBtn.addEventListener("click", () => { if (currentPage > 1) { currentPage--; loadDataset(); } });
nextBtn.addEventListener("click", () => { currentPage++; loadDataset(); });


// ── Training Logic ──
const terminal = document.getElementById("terminal");
const startTrainBtn = document.getElementById("start-train-btn");
const trainStatusText = document.getElementById("train-status-text");
const trainProgressText = document.getElementById("train-progress-text");
const trainProgressFill = document.getElementById("train-progress-fill");

let eventSource = null;

async function startTraining() {
  if (FRONTEND_ONLY && !API_BASE) {
    terminal.textContent = backendUnavailableMessage();
    startTrainBtn.disabled = false;
    return;
  }

  startTrainBtn.disabled = true;
  terminal.textContent = "Connecting to pipeline...";
  
  try {
    const response = await fetch(apiUrl("/api/train"), { method: "POST" });
    const data = await response.json();
    
    if (data.success) {
      connectLogs();
    } else {
      terminal.textContent = `Error: ${data.error}`;
      startTrainBtn.disabled = false;
    }
  } catch (error) {
    terminal.textContent = `Failed to start: ${error.message}`;
    startTrainBtn.disabled = false;
  }
}

function connectLogs() {
  if (eventSource) eventSource.close();
  
  terminal.textContent = "";
  eventSource = new EventSource(apiUrl("/api/train-logs"));
  
  eventSource.onmessage = (event) => {
    const line = event.data;
    const div = document.createElement("div");
    div.textContent = line;
    terminal.appendChild(div);
    terminal.scrollTop = terminal.scrollHeight;
    
    // Simple progress parsing
    if (line.includes("Epoch")) {
       const match = line.match(/Epoch (\d+)\/(\d+)/);
       if (match) {
         const current = parseInt(match[1]);
         const total = parseInt(match[2]);
         const pct = Math.round((current / total) * 100);
         updateTrainProgress(pct, `Training Epoch ${current}/${total}`);
       }
    }
    
    if (line.includes("Training complete")) {
      updateTrainProgress(100, "Completed");
      trainStatusText.textContent = "Finished";
      startTrainBtn.disabled = false;
      eventSource.close();
    }
  };
  
  eventSource.onerror = () => {
    eventSource.close();
    pollTrainingStatus();
  };
}

async function pollTrainingStatus() {
  if (FRONTEND_ONLY && !API_BASE) {
    trainStatusText.textContent = "Backend required";
    startTrainBtn.disabled = true;
    terminal.textContent = backendUnavailableMessage();
    return;
  }

  try {
    const response = await fetch(apiUrl("/api/train-status"));
    const data = await response.json();
    
    trainStatusText.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
    
    if (data.status === "running") {
      startTrainBtn.disabled = true;
      if (!eventSource) connectLogs();
    } else {
      startTrainBtn.disabled = false;
    }
  } catch (error) {}
}

function updateTrainProgress(pct, status) {
  trainProgressText.textContent = `${pct}%`;
  trainProgressFill.style.width = `${pct}%`;
  if (status) trainStatusText.textContent = status;
}

startTrainBtn.addEventListener("click", startTraining);

// Initial state
setIdleState();
if (FRONTEND_ONLY && !API_BASE) {
  resultMode.textContent = "Frontend only";
  resultNote.textContent = backendUnavailableMessage();
  trainStatusText.textContent = "Backend required";
  startTrainBtn.disabled = true;
  terminal.textContent = backendUnavailableMessage();
}
