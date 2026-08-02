/**
 * TrustLens AI — Temporal Fraud Detection Script
 */

let temporalData = null;
let currentGranularity = "hourly";

document.addEventListener("DOMContentLoaded", () => {
  const btnHourly = document.getElementById("btnGranularityHourly");
  const btnDaily = document.getElementById("btnGranularityDaily");
  const btnWeekly = document.getElementById("btnGranularityWeekly");

  if (btnHourly) btnHourly.addEventListener("click", () => setGranularity("hourly", btnHourly));
  if (btnDaily) btnDaily.addEventListener("click", () => setGranularity("daily", btnDaily));
  if (btnWeekly) btnWeekly.addEventListener("click", () => setGranularity("weekly", btnWeekly));

  loadTemporalData();
});

async function loadTemporalData() {
  try {
    const res = await fetch("/api/temporal/data", {
      cache: "no-store",
      headers: { "Cache-Control": "no-store" },
    });
    if (!res.ok) throw new Error("Failed to fetch temporal data");
    temporalData = await res.json();

    document.getElementById("totalBurstsCount").textContent = temporalData.total_bursts_found || 0;

    renderPlotlyTimeline();
    renderBurstsTable(temporalData.detected_bursts || []);
  } catch (err) {
    console.error("Error loading temporal fraud data:", err);
    const container = document.getElementById("timelinePlotly");
    if (container) {
      container.innerHTML = `<div class="d-flex align-items-center justify-content-center h-100 w-100 p-5 text-muted" style="min-height:240px;"><div class="text-center"><i class="bi bi-exclamation-triangle fs-1 mb-2 d-block opacity-50"></i><span class="fw-semibold">Temporal data unavailable.</span></div></div>`;
    }
  }
}

function setGranularity(type, activeBtn) {
  currentGranularity = type;
  document.querySelectorAll(".btn-group button").forEach((b) => b.classList.remove("active"));
  if (activeBtn) activeBtn.classList.add("active");
  renderPlotlyTimeline();
}

function renderPlotlyTimeline() {
  const container = document.getElementById("timelinePlotly");
  if (!container) return;

  if (!temporalData) {
    container.innerHTML = noDataMarkup("Temporal data loading…");
    return;
  }

  let trendMap = {};
  if (currentGranularity === "hourly") trendMap = temporalData.hourly_trends || {};
  else if (currentGranularity === "daily") trendMap = temporalData.daily_trends || {};
  else trendMap = temporalData.weekly_trends || {};

  const xVals = Object.keys(trendMap);
  const yVals = Object.values(trendMap);
  const total = yVals.reduce((s, n) => s + (Number(n) || 0), 0);

  if (!xVals.length || total === 0) {
    container.innerHTML = noDataMarkup("No temporal activity data available yet. Upload CSV with timestamps (review_time/timestamp).");
    return;
  }

  container.innerHTML = "";

  const trace = {
    x: xVals,
    y: yVals,
    type: "scatter",
    mode: "lines+markers",
    line: { color: "#8f6b3d", width: 3, shape: "spline" },
    marker: { color: "#bfae8b", size: 8, line: { color: "#8f6b3d", width: 1 } },
    name: "Review Volume",
  };

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#4b5563", family: "Inter, sans-serif", size: 12 },
    margin: { l: 52, r: 20, t: 24, b: 56 },
    xaxis: {
      gridcolor: "rgba(75, 85, 99, 0.12)",
      title: { text: currentGranularity.toUpperCase() + " TIMELINE", font: { size: 11, color: "#4b5563" } },
      tickfont: { color: "#4b5563" },
      linecolor: "rgba(75, 85, 99, 0.2)",
    },
    yaxis: {
      gridcolor: "rgba(75, 85, 99, 0.12)",
      title: { text: "REVIEW COUNT", font: { size: 11, color: "#4b5563" } },
      tickfont: { color: "#4b5563" },
      linecolor: "rgba(75, 85, 99, 0.2)",
      zeroline: false,
    },
  };

  Plotly.newPlot("timelinePlotly", [trace], layout, { responsive: true, displayModeBar: false });
}

function renderBurstsTable(bursts) {
  const tbody = document.getElementById("burstsTableBody");
  if (!tbody) return;

  if (!bursts || bursts.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="text-muted text-center py-4">No suspicious temporal bursts detected.</td></tr>';
    return;
  }

  tbody.innerHTML = bursts
    .map(
      (b) => `
    <tr>
      <td class="fw-bold font-monospace text-warning">${escapeHtml(b.burst_id)}</td>
      <td>${escapeHtml(b.timestamp)}</td>
      <td><span class="badge bg-danger fs-6">${b.review_count} Reviews</span></td>
      <td>${b.user_count} Users</td>
      <td>${b.product_count} Products</td>
      <td>${Number(b.avg_rating).toFixed(1)} ★</td>
      <td>
        <div class="d-flex align-items-center gap-2">
          <span class="fw-bold text-danger">${Number(b.anomaly_score).toFixed(1)}%</span>
          <div class="confidence-bar flex-grow-1" style="width: 70px;">
            <div class="confidence-fill bg-danger" style="width: ${Math.min(100, Math.max(0, Number(b.anomaly_score) || 0))}%"></div>
          </div>
        </div>
      </td>
      <td><span class="badge bg-danger"><i class="bi bi-exclamation-triangle me-1"></i>BURST FLAGGED</span></td>
    </tr>
  `
    )
    .join("");
}

function noDataMarkup(msg) {
  return `<div class="d-flex align-items-center justify-content-center h-100 w-100 p-5 text-muted" style="min-height:240px;"><div class="text-center"><i class="bi bi-database-x fs-1 mb-2 d-block opacity-50"></i><span class="fw-semibold">${escapeHtml(msg)}</span></div></div>`;
}

function escapeHtml(str) {
  if (str == null) return "";
  const div = document.createElement("div");
  div.textContent = String(str);
  return div.innerHTML;
}
