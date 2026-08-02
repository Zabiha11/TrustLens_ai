/**
 * TrustLens AI — Reviewer Risk Analysis Script
 */

let allReviewers = [];

document.addEventListener("DOMContentLoaded", () => {
  const searchInput = document.getElementById("reviewerSearch");
  const filterSelect = document.getElementById("riskLevelFilter");

  if (searchInput) searchInput.addEventListener("input", filterAndRenderReviewers);
  if (filterSelect) filterSelect.addEventListener("change", filterAndRenderReviewers);

  loadRiskData();
});

async function loadRiskData() {
  try {
    const res = await fetch("/api/risk/data", {
      cache: "no-store",
      headers: { "Cache-Control": "no-store" },
    });
    if (!res.ok) throw new Error("Failed to fetch reviewer risk data");
    const data = await res.json();

    document.getElementById("riskTotalReviewers").textContent = data.total_reviewers || 0;
    document.getElementById("riskHighRiskCount").textContent = data.high_risk_count || 0;
    document.getElementById("riskAvgScore").textContent = (Number(data.avg_risk_score) || 0).toFixed(1) + "%";

    allReviewers = data.reviewers || [];
    filterAndRenderReviewers();
  } catch (err) {
    console.error("Error loading reviewer risk profiles:", err);
    const tbody = document.getElementById("reviewerTableBody");
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="8" class="text-danger text-center py-4">Failed to load reviewer risk data.</td></tr>';
    }
  }
}

function escapeHtml(str) {
  if (str == null) return "";
  const div = document.createElement("div");
  div.textContent = String(str);
  return div.innerHTML;
}

function filterAndRenderReviewers() {
  const tbody = document.getElementById("reviewerTableBody");
  if (!tbody) return;

  const query = (document.getElementById("reviewerSearch")?.value || "").toLowerCase().trim();
  const levelFilter = document.getElementById("riskLevelFilter")?.value || "ALL";

  if (!allReviewers.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="text-muted text-center py-4">No reviewer risk data yet. Upload a CSV with user_id and review_text columns.</td></tr>';
    return;
  }

  const filtered = allReviewers.filter((r) => {
    const matchesQuery = String(r.user_id || "").toLowerCase().includes(query);
    const matchesLevel = levelFilter === "ALL" || r.risk_level === levelFilter;
    return matchesQuery && matchesLevel;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="text-muted text-center py-4">No matching reviewer risk profiles found for this filter.</td></tr>';
    return;
  }

  tbody.innerHTML = filtered
    .map((r) => {
      const badgeClass =
        r.risk_level === "Critical" ? "bg-danger" :
        r.risk_level === "High" ? "bg-warning text-dark" :
        r.risk_level === "Medium" ? "bg-info text-dark" : "bg-success";

      return `
      <tr>
        <td class="fw-bold font-monospace text-cyan">${escapeHtml(r.user_id)}</td>
        <td>
          <div class="d-flex align-items-center gap-2">
            <span class="fw-bold">${Number(r.risk_score).toFixed(1)}%</span>
            <div class="confidence-bar flex-grow-1" style="width: 70px;">
              <div class="confidence-fill ${r.risk_score >= 60 ? 'bg-danger' : 'bg-info'}" style="width: ${Math.min(100, Math.max(0, Number(r.risk_score) || 0))}%"></div>
            </div>
          </div>
        </td>
        <td><span class="badge ${badgeClass}">${escapeHtml(r.risk_level)}</span></td>
        <td>${r.review_count ?? 0}</td>
        <td>${Number(r.avg_rating).toFixed(1)} ★</td>
        <td>${r.unique_products ?? 0}</td>
        <td>${Number(r.breakdown?.duplication_score ?? 0).toFixed(0)} / 25</td>
        <td>${Number(r.breakdown?.velocity_score ?? 0).toFixed(0)} / 15</td>
      </tr>
    `;
    })
    .join("");
}
