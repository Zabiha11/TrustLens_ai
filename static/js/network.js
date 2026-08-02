/**
 * TrustLens AI — Reviewer Network Analysis Script
 */

document.addEventListener("DOMContentLoaded", () => {
  const reloadBtn = document.getElementById("reloadGraphBtn");
  if (reloadBtn) {
    reloadBtn.addEventListener("click", loadNetworkData);
  }

  loadNetworkData();
});

async function loadNetworkData() {
  const loader = document.getElementById("graphLoader");
  const iframe = document.getElementById("pyvisIframe");
  if (loader) loader.style.display = "flex";
  if (iframe) iframe.style.display = "none";

  try {
    const res = await fetch("/api/network/data", {
      cache: "no-store",
      headers: { "Cache-Control": "no-store" },
    });
    if (!res.ok) throw new Error("Failed to fetch network graph data");
    const data = await res.json();

    document.getElementById("netUsers").textContent = data.total_users || 0;
    document.getElementById("netProducts").textContent = data.total_products || 0;
    document.getElementById("netEdges").textContent = data.total_edges || 0;
    document.getElementById("netCommunities").textContent = data.community_count || 0;

    if (iframe && data.graph_html) {
      iframe.srcdoc = data.graph_html;
      iframe.onload = () => {
        if (loader) loader.style.display = "none";
        iframe.style.display = "block";
      };
    } else if (loader) {
      loader.innerHTML = `<div class="text-muted text-center p-4"><i class="bi bi-database-x fs-1 mb-2 d-block opacity-50"></i><span class="fw-semibold">No network data yet. Upload CSV with user_id & product_id columns.</span></div>`;
    }

    renderClusterTable(data.clusters || []);
  } catch (err) {
    console.error("Error loading network graph:", err);
    if (loader) {
      loader.innerHTML = `<span class="text-danger"><i class="bi bi-exclamation-octagon me-2"></i>Failed to render network graph: ${escapeHtml(err.message)}</span>`;
    }
  }
}

function escapeHtml(str) {
  if (str == null) return "";
  const div = document.createElement("div");
  div.textContent = String(str);
  return div.innerHTML;
}

function renderClusterTable(clusters) {
  const tbody = document.getElementById("clustersTableBody");
  if (!tbody) return;

  if (clusters.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-muted text-center py-4">No coordinated reviewer clusters detected.</td></tr>';
    return;
  }

  tbody.innerHTML = clusters
    .map((c) => {
      const isSuspicious = c.suspicion_score >= 60;
      const statusBadge = isSuspicious
        ? '<span class="badge badge-fake"><i class="bi bi-exclamation-triangle me-1"></i>Suspicious</span>'
        : '<span class="badge badge-genuine"><i class="bi bi-shield-check me-1"></i>Normal</span>';

      return `
      <tr>
        <td class="fw-bold font-monospace text-cyan">${c.cluster_id}</td>
        <td><span class="badge bg-dark border">${c.user_count} Reviewers</span></td>
        <td>${c.product_count} Products</td>
        <td>${c.avg_rating} ★</td>
        <td>
          <div class="d-flex align-items-center gap-2">
            <span class="fw-bold ${isSuspicious ? 'text-danger' : 'text-success'}">${c.suspicion_score}%</span>
            <div class="confidence-bar flex-grow-1" style="width: 80px;">
              <div class="confidence-fill ${isSuspicious ? 'bg-danger' : 'bg-success'}" style="width: ${c.suspicion_score}%"></div>
            </div>
          </div>
        </td>
        <td>${statusBadge}</td>
      </tr>
    `;
    })
    .join("");
}
