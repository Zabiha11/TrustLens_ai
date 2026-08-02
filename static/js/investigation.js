document.addEventListener('DOMContentLoaded', () => {
  let investigationState = null;
  const chartRegistry = {};

  const exportJsonBtn = document.getElementById('exportJsonBtn');
  const exportPdfBtn = document.getElementById('exportPdfBtn');

  if (exportJsonBtn) exportJsonBtn.addEventListener('click', exportJsonReport);
  if (exportPdfBtn) exportPdfBtn.addEventListener('click', exportPdfReport);

  loadInvestigationData();

  async function loadInvestigationData() {
    try {
      const [networkRes, riskRes, temporalRes, dashboardRes] = await Promise.all([
        fetch('/api/network/data', { cache: 'no-store', headers: { 'Cache-Control': 'no-store' } }),
        fetch('/api/risk/data', { cache: 'no-store', headers: { 'Cache-Control': 'no-store' } }),
        fetch('/api/temporal/data', { cache: 'no-store', headers: { 'Cache-Control': 'no-store' } }),
        fetch('/api/dashboard/stats', { cache: 'no-store', headers: { 'Cache-Control': 'no-store' } }),
      ]);

      const networkData = await networkRes.json();
      const riskData = await riskRes.json();
      const temporalData = await temporalRes.json();
      const dashboardData = await dashboardRes.json();

      investigationState = { networkData, riskData, temporalData, dashboardData };

      renderSummary(networkData, riskData, dashboardData);
      renderRiskTable(riskData.reviewers || []);
      renderDistributionChart(dashboardData);
      renderTemporalChart(temporalData);
      renderTrustChart(dashboardData, riskData);
      renderAiChart(dashboardData);
      renderNetworkGraph(networkData.graph_html, networkData.total_edges || 0);
      renderCommunityGraph(networkData);
    } catch (error) {
      console.error(error);
      const tbody = document.getElementById('riskTableBody');
      if (tbody) tbody.innerHTML = '<tr><td colspan="5" class="text-danger text-center py-4">Investigation data unavailable.</td></tr>';
    }
  }

  function noData(containerId, message) {
    const c = document.getElementById(containerId);
    if (!c) return;
    if (chartRegistry[containerId]) {
      try { chartRegistry[containerId].destroy(); } catch (_) {}
      delete chartRegistry[containerId];
    }
    c.innerHTML = `<div class="d-flex align-items-center justify-content-center h-100 w-100 p-5 text-muted" style="min-height:240px;"><div class="text-center"><i class="bi bi-database-x fs-1 mb-2 d-block opacity-50"></i><span class="fw-semibold">${escapeHtml(message || 'No data available')}</span></div></div>`;
  }

  function renderSummary(networkData, riskData, dashboardData) {
    const avgTrust = dashboardData.average_trust_score || 0;
    document.getElementById('summaryTrust').textContent = `${Number(avgTrust).toFixed(1)}%`;
    document.getElementById('summaryHighRisk').textContent = riskData.high_risk_count || 0;
    const aiStats = dashboardData.ai_stats || {};
    document.getElementById('summaryAI').textContent = `${aiStats.highly_likely_ai_count || 0} flagged`;
  }

  function renderRiskTable(reviewers) {
    const tbody = document.getElementById('riskTableBody');
    if (!tbody) return;
    if (!reviewers.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-muted text-center py-4">No reviewer risk data available.</td></tr>';
      return;
    }
    tbody.innerHTML = reviewers.slice(0, 10).map((reviewer) => `
      <tr>
        <td>${escapeHtml(reviewer.user_id)}</td>
        <td><span class="badge bg-danger-subtle text-danger">${Number(reviewer.risk_score).toFixed(1)}</span></td>
        <td><span class="badge bg-${reviewer.level_badge === 'success' ? 'success' : reviewer.level_badge === 'warning' ? 'warning' : reviewer.level_badge === 'info' ? 'info' : 'danger'}">${reviewer.risk_level}</span></td>
        <td>${reviewer.review_count}</td>
        <td>${Number(reviewer.avg_rating).toFixed(1)}</td>
      </tr>
    `).join('');
  }

  function exportJsonReport() {
    if (!investigationState) return;
    const reportPayload = {
      generated_at: new Date().toISOString(),
      summary: {
        average_trust_score: investigationState.dashboardData.average_trust_score || 0,
        high_risk_reviewers: investigationState.riskData.high_risk_count || 0,
        ai_flagged_reviews: investigationState.dashboardData.ai_stats?.highly_likely_ai_count || 0,
      },
      risk_reviewers: (investigationState.riskData.reviewers || []).slice(0, 10),
      dashboard_summary: investigationState.dashboardData,
      temporal_summary: investigationState.temporalData,
      network_summary: {
        total_users: investigationState.networkData.total_users,
        total_products: investigationState.networkData.total_products,
        total_edges: investigationState.networkData.total_edges,
        community_count: investigationState.networkData.community_count,
        clusters: investigationState.networkData.clusters || [],
      },
    };
    const blob = new Blob([JSON.stringify(reportPayload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'investigation-report.json';
    link.click();
    URL.revokeObjectURL(url);
  }

  function canvasToImg(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !canvas.toDataURL) return '';
    try { return `<img src="${canvas.toDataURL('image/png')}" style="max-width:100%; height:auto; display:block;" />`; }
    catch (_) { return ''; }
  }

  function exportPdfReport() {
    if (!investigationState) return;
    const printReport = document.getElementById('printReport');
    if (!printReport) return;

    const topReviewers = (investigationState.riskData.reviewers || []).slice(0, 6);
    const clusterSummary = investigationState.networkData.clusters || [];
    const burstSummary = investigationState.temporalData.detected_bursts || [];
    const aiStats = investigationState.dashboardData.ai_stats || {};

    printReport.innerHTML = `
      <div style="font-family: Inter, sans-serif; color:#111; padding: 10px 20px;">
        <h2 style="margin:0 0 4px 0; color:#8f6b3d;">TrustLens AI — Investigation Report</h2>
        <p style="color:#6b7280; margin:0 0 24px 0;">Generated ${new Date().toLocaleString()}</p>

        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:14px; margin-bottom:26px;">
          <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px;">
            <div style="font-size:13px; color:#6b7280;">Average Trust Score</div>
            <div style="font-size:28px; font-weight:700; color:#22c55e;">${Number(investigationState.dashboardData.average_trust_score || 0).toFixed(1)}%</div>
          </div>
          <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px;">
            <div style="font-size:13px; color:#6b7280;">High-Risk Reviewers</div>
            <div style="font-size:28px; font-weight:700; color:#ef4444;">${investigationState.riskData.high_risk_count || 0}</div>
          </div>
          <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px;">
            <div style="font-size:13px; color:#6b7280;">AI-Generated Flagged</div>
            <div style="font-size:28px; font-weight:700; color:#8f6b3d;">${aiStats.highly_likely_ai_count || 0}</div>
          </div>
        </div>

        <h4 style="margin:0 0 12px 0; color:#8f6b3d;">Prediction Snapshot</h4>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-bottom:26px;">
          <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px;">
            <div style="font-size:13px; color:#6b7280; margin-bottom:10px;">Review Distribution</div>
            ${canvasToImg('distributionChart') || '<p style="color:#6b7280;"><em>No chart available.</em></p>'}
          </div>
          <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px;">
            <div style="font-size:13px; color:#6b7280; margin-bottom:10px;">AI Review Statistics</div>
            ${canvasToImg('aiStatsChart') || '<p style="color:#6b7280;"><em>No chart available.</em></p>'}
          </div>
        </div>

        <h4 style="margin:0 0 12px 0; color:#8f6b3d;">Temporal & Trust Analytics</h4>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-bottom:26px;">
          <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px;">
            <div style="font-size:13px; color:#6b7280; margin-bottom:10px;">Temporal Fraud Timeline</div>
            ${canvasToImg('temporalChart') || '<p style="color:#6b7280;"><em>No chart available.</em></p>'}
          </div>
          <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px;">
            <div style="font-size:13px; color:#6b7280; margin-bottom:10px;">Trust Score Analytics</div>
            ${canvasToImg('trustScoreChart') || '<p style="color:#6b7280;"><em>No chart available.</em></p>'}
          </div>
        </div>

        <h4 style="margin:0 0 12px 0; color:#8f6b3d;">Network Graph</h4>
        <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px; margin-bottom:26px;">
          <div style="display:grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap:10px; font-size:13px; margin-bottom:10px;">
            <div><span style="color:#6b7280;">Users</span>: <strong>${investigationState.networkData.total_users || 0}</strong></div>
            <div><span style="color:#6b7280;">Products</span>: <strong>${investigationState.networkData.total_products || 0}</strong></div>
            <div><span style="color:#6b7280;">Edges</span>: <strong>${investigationState.networkData.total_edges || 0}</strong></div>
            <div><span style="color:#6b7280;">Communities</span>: <strong>${investigationState.networkData.community_count || 0}</strong></div>
          </div>
          ${canvasToImg('communityGraph') || '<p style="color:#6b7280;"><em>No community chart available.</em></p>'}
        </div>

        <h4 style="margin:0 0 12px 0; color:#8f6b3d;">Top Risk Reviewers</h4>
        ${topReviewers.length ? `
          <table style="width:100%; border-collapse:collapse; font-size:13px; margin-bottom:26px;">
            <thead><tr style="background:#fafaf9;">
              <th style="padding:8px; border-bottom:1px solid #e5e7eb; text-align:left;">Reviewer</th>
              <th style="padding:8px; border-bottom:1px solid #e5e7eb; text-align:left;">Risk Score</th>
              <th style="padding:8px; border-bottom:1px solid #e5e7eb; text-align:left;">Level</th>
              <th style="padding:8px; border-bottom:1px solid #e5e7eb; text-align:left;">Reviews</th>
              <th style="padding:8px; border-bottom:1px solid #e5e7eb; text-align:left;">Avg Rating</th>
            </tr></thead>
            <tbody>
              ${topReviewers.map((r) => `
                <tr>
                  <td style="padding:8px; border-bottom:1px solid #f3f4f6;">${escapeHtml(r.user_id)}</td>
                  <td style="padding:8px; border-bottom:1px solid #f3f4f6; color:#ef4444;">${Number(r.risk_score).toFixed(1)}</td>
                  <td style="padding:8px; border-bottom:1px solid #f3f4f6;">${r.risk_level}</td>
                  <td style="padding:8px; border-bottom:1px solid #f3f4f6;">${r.review_count}</td>
                  <td style="padding:8px; border-bottom:1px solid #f3f4f6;">${Number(r.avg_rating).toFixed(1)}</td>
                </tr>`).join('')}
            </tbody>
          </table>` : '<p style="color:#6b7280;"><em>No reviewer risk data available.</em></p>'}

        <h4 style="margin:0 0 12px 0; color:#8f6b3d;">Detected Reviewer Communities</h4>
        <p style="margin:0 0 26px 0; font-size:14px;">${clusterSummary.length ? clusterSummary.map(c => `${escapeHtml(c.cluster_id)}: ${c.user_count} users, ${c.product_count} products, suspicion ${c.suspicion_score}%`).join('; ') : 'No clusters detected.'}</p>

        <h4 style="margin:0 0 12px 0; color:#8f6b3d;">Recent Temporal Bursts</h4>
        <p style="margin:0 0 26px 0; font-size:14px;">${burstSummary.length ? burstSummary.slice(0, 5).map(b => `${escapeHtml(b.burst_id)} at ${escapeHtml(b.timestamp)} — ${b.review_count} reviews, anomaly ${b.anomaly_score}%`).join('; ') : 'No suspicious bursts detected.'}</p>
      </div>
    `;

    setTimeout(() => {
      window.print();
    }, 700);
  }

  function renderDistributionChart(dashboardData) {
    const genuine = dashboardData.genuine_count || 0;
    const fake = dashboardData.fake_count || 0;
    if (!genuine && !fake) { noData('distributionChart', 'No review distribution data available'); return; }
    const container = document.getElementById('distributionChart');
    if (!container) return;
    container.innerHTML = '<canvas id="distributionChartCanvas"></canvas>';
    const ctx = document.getElementById('distributionChartCanvas');
    if (chartRegistry.distributionChart) { try { chartRegistry.distributionChart.destroy(); } catch (_) {} }
    chartRegistry.distributionChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Genuine', 'Fake'],
        datasets: [{
          data: [genuine, fake],
          backgroundColor: ['#bfae8b', '#8f6b3d'],
          borderColor: ['#f5efe2', '#8f6b3d'],
          borderWidth: 1,
        }],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#4b5563' } } } },
    });
  }

  function renderTemporalChart(temporalData) {
    const hourly = temporalData.hourly_trends || {};
    const labels = Object.keys(hourly).slice(-12);
    const values = labels.map((label) => hourly[label]);
    if (!labels.length || values.every(v => !v)) { noData('temporalChart', 'No temporal timeline data available'); return; }
    const container = document.getElementById('temporalChart');
    if (!container) return;
    container.innerHTML = '<canvas id="temporalChartCanvas"></canvas>';
    const ctx = document.getElementById('temporalChartCanvas');
    if (chartRegistry.temporalChart) { try { chartRegistry.temporalChart.destroy(); } catch (_) {} }
    chartRegistry.temporalChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Hourly Volume',
          data: values,
          borderColor: '#8f6b3d',
          backgroundColor: 'rgba(143, 107, 61, 0.18)',
          fill: true,
          tension: 0.3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#4b5563' } } },
        scales: {
          x: { ticks: { color: '#4b5563' } },
          y: { ticks: { color: '#4b5563' }, beginAtZero: true },
        },
      },
    });
  }

  function renderTrustChart(dashboardData, riskData) {
    const trustScore = dashboardData.average_trust_score || 0;
    const highRisk = riskData.high_risk_count || 0;
    if (!trustScore && !highRisk) { noData('trustScoreChart', 'No trust score analytics available'); return; }
    const container = document.getElementById('trustScoreChart');
    if (!container) return;
    container.innerHTML = '<canvas id="trustScoreChartCanvas"></canvas>';
    const ctx = document.getElementById('trustScoreChartCanvas');
    if (chartRegistry.trustScoreChart) { try { chartRegistry.trustScoreChart.destroy(); } catch (_) {} }
    chartRegistry.trustScoreChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Average Trust %', 'High Risk Reviewers'],
        datasets: [{
          data: [trustScore, highRisk],
          backgroundColor: ['#bfae8b', '#8f6b3d'],
          borderRadius: 8,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, ticks: { color: '#4b5563' } },
          x: { ticks: { color: '#4b5563' } },
        },
      },
    });
  }

  function renderAiChart(dashboardData) {
    const aiStats = dashboardData.ai_stats || {};
    const a = aiStats.human_count || 0;
    const b = aiStats.possibly_ai_count || 0;
    const c = aiStats.highly_likely_ai_count || 0;
    if (!a && !b && !c) { noData('aiStatsChart', 'No AI review statistics available'); return; }
    const container = document.getElementById('aiStatsChart');
    if (!container) return;
    container.innerHTML = '<canvas id="aiStatsChartCanvas"></canvas>';
    const ctx = document.getElementById('aiStatsChartCanvas');
    if (chartRegistry.aiStatsChart) { try { chartRegistry.aiStatsChart.destroy(); } catch (_) {} }
    chartRegistry.aiStatsChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Likely Human', 'Possibly AI', 'Highly Likely AI'],
        datasets: [{
          data: [a, b, c],
          backgroundColor: ['#bfae8b', '#c7b98e', '#8f6b3d'],
          borderWidth: 1,
        }],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#4b5563' } } } },
    });
  }

  function renderNetworkGraph(graphHtml, edgeCount) {
    const container = document.getElementById('networkGraph');
    if (!container) return;
    if (!graphHtml || !edgeCount) {
      noData('networkGraph', 'No network graph data available. Upload CSV with user_id & product_id columns to generate.');
      return;
    }
    container.innerHTML = `<iframe id="investigationPyvisIframe" style="width:100%; height:360px; border:0; border-radius:8px;" title="Fraud Network Graph"></iframe>`;
    const iframe = document.getElementById('investigationPyvisIframe');
    if (iframe) {
      iframe.srcdoc = graphHtml;
    }
  }

  function renderCommunityGraph(networkData) {
    const clusters = networkData.clusters || [];
    if (!clusters.length) { noData('communityGraph', 'No reviewer communities detected yet.'); return; }
    const labels = clusters.map((cluster) => cluster.cluster_id);
    const values = clusters.map((cluster) => cluster.user_count);
    const container = document.getElementById('communityGraph');
    if (!container) return;
    container.innerHTML = '<canvas id="communityGraphCanvas"></canvas>';
    const ctx = document.getElementById('communityGraphCanvas');
    if (chartRegistry.communityGraph) { try { chartRegistry.communityGraph.destroy(); } catch (_) {} }
    chartRegistry.communityGraph = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Reviewers',
          data: values,
          backgroundColor: '#8f6b3d',
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, ticks: { color: '#4b5563' } },
          x: { ticks: { color: '#4b5563' } },
        },
      },
    });
  }

  function escapeHtml(str) {
    if (str == null) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
  }
});
