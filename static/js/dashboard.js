/**
 * TrustLens AI — Comprehensive Fraud Intelligence Dashboard Script
 */

document.addEventListener('DOMContentLoaded', () => {
  let pieChart = null;
  let barChart = null;
  let shapImportanceChart = null;
  let aiStatsChart = null;
  let timelineChart = null;
  let burstGraphChart = null;
  let reviewerActivityChart = null;
  let communitySizeChart = null;

  loadDashboard();

  async function loadDashboard() {
    try {
      const res = await fetch('/api/dashboard/stats', {
        cache: 'no-store',
        headers: { 'Cache-Control': 'no-store' },
      });
      if (!res.ok) throw new Error('Failed to fetch stats');
      const data = await res.json();
      updateStats(data);
      renderAllCharts(data);
      renderRecentTable(data.recent_predictions || []);
    } catch (err) {
      console.error('Error loading dashboard stats:', err);
      const tbody = document.getElementById('recentTableBody');
      if (tbody) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-danger text-center py-4">Failed to load dashboard data.</td></tr>';
      }
    }
  }

  function updateStats(data) {
    document.getElementById('statTotal').textContent = (data.total_analyzed || 0).toLocaleString();
    document.getElementById('statFake').textContent = (data.fake_count || 0).toLocaleString();
    document.getElementById('statGenuine').textContent = (data.genuine_count || 0).toLocaleString();
    
    // Phase 2 Stat Cards
    document.getElementById('statAvgTrust').textContent = (data.average_trust_score || 100.0) + '%';
    document.getElementById('statSuspiciousUsers').textContent = (data.suspicious_users || 0).toLocaleString();
    document.getElementById('statDetectedBursts').textContent = (data.detected_bursts || 0).toLocaleString();
    document.getElementById('statCommunities').textContent = (data.reviewer_communities || 0).toLocaleString();
  }

  function renderAllCharts(data) {
    const chartDefaults = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#4b5563', font: { family: 'Inter' } } } },
      scales: {
        x: { ticks: { color: '#4b5563' }, grid: { color: 'rgba(17, 17, 17, 0.08)' } },
        y: { ticks: { color: '#4b5563' }, grid: { color: 'rgba(17, 17, 17, 0.08)' }, beginAtZero: true },
      },
    };

    // 1. Pie / Doughnut Chart
    if (pieChart) pieChart.destroy();
    pieChart = new Chart(document.getElementById('pieChart'), {
      type: 'doughnut',
      data: {
        labels: ['Genuine', 'Fake'],
        datasets: [{
          data: data.total_analyzed ? [data.genuine_count, data.fake_count] : [1, 0],
          backgroundColor: ['rgba(34, 197, 94, 0.8)', 'rgba(239, 68, 68, 0.8)'],
          borderColor: ['#22c55e', '#ef4444'],
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: { legend: { labels: { color: '#4b5563' } } },
      },
    });

    // 2. Bar Chart
    if (barChart) barChart.destroy();
    barChart = new Chart(document.getElementById('barChart'), {
      type: 'bar',
      data: {
        labels: ['Genuine', 'Fake', 'Total'],
        datasets: [{
          data: [data.genuine_count, data.fake_count, data.total_analyzed],
          backgroundColor: ['rgba(34, 197, 94, 0.7)', 'rgba(239, 68, 68, 0.7)', 'rgba(6, 182, 212, 0.7)'],
          borderRadius: 8,
        }],
      },
      options: { ...chartDefaults, plugins: { legend: { display: false } } },
    });

    // 3. SHAP Global Feature Importance
    if (shapImportanceChart) shapImportanceChart.destroy();
    const shapItems = (data.explainability_summary && data.explainability_summary.global_feature_importances) || [];
    const shapLabels = shapItems.slice(0, 8).map(item => item.feature);
    const shapValues = shapItems.slice(0, 8).map(item => item.importance);
    shapImportanceChart = new Chart(document.getElementById('shapImportanceChart'), {
      type: 'bar',
      data: {
        labels: shapLabels.length ? shapLabels : ['No Feature Data'],
        datasets: [{
          label: 'Feature Importance',
          data: shapValues.length ? shapValues : [0],
          backgroundColor: 'rgba(6, 182, 212, 0.8)',
          borderRadius: 6,
        }],
      },
      options: { ...chartDefaults, plugins: { legend: { display: false } } },
    });

    // 4. AI Review Statistics
    if (aiStatsChart) aiStatsChart.destroy();
    const aiStats = data.ai_stats || {};
    const aiLabels = ['Likely Human', 'Possibly AI Generated', 'Highly Likely AI Generated'];
    const aiValues = [
      aiStats.human_count || 0,
      aiStats.possibly_ai_count || 0,
      aiStats.highly_likely_ai_count || 0,
    ];
    aiStatsChart = new Chart(document.getElementById('aiStatsChart'), {
      type: 'doughnut',
      data: {
        labels: aiLabels,
        datasets: [{
          data: aiValues,
          backgroundColor: ['rgba(34, 197, 94, 0.8)', 'rgba(245, 158, 11, 0.8)', 'rgba(239, 68, 68, 0.8)'],
          borderColor: ['#22c55e', '#f59e0b', '#ef4444'],
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#4b5563' } } },
      },
    });

    // 5. Timeline of Reviews Chart
    const temporal = data.temporal_summary || {};
    const hourlyMap = temporal.hourly_trends || {};
    const timelineLabels = Object.keys(hourlyMap);
    const timelineCounts = Object.values(hourlyMap);

    if (timelineChart) timelineChart.destroy();
    timelineChart = new Chart(document.getElementById('timelineChart'), {
      type: 'line',
      data: {
        labels: timelineLabels.length ? timelineLabels : ['No Data'],
        datasets: [{
          label: 'Hourly Volume',
          data: timelineCounts.length ? timelineCounts : [0],
          borderColor: '#06b6d4',
          backgroundColor: 'rgba(6, 182, 212, 0.15)',
          fill: true,
          tension: 0.4,
        }],
      },
      options: chartDefaults,
    });

    // 6. Fraud Burst Frequency Graph
    const bursts = temporal.detected_bursts || [];
    const burstLabels = bursts.map(b => b.burst_id);
    const burstScores = bursts.map(b => b.anomaly_score);

    if (burstGraphChart) burstGraphChart.destroy();
    burstGraphChart = new Chart(document.getElementById('burstGraphChart'), {
      type: 'bar',
      data: {
        labels: burstLabels.length ? burstLabels : ['No Bursts'],
        datasets: [{
          label: 'Anomaly Confidence %',
          data: burstScores.length ? burstScores : [0],
          backgroundColor: 'rgba(239, 68, 68, 0.8)',
          borderRadius: 6,
        }],
      },
      options: chartDefaults,
    });

    // 7. Reviewer Activity Graph
    const riskSummary = data.risk_summary || {};
    const reviewers = riskSummary.reviewers || [];
    const revLabels = reviewers.slice(0, 8).map(r => r.user_id);
    const revCounts = reviewers.slice(0, 8).map(r => r.review_count);

    if (reviewerActivityChart) reviewerActivityChart.destroy();
    reviewerActivityChart = new Chart(document.getElementById('reviewerActivityChart'), {
      type: 'bar',
      data: {
        labels: revLabels.length ? revLabels : ['No Reviewers'],
        datasets: [{
          label: 'Reviews Posted',
          data: revCounts.length ? revCounts : [0],
          backgroundColor: 'rgba(245, 158, 11, 0.8)',
          borderRadius: 6,
        }],
      },
      options: chartDefaults,
    });

    // 8. Community Size Chart (Louvain Clusters)
    const netSummary = data.network_summary || {};
    const clusters = netSummary.clusters || [];
    const clusterLabels = clusters.map(c => c.cluster_id);
    const clusterSizes = clusters.map(c => c.user_count);

    if (communitySizeChart) communitySizeChart.destroy();
    communitySizeChart = new Chart(document.getElementById('communitySizeChart'), {
      type: 'doughnut',
      data: {
        labels: clusterLabels.length ? clusterLabels : ['No Clusters'],
        datasets: [{
          data: clusterSizes.length ? clusterSizes : [1],
          backgroundColor: [
            '#ef4444', '#06b6d4', '#a855f7', '#3b82f6', '#10b981', '#f59e0b'
          ],
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#4b5563' } } },
      },
    });
  }

  function renderRecentTable(recent) {
    const tbody = document.getElementById('recentTableBody');
    if (!tbody) return;

    if (!recent || recent.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-muted text-center py-4">No predictions recorded yet.</td></tr>';
      return;
    }

    tbody.innerHTML = recent.map((row) => `
      <tr>
        <td class="small text-muted">${row.timestamp ? row.timestamp.substring(0, 19).replace('T', ' ') : '—'}</td>
        <td class="text-truncate" style="max-width:280px">${escapeHtml(row.review_preview || '')}</td>
        <td>${row.rating ?? '—'} ★</td>
        <td><span class="badge ${row.prediction === 'Fake' ? 'badge-fake' : 'badge-genuine'}">${row.prediction}</span></td>
        <td>${((row.confidence || 0) * 100).toFixed(1)}%</td>
      </tr>
    `).join('');
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
});
