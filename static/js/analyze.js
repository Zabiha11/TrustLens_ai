/**
 * Analyze Review page — Single & Batch prediction with Phase 3 Explainability & AI Detection
 */

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('reviewForm');
  const predictBtn = document.getElementById('predictBtn');
  const productUrlForm = document.getElementById('productUrlForm');
  const productUrlBtn = document.getElementById('productUrlBtn');
  const uploadZone = document.getElementById('uploadZone');
  const csvFile = document.getElementById('csvFile');
  const uploadBtn = document.getElementById('uploadBtn');

  form?.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideBatchResults();
    await runSinglePredict();
  });

  productUrlForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideBatchResults();
    await runProductUrlAnalyze();
  });

  uploadZone?.addEventListener('click', () => csvFile.click());

  uploadZone?.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });

  uploadZone?.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));

  uploadZone?.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      csvFile.files = e.dataTransfer.files;
      onFileSelected();
    }
  });

  csvFile?.addEventListener('change', onFileSelected);
  uploadBtn?.addEventListener('click', runCsvUpload);

  function onFileSelected() {
    const file = csvFile.files[0];
    uploadBtn.disabled = !file;
    if (file) {
      uploadZone.querySelector('p').textContent = file.name;
    }
  }

  async function runSinglePredict() {
    const text = document.getElementById('reviewText').value.trim();
    const rating = parseFloat(document.getElementById('reviewRating').value) || 3;
    const resultSection = document.getElementById('resultSection');
    const singleResult = document.getElementById('singleResult');

    setButtonLoading(predictBtn, true);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_text: text, rating }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed');

      singleResult.innerHTML = renderSingleResult(data);
      resultSection.classList.remove('d-none');
      resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (err) {
      singleResult.innerHTML = '';
      showAlert(form, err.message);
      resultSection.classList.remove('d-none');
    } finally {
      setButtonLoading(predictBtn, false);
    }
  }

  async function runProductUrlAnalyze() {
    const productUrl = document.getElementById('productUrl').value.trim();
    const resultSection = document.getElementById('resultSection');
    const singleResult = document.getElementById('singleResult');
    const urlForm = document.getElementById('productUrlForm');

    setButtonLoading(productUrlBtn, true);
    resultSection.classList.remove('d-none');
    singleResult.innerHTML = `
      <div class="text-center py-5 text-muted">
        <div class="spinner-border text-cyan mb-3" role="status"></div>
        <div><strong>Fetching reviews from:</strong> ${escapeHtml(productUrl)}</div>
        <small class="text-muted">Attempting to extract review content from the product page...</small>
      </div>`;

    try {
      console.log('[URL] POST /api/product-url-analyze with', { product_url: productUrl });
      const res = await fetch('/api/product-url-analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
        },
        cache: 'no-store',
        body: JSON.stringify({ product_url: productUrl }),
      });

      const data = await res.json();
      console.log('[URL] Response:', { ok: res.ok, status: res.status, count: data.count, error: data.error });

      if (!res.ok) {
        const msg = data.error || 'URL analysis failed';
        singleResult.innerHTML = `
          <div class="glass-card border-danger mb-3 p-4">
            <h6 class="text-danger mb-2"><i class="bi bi-exclamation-octagon me-2"></i>Review Extraction Not Possible</h6>
            <p class="mb-0">${escapeHtml(msg)}</p>
            <hr class="border-secondary">
            <p class="small text-muted mb-0">
              <i class="bi bi-info-circle me-1"></i>
              Many e-commerce sites block automated review extraction due to anti-bot protections,
              dynamic JavaScript rendering, or login requirements. For these products, please
              use the CSV Batch Upload option instead.
            </p>
          </div>`;
        showAlert(urlForm, msg);
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        return;
      }

      if (!data.predictions || data.predictions.length === 0) {
        singleResult.innerHTML = `
          <div class="glass-card border-warning mb-3 p-4">
            <h6 class="text-warning mb-2"><i class="bi bi-exclamation-triangle me-2"></i>No Reviews Found</h6>
            <p class="mb-0">The page was loaded but no review content could be extracted. Please upload a CSV instead.</p>
          </div>`;
      } else {
        singleResult.innerHTML = renderBatchResultsFromPayload(data.predictions, `Fetched from ${data.source_url}`);
      }
      resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (err) {
      singleResult.innerHTML = `
        <div class="glass-card border-danger mb-3 p-4">
          <h6 class="text-danger mb-2"><i class="bi bi-x-octagon me-2"></i>Request Failed</h6>
          <p class="mb-0">${escapeHtml(err.message)}</p>
          <p class="small text-muted mb-0 mt-2">Please check your network connection and try again, or use CSV upload.</p>
        </div>`;
      showAlert(urlForm, err.message);
    } finally {
      setButtonLoading(productUrlBtn, false);
    }
  }

  async function runCsvUpload() {
    const file = csvFile.files[0];
    if (!file) return;

    hideSingleResult();
    setButtonLoading(uploadBtn, true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');

      renderBatchResults(data.predictions);
    } catch (err) {
      showAlert(uploadZone.parentElement, err.message);
    } finally {
      setButtonLoading(uploadBtn, false);
    }
  }

  function renderSingleResult(data) {
    const isFake = data.prediction === 'Fake';
    const badgeClass = isFake ? 'result-fake' : 'result-genuine';
    const barColor = isFake ? '#ef4444' : '#22c55e';
    const icon = isFake ? 'exclamation-octagon' : 'shield-check';

    // Phase 3 Extract Data
    const trust = data.trust_score;
    const aiDet = data.ai_detection;
    const lime = data.lime || { words_increasing_fake: [], words_supporting_genuine: [] };
    const shap = data.shap || { top_features: [] };

    // Format LIME Highlighted Text
    const highlightedHtml = formatLimeHighlightedText(data.review_text, lime);

    return `
      <!-- Trust Score & Risk Level Top Banner -->
      <div class="glass-card mb-4 border-${trust.badge_class} bg-dark">
        <div class="row align-items-center g-3">
          <div class="col-md-3 text-center border-end border-secondary pe-md-4">
            <span class="text-muted small uppercase fw-bold d-block mb-1">Unified Trust Score</span>
            <div class="display-4 fw-bold text-${trust.badge_class}">${trust.trust_score}</div>
            <span class="badge bg-${trust.badge_class} px-3 py-1 fs-6 mt-1">${trust.risk_level} (${trust.badge_color})</span>
          </div>
          <div class="col-md-9 ps-md-4">
            <h6 class="text-cyan mb-2"><i class="bi bi-robot me-2"></i>Natural Language Fraud Summary</h6>
            <p class="lead fs-6 mb-0 text-light">${escapeHtml(trust.natural_summary || '')}</p>
          </div>
        </div>
      </div>

      <!-- Result Header & Confidence -->
      <div class="result-header mb-3">
        <span class="result-badge ${badgeClass}">
          <i class="bi bi-${icon} me-1"></i>${data.prediction}
        </span>
        <span class="text-muted">Model Confidence: <strong>${(data.confidence * 100).toFixed(1)}%</strong></span>
      </div>

      <!-- LIME Highlighted Review Display -->
      <div class="card bg-dark border-secondary p-3 mb-4">
        <div class="d-flex justify-content-between align-items-center mb-2">
          <span class="badge bg-secondary"><i class="bi bi-highlighter me-1"></i>LIME Word-Level Attribution</span>
          <small class="text-muted"><span class="badge bg-danger-subtle text-danger me-1">Red: Fake Indicator</span> <span class="badge bg-success-subtle text-success">Green: Genuine Indicator</span></small>
        </div>
        <div class="review-display mb-2 fs-6 leading-relaxed">${highlightedHtml}</div>
        <div class="d-flex justify-content-between small text-muted">
          <span>Rating: ${data.rating} ★ / 5</span>
          <span>Genuine ${(data.probabilities.Genuine * 100).toFixed(1)}% · Fake ${(data.probabilities.Fake * 100).toFixed(1)}%</span>
        </div>
        <div class="confidence-bar mt-2">
          <div class="confidence-fill" style="width:${data.confidence * 100}%; background:${barColor}"></div>
        </div>
      </div>

      <!-- Phase 3 Cards Row: AI Detection & SHAP Features -->
      <div class="row g-4 mb-3">
        <!-- AI Review Detection Card (Groq API) -->
        <div class="col-lg-6">
          <div class="card bg-dark border-cyan h-100 p-3">
            <div class="d-flex justify-content-between align-items-center mb-3">
              <h6 class="text-cyan mb-0"><i class="bi bi-cpu me-2"></i>AI-Generated Review Detection (Groq)</h6>
              <span class="badge ${aiDet.ai_probability >= 0.5 ? 'bg-danger' : 'bg-success'}">${aiDet.status}</span>
            </div>
            <div class="d-flex justify-content-between mb-2 small text-muted">
              <span>AI Probability: <strong>${(aiDet.ai_probability * 100).toFixed(1)}%</strong></span>
              <span>Confidence: ${(aiDet.confidence_score * 100).toFixed(0)}%</span>
            </div>
            <div class="confidence-bar mb-3">
              <div class="confidence-fill ${aiDet.ai_probability >= 0.5 ? 'bg-danger' : 'bg-success'}" style="width:${aiDet.ai_probability * 100}%"></div>
            </div>
            <p class="small text-muted mb-0"><i class="bi bi-info-circle me-1"></i>${escapeHtml(aiDet.explanation)}</p>
          </div>
        </div>

        <!-- SHAP Feature Attributions Card -->
        <div class="col-lg-6">
          <div class="card bg-dark border-cyan h-100 p-3">
            <h6 class="text-cyan mb-3"><i class="bi bi-bar-chart-steps me-2"></i>SHAP Feature Importance</h6>
            <div class="shap-list small">
              ${(shap.top_features || []).slice(0, 5).map(f => `
                <div class="d-flex justify-content-between align-items-center mb-2">
                  <span class="font-monospace text-light">${escapeHtml(f.feature)}</span>
                  <span class="badge ${f.weight > 0 ? 'bg-danger' : 'bg-success'}">${f.weight > 0 ? '+' : ''}${f.weight}</span>
                </div>
              `).join('') || '<span class="text-muted">No SHAP attributions extracted.</span>'}
            </div>
          </div>
        </div>
      </div>

      ${renderDebugPanel(data.debug)}
    `;
  }

  function renderDebugPanel(debug) {
    if (!debug) return '';
    const rows = [
      ['Submitted Review', debug.submitted_review], ['Processed Review', debug.processed_review],
      ['Prediction', debug.prediction || 'See result above'], ['Prediction Probability', debug.prediction_probability],
      ['Trust Score', debug.trust_score || 'See result above'], ['Model Name', debug.model_name],
      ['Vector Shape', (debug.vector_shape || []).join(' × ')], ['Inference Time', `${debug.inference_time_ms} ms`],
      ['Timestamp', debug.timestamp], ['Request ID', debug.request_id],
    ];
    return `<details class="card bg-dark border-secondary p-3 mt-4"><summary class="text-info fw-semibold">Developer Debug Panel</summary><dl class="row small mt-3 mb-0">${rows.map(([label, value]) => `<dt class="col-sm-4 text-muted">${escapeHtml(label)}</dt><dd class="col-sm-8 font-monospace text-break">${escapeHtml(String(value ?? ''))}</dd>`).join('')}</dl></details>`;
  }

  function formatLimeHighlightedText(text, lime) {
    if (!lime) return escapeHtml(text);

    const fakeWords = new Set((lime.words_increasing_fake || []).map(w => w.word.toLowerCase()));
    const genuineWords = new Set((lime.words_supporting_genuine || []).map(w => w.word.toLowerCase()));

    const tokens = text.split(/(\s+)/);
    return tokens.map(token => {
      const cleanToken = token.toLowerCase().replace(/[^a-z0-9]/g, '');
      if (fakeWords.has(cleanToken)) {
        return `<mark class="bg-danger text-white px-1 rounded">${escapeHtml(token)}</mark>`;
      } else if (genuineWords.has(cleanToken)) {
        return `<mark class="bg-success text-white px-1 rounded">${escapeHtml(token)}</mark>`;
      } else {
        return escapeHtml(token);
      }
    }).join('');
  }

  function renderBatchResults(predictions) {
    const section = document.getElementById('batchSection');
    const tbody = document.getElementById('batchTableBody');
    const count = document.getElementById('batchCount');

    count.textContent = `${predictions.length} reviews`;
    tbody.innerHTML = predictions.map((p, i) => `
      <tr>
        <td>${i + 1}</td>
        <td class="text-truncate" style="max-width:300px">${escapeHtml(p.review_text)}</td>
        <td>${p.rating} ★</td>
        <td><span class="badge ${predictionBadgeClass(p.prediction)}">${p.prediction}</span></td>
        <td>${formatConfidence(p.confidence)}</td>
      </tr>
    `).join('');

    section.classList.remove('d-none');
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function renderBatchResultsFromPayload(predictions, titleText = 'Batch Results') {
    const section = document.getElementById('batchSection');
    const tbody = document.getElementById('batchTableBody');
    const count = document.getElementById('batchCount');

    count.textContent = `${predictions.length} reviews`;
    tbody.innerHTML = predictions.map((p, i) => `
      <tr>
        <td>${i + 1}</td>
        <td class="text-truncate" style="max-width:300px">${escapeHtml(p.review_text)}</td>
        <td>${p.rating} ★</td>
        <td><span class="badge ${predictionBadgeClass(p.prediction)}">${p.prediction}</span></td>
        <td>${formatConfidence(p.confidence)}</td>
      </tr>
    `).join('');

    section.classList.remove('d-none');
    section.querySelector('h5').textContent = titleText;
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    return `
      <div class="glass-card">
        <div class="d-flex justify-content-between align-items-center mb-3">
          <h5 class="mb-0"><i class="bi bi-table me-2"></i>${escapeHtml(titleText)}</h5>
          <span class="badge bg-info" id="batchCount">${predictions.length} reviews</span>
        </div>
        <div class="table-responsive">
          <table class="table table-dark table-hover tl-table mb-0">
            <thead><tr><th>#</th><th>Review</th><th>Rating</th><th>Prediction</th><th>Confidence</th></tr></thead>
            <tbody id="batchTableBody">${predictions.map((p, i) => `<tr><td>${i + 1}</td><td class="text-truncate" style="max-width:300px">${escapeHtml(p.review_text)}</td><td>${p.rating} ★</td><td><span class="badge ${predictionBadgeClass(p.prediction)}">${p.prediction}</span></td><td>${formatConfidence(p.confidence)}</td></tr>`).join('')}</tbody>
          </table>
        </div>
      </div>
    `;
  }

  function hideBatchResults() {
    document.getElementById('batchSection')?.classList.add('d-none');
  }

  function hideSingleResult() {
    document.getElementById('resultSection')?.classList.add('d-none');
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
});
