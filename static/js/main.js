/** Global utilities for TrustLens AI */

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.fade-up').forEach((el, i) => {
    el.style.animationDelay = `${i * 0.05}s`;
  });
});

function showAlert(container, message, type = 'error') {
  const existing = container.querySelector('.tl-alert');
  if (existing) existing.remove();

  const alert = document.createElement('div');
  alert.className = `tl-alert tl-alert-${type}`;
  alert.textContent = message;
  container.appendChild(alert);
}

function setButtonLoading(btn, loading) {
  const text = btn.querySelector('.btn-text');
  const spinner = btn.querySelector('.btn-spinner');
  if (loading) {
    text?.classList.add('d-none');
    spinner?.classList.remove('d-none');
    btn.disabled = true;
  } else {
    text?.classList.remove('d-none');
    spinner?.classList.add('d-none');
    btn.disabled = false;
  }
}

function predictionBadgeClass(prediction) {
  return prediction === 'Fake' ? 'badge-fake' : 'badge-genuine';
}

function formatConfidence(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatTimestamp(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}
