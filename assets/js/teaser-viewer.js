import { modelURL, releaseModelURL } from './model-source.js';

const find = (id) => document.getElementById(id);
const trigger = find('simulation-viewer');
const dialog = find('teaser-viewer');
const stage = find('teaser-viewer-stage');
const canvas = find('teaser-viewer-canvas');
const status = find('teaser-viewer-status');
const reset = find('teaser-viewer-reset');
const retry = find('teaser-viewer-retry');
let assets;
let model;
let selected;
let request = 0;
let failed = false;
let download;
let activeURL;
const modelAttempts = new Map();

function loadAssets() {
  // Load the renderer and scene metadata only when the viewer is opened.
  return assets ||= Promise.all([
    import('../vendor/model-viewer.min.js'),
    fetch('data/gallery.json', { cache: 'no-cache' }).then((response) => {
      if (!response.ok) throw new Error('Gallery unavailable');
      return response.json();
    }),
  ]).then(async ([, gallery]) => {
    await customElements.whenDefined('model-viewer');
    customElements.get('model-viewer').meshoptDecoderLocation = new URL('../vendor/meshopt_decoder.js', import.meta.url).href;
    return gallery;
  }).catch((error) => {
    assets = null;
    throw error;
  });
}

function resetView() {
  if (!model || !selected) return;
  const camera = selected.model.camera;
  model.cameraOrbit = camera?.orbit || '0deg 65deg 100%';
  model.cameraTarget = camera?.target || 'auto auto auto';
  model.fieldOfView = camera?.fov || '45deg';
  const distance = camera?.orbit.match(/([\d.]+)m$/)?.[1];
  model.maxCameraOrbit = distance ? `auto 180deg ${Number(distance) * 20}m` : 'auto 180deg 1000%';
  model.jumpCameraToGoal();
}

function showError() {
  failed = true;
  status.textContent = 'The 3D scene could not load. Check your connection and try again.';
  status.hidden = false;
  stage.setAttribute('aria-busy', 'false');
  reset.disabled = true;
  retry.hidden = false;
}

async function openViewer() {
  const current = ++request;
  download?.abort();
  download = new AbortController();
  releaseModelURL(activeURL);
  activeURL = null;
  failed = false;
  canvas.replaceChildren();
  model = selected = null;
  reset.disabled = true;
  retry.hidden = true;
  status.textContent = 'Loading scene…';
  status.hidden = false;
  stage.setAttribute('aria-busy', 'true');
  find('teaser-viewer-title').textContent = '3D reconstruction';
  if (!dialog.open) dialog.showModal();
  const params = new URL(trigger.dataset.viewer, location.href).searchParams;
  try {
    const gallery = await loadAssets();
    if (current !== request || !dialog.open) return;
    // The teaser exposes only the Opus 5 headline comparison.
    const scene = gallery.scenes.find((item) => item.group === 'main' && item.sourceId === params.get('scene'));
    const methodId = params.get('method');
    selected = scene?.methods.find((item) => item.id === methodId && ['ours', 'viga', 'simfoundry'].includes(item.id));
    if (!selected?.model) throw new Error('Reconstruction unavailable');
    find('teaser-viewer-title').textContent = `${selected.label} · 3D reconstruction`;
    model = document.createElement('model-viewer');
    model.setAttribute('camera-controls', '');
    model.setAttribute('touch-action', 'none');
    model.setAttribute('interaction-prompt', 'none');
    model.setAttribute('environment-image', 'neutral');
    model.setAttribute('shadow-intensity', '0');
    model.setAttribute('min-camera-orbit', 'auto 0deg 0.001m');
    model.setAttribute('min-field-of-view', '10deg');
    model.setAttribute('max-field-of-view', '100deg');
    model.setAttribute('loading', 'eager');
    model.alt = `${selected.label} headline reconstruction. Drag to orbit, scroll to zoom, or use the arrow keys.`;
    const progress = document.createElement('div');
    progress.slot = 'progress-bar';
    model.append(progress);
    model.addEventListener('load', () => {
      if (current !== request) return;
      resetView();
      status.hidden = true;
      stage.setAttribute('aria-busy', 'false');
      reset.disabled = false;
    });
    const source = selected.model.src;
    model.addEventListener('error', () => {
      if (current !== request) return;
      // The renderer caches failed loads too; use a fresh URL when retrying.
      modelAttempts.set(source, (modelAttempts.get(source) || 0) + 1);
      showError();
    });
    model.addEventListener('progress', (event) => {
      if (current === request && !failed && !status.hidden) status.textContent = `Loading scene… ${Math.round(event.detail.totalProgress * 100)}%`;
    });
    canvas.append(model);
    resetView();
    const attempt = modelAttempts.get(source) || 0;
    const url = await modelURL(source + (attempt ? `?retry=${attempt}` : ''), download.signal);
    if (current !== request || !dialog.open) { releaseModelURL(url); return; }
    activeURL = url;
    model.src = url;
  } catch {
    if (current === request && dialog.open) showError();
  }
}

trigger.addEventListener('click', openViewer);
reset.addEventListener('click', resetView);
retry.addEventListener('click', openViewer);
find('teaser-viewer-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('keydown', (event) => {
  if (event.key !== 'Tab') return;
  const buttons = [...dialog.querySelectorAll('button:not(:disabled):not([hidden])')];
  const first = buttons[0];
  const last = buttons.at(-1);
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
});
dialog.addEventListener('click', (event) => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
});
dialog.addEventListener('close', () => {
  request++;
  download?.abort();
  canvas.replaceChildren();
  releaseModelURL(activeURL);
  activeURL = null;
  model = selected = null;
  trigger.focus({ preventScroll: true });
});
