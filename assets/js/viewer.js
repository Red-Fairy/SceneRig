import "../vendor/model-viewer.min.js";
import { modelURL, releaseModelURL } from "./model-source.js";

const $ = (selector) => document.querySelector(selector);
const model = $("#scene-model");
const status = $("#viewer-loading");
const stage = $(".viewer-stage");
const selects = [$("#viewer-group"), $("#viewer-scene"), $("#viewer-method")];
// Keep scene decoding on the same host as the page.
await customElements.whenDefined("model-viewer");
customElements.get("model-viewer").meshoptDecoderLocation = new URL("../vendor/meshopt_decoder.js", import.meta.url).href;
let gallery;
let selected;
let request = 0;
let download;
let activeURL;

function showError(message) {
  status.textContent = message;
  status.hidden = false;
  stage.setAttribute("aria-busy", "false");
  $("#viewer-reset").disabled = true;
  $("#viewer-retry").hidden = false;
}

function options(select, items, value) {
  select.replaceChildren(...items.map(({ id, label }) => new Option(label, id)));
  select.value = value;
  select.disabled = false;
}

function resetView() {
  const camera = selected.model.camera;
  model.cameraOrbit = camera?.orbit || "0deg 65deg 100%";
  model.cameraTarget = camera?.target || "auto auto auto";
  model.fieldOfView = camera?.fov || "45deg";
  // Large support planes must not push the near clipping plane through objects.
  const distance = camera?.orbit.match(/([\d.]+)m$/)?.[1];
  model.maxCameraOrbit = distance
    ? `auto 180deg ${Number(distance) * 20}m` : "auto 180deg 1000%";
  model.jumpCameraToGoal();
}

async function loadSelection(groupId, sourceId, methodId, push = false) {
  const current = ++request;
  download?.abort();
  download = new AbortController();
  model.removeAttribute("src");
  releaseModelURL(activeURL);
  activeURL = null;
  const scene = gallery.scenes.find((item) => item.group === groupId && item.sourceId === sourceId);
  const method = scene?.methods.find((item) => item.id === methodId);
  if (!method?.model) {
    showError("This reconstruction could not be found. Return to the gallery to choose a scene.");
    return;
  }
  selected = method;
  const group = gallery.groups.find((item) => item.id === groupId);
  const sceneIds = [...new Set(gallery.scenes.map((item) => item.sourceId))];
  const sceneLabel = (item) => `Scene ${String(sceneIds.indexOf(item.sourceId) + 1).padStart(2, "0")}`;
  options(selects[0], gallery.groups, groupId);
  options(selects[1], gallery.scenes.filter((item) => item.group === groupId).map((item) => ({ id: item.sourceId, label: sceneLabel(item) })), sourceId);
  options(selects[2], scene.methods, methodId);
  $("#viewer-title").textContent = `${sceneLabel(scene)} · ${method.label}`;
  document.title = `${sceneLabel(scene)} · ${method.label} · SceneRig`;
  $("#viewer-input").src = scene.input;
  $("#viewer-render").src = method.source;
  $("#viewer-render-label").textContent = `${method.label} · Source view`;
  $(".viewer-appearance-note").textContent = "The 3D viewer uses neutral lighting; shading may differ from the rendered comparison."
    + (method.model.displayNote ? ` ${method.model.displayNote}` : "");
  $("#viewer-note").textContent = group.note;
  const params = new URLSearchParams({ group: groupId, scene: sourceId, method: methodId });
  $("#back-to-gallery").href = `./?group=${encodeURIComponent(groupId)}&scene=${encodeURIComponent(sourceId)}#reconstructions`;
  if (push) history.pushState(null, "", `?${params}`);
  status.textContent = "Loading scene…";
  status.hidden = false;
  stage.setAttribute("aria-busy", "true");
  $("#viewer-reset").disabled = true;
  $("#viewer-retry").hidden = true;
  model.alt = `${method.label} reconstruction of ${sceneLabel(scene)}. Drag to orbit, scroll to zoom, or use the arrow keys.`;
  resetView();
  try {
    const source = await modelURL(method.model.src, download.signal);
    if (current !== request) { releaseModelURL(source); return; }
    activeURL = source;
    model.src = source;
  } catch {
    if (current === request) showError("The 3D scene could not load. Check your connection and try again.");
  }
}

function loadURL() {
  const params = new URLSearchParams(location.search);
  loadSelection(params.get("group") || gallery.groups[0].id,
    params.get("scene") || gallery.scenes[0].sourceId,
    params.get("method") || "ours");
}

model.addEventListener("load", () => {
  if (!activeURL || model.src !== activeURL) return;
  resetView();
  status.hidden = true;
  stage.setAttribute("aria-busy", "false");
  $("#viewer-reset").disabled = false;
});
model.addEventListener("error", () => {
  if (activeURL && model.src === activeURL) showError("The 3D scene could not load. Check your connection and try again.");
});
model.addEventListener("progress", (event) => {
  if (!status.hidden) status.textContent = `Loading scene… ${Math.round(event.detail.totalProgress * 100)}%`;
});
$("#viewer-reset").addEventListener("click", resetView);
$("#viewer-retry").addEventListener("click", () => location.reload());
selects.forEach((select) => select.addEventListener("change", () => {
  const scene = gallery.scenes.find((item) => item.group === selects[0].value && item.sourceId === selects[1].value);
  const methodId = scene.methods.some((item) => item.id === selects[2].value) ? selects[2].value : "ours";
  loadSelection(selects[0].value, selects[1].value, methodId, true);
}));
window.addEventListener("popstate", () => gallery && loadURL());

try {
  const response = await fetch("data/gallery.json", { cache: "no-cache" });
  if (!response.ok) throw new Error("Gallery unavailable");
  gallery = await response.json();
  loadURL();
} catch {
  showError("The scene list could not load. Check your connection and try again.");
}
