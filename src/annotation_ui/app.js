"use strict";

const COLORS = {
  shock: "#ff334f",
  shock_centerline: "#ffffff",
  vortex_core: "#00d9ff",
  expansion_fan: "#b56cff",
  expansion_mixed: "#ff74c8",
  shock_ignore: "#ffb000",
  vortex_ignore: "#a8ff3e",
  expansion_ignore: "#c89962",
  geometry: "#191919",
};
let editable = [];

const dom = Object.fromEntries([
  "canvas", "viewport", "frameSelect", "prevFrame", "nextFrame", "frameMeta",
  "evidenceSelect", "contrast", "contrastValue", "opacity", "opacityValue",
  "zoom", "zoomValue", "brush", "brushValue", "drawMode", "eraseMode",
  "undo", "redo", "reviewer", "reviewStatus", "shockDecision", "vortexDecision",
  "expansionDecision",
  "needsCorrection", "reviewNotes", "saveReview", "saveState", "layerButtons",
].map(id => [id, document.getElementById(id)]));

const ctx = dom.canvas.getContext("2d", { alpha: false });
let manifest;
let frameIndex = 0;
let baseImage = null;
let layerCanvases = {};
let visible = {
  shock: true,
  shock_centerline: true,
  vortex_core: true,
  expansion_fan: true,
  expansion_mixed: false,
  shock_ignore: false,
  vortex_ignore: false,
  expansion_ignore: false,
  geometry: true,
};
let activeLayer = "shock";
let mode = "draw";
let drawing = false;
let lastPoint = null;
let beforeStroke = null;
let history = [];
let redoStack = [];
let dirty = false;
let loadToken = 0;

function setSaveState(text, kind = "") {
  dom.saveState.textContent = text;
  dom.saveState.className = `save-state ${kind}`.trim();
}

function setDirty(value = true) {
  dirty = value;
  if (dirty) setSaveState("تغییرات ذخیره‌نشده", "dirty");
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function currentFrame() { return manifest.frames[frameIndex]; }

async function imageBitmap(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Cannot load ${url}`);
  return createImageBitmap(await response.blob());
}

function emptyLayer() {
  const canvas = document.createElement("canvas");
  canvas.width = manifest.width;
  canvas.height = manifest.height;
  return canvas;
}

async function loadLayer(step, layer) {
  const bitmap = await imageBitmap(`/api/frame/${step}/mask/${layer}.png`);
  const source = emptyLayer();
  const sourceCtx = source.getContext("2d");
  sourceCtx.drawImage(bitmap, 0, 0);
  const sourceData = sourceCtx.getImageData(0, 0, source.width, source.height).data;
  const canvas = emptyLayer();
  const layerCtx = canvas.getContext("2d");
  const colored = layerCtx.createImageData(canvas.width, canvas.height);
  const rgb = hexToRgb(COLORS[layer]);
  for (let i = 0; i < sourceData.length; i += 4) {
    colored.data[i] = rgb.r;
    colored.data[i + 1] = rgb.g;
    colored.data[i + 2] = rgb.b;
    colored.data[i + 3] = sourceData[i];
  }
  layerCtx.putImageData(colored, 0, 0);
  bitmap.close();
  return canvas;
}

async function loadEvidence() {
  const step = currentFrame().step;
  const contrast = Number(dom.contrast.value).toFixed(2);
  baseImage?.close?.();
  baseImage = await imageBitmap(
    `/api/frame/${step}/evidence/${dom.evidenceSelect.value}.png?contrast=${contrast}`
  );
  render();
}

async function loadFrame(nextIndex, force = false) {
  if (dirty && !force && !confirm("تغییرات این فریم ذخیره نشده‌اند. فریم عوض شود؟")) return;
  const token = ++loadToken;
  frameIndex = (nextIndex + manifest.frames.length) % manifest.frames.length;
  const frame = currentFrame();
  setSaveState("در حال بارگذاری…");
  history = [];
  redoStack = [];
  updateHistoryButtons();
  dom.frameSelect.value = String(frame.step);

  const layers = await Promise.all(
    [...editable, "geometry"].map(layer => loadLayer(frame.step, layer))
  );
  if (token !== loadToken) return;
  layerCanvases = Object.fromEntries([...editable, "geometry"].map((x, i) => [x, layers[i]]));
  dom.canvas.width = manifest.width;
  dom.canvas.height = manifest.height;
  applyZoom();

  dom.reviewer.value = frame.reviewer || "";
  dom.reviewStatus.value = frame.status || "unreviewed";
  dom.shockDecision.value = frame.shock_decision || "pending";
  dom.vortexDecision.value = frame.vortex_decision || "pending";
  dom.expansionDecision.value = frame.expansion_decision || "pending";
  dom.needsCorrection.checked = frame.needs_pixel_correction === "yes";
  dom.reviewNotes.value = frame.review_notes || "";
  dom.frameMeta.innerHTML = [
    `<strong>${frame.dataset_id || `frame ${frame.step}`}</strong>`,
    `${frame.case || "case"} · Re=${frame.reynolds ?? "—"} · grid=${frame.grid || "—"}`,
    frame.angle_deg != null ? `angle of attack: ${frame.angle_deg}°` : "",
    frame.source_qualification ? `source: ${frame.source_qualification}` : "",
    frame.evaluation_role ? `role: ${frame.evaluation_role}` : "",
    frame.source_qualification
      ? `quantitative locus audit: ${frame.quantitative_locus_audit_eligible ? "eligible" : "excluded"}`
      : "",
    `source step ${frame.source_step ?? frame.step} · t=${frame.time.toFixed(3)}`,
    frame.selection_roles?.length ? `selection: ${frame.selection_roles.join(", ")}` : "",
    `shock loss: ${frame.shock_loss_eligible ? "eligible" : "QA excluded"}`,
    `proposal tracks: ${frame.provisional_vortex_track_ids.length}`,
    `seed overlap: ${frame.seed_overlap_pixels} px`,
    `working copy: ${frame.saved ? "saved" : "seed masks"}`,
  ].filter(Boolean).join("<br>");
  await loadEvidence();
  dirty = false;
  setSaveState(frame.saved ? "نسخهٔ کاری بارگذاری شد" : "ماسک اولیه بارگذاری شد", "saved");
}

function render() {
  if (!baseImage || !manifest) return;
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  ctx.drawImage(baseImage, 0, 0, manifest.width, manifest.height);
  ctx.globalAlpha = Number(dom.opacity.value);
  for (const layer of [...editable, "geometry"]) {
    if (visible[layer] && layerCanvases[layer]) ctx.drawImage(layerCanvases[layer], 0, 0);
  }
  ctx.globalAlpha = 1;
}

function canvasPoint(event) {
  const rect = dom.canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * manifest.width / rect.width,
    y: (event.clientY - rect.top) * manifest.height / rect.height,
  };
}

function captureAlpha(layer) {
  const data = layerCanvases[layer].getContext("2d").getImageData(
    0, 0, manifest.width, manifest.height
  ).data;
  const alpha = new Uint8ClampedArray(manifest.width * manifest.height);
  for (let i = 0, j = 3; i < alpha.length; i++, j += 4) alpha[i] = data[j];
  return alpha;
}

function restoreAlpha(layer, alpha) {
  const layerCtx = layerCanvases[layer].getContext("2d");
  const image = layerCtx.createImageData(manifest.width, manifest.height);
  const rgb = hexToRgb(COLORS[layer]);
  for (let i = 0, j = 0; i < alpha.length; i++, j += 4) {
    image.data[j] = rgb.r;
    image.data[j + 1] = rgb.g;
    image.data[j + 2] = rgb.b;
    image.data[j + 3] = alpha[i];
  }
  layerCtx.putImageData(image, 0, 0);
}

function hexToRgb(hex) {
  const value = Number.parseInt(hex.slice(1), 16);
  return { r: value >> 16, g: (value >> 8) & 255, b: value & 255 };
}

function drawSegment(from, to) {
  const layerCtx = layerCanvases[activeLayer].getContext("2d");
  layerCtx.save();
  layerCtx.globalCompositeOperation = mode === "draw" ? "source-over" : "destination-out";
  layerCtx.strokeStyle = COLORS[activeLayer];
  layerCtx.fillStyle = COLORS[activeLayer];
  layerCtx.lineCap = "round";
  layerCtx.lineJoin = "round";
  layerCtx.lineWidth = Number(dom.brush.value);
  layerCtx.beginPath();
  layerCtx.moveTo(from.x, from.y);
  layerCtx.lineTo(to.x, to.y);
  layerCtx.stroke();
  layerCtx.restore();
  render();
}

function beginStroke(event) {
  if (event.button !== 0) return;
  event.preventDefault();
  dom.canvas.setPointerCapture(event.pointerId);
  drawing = true;
  lastPoint = canvasPoint(event);
  beforeStroke = captureAlpha(activeLayer);
  drawSegment(lastPoint, lastPoint);
}

function moveStroke(event) {
  if (!drawing) return;
  const point = canvasPoint(event);
  drawSegment(lastPoint, point);
  lastPoint = point;
}

function endStroke(event) {
  if (!drawing) return;
  drawing = false;
  dom.canvas.releasePointerCapture?.(event.pointerId);
  const after = captureAlpha(activeLayer);
  history.push({ layer: activeLayer, before: beforeStroke, after });
  if (history.length > 20) history.shift();
  redoStack = [];
  beforeStroke = null;
  updateHistoryButtons();
  setDirty();
}

function updateHistoryButtons() {
  dom.undo.disabled = history.length === 0;
  dom.redo.disabled = redoStack.length === 0;
}

function undo() {
  const entry = history.pop();
  if (!entry) return;
  restoreAlpha(entry.layer, entry.before);
  redoStack.push(entry);
  updateHistoryButtons();
  setDirty();
  render();
}

function redo() {
  const entry = redoStack.pop();
  if (!entry) return;
  restoreAlpha(entry.layer, entry.after);
  history.push(entry);
  updateHistoryButtons();
  setDirty();
  render();
}

function selectLayer(layer) {
  activeLayer = layer;
  document.querySelectorAll("[data-layer]").forEach(button =>
    button.classList.toggle("active", button.dataset.layer === layer)
  );
}

function selectMode(next) {
  mode = next;
  dom.drawMode.classList.toggle("active", mode === "draw");
  dom.eraseMode.classList.toggle("active", mode === "erase");
}

function applyZoom() {
  if (!manifest) return;
  const scale = Number(dom.zoom.value) / 100;
  dom.canvas.style.width = `${manifest.width * scale}px`;
  dom.canvas.style.height = `${manifest.height * scale}px`;
  dom.zoomValue.value = `${dom.zoom.value}%`;
}

async function saveReview() {
  const frame = currentFrame();
  const masks = Object.fromEntries(editable.map(layer => [
    layer, layerCanvases[layer].toDataURL("image/png")
  ]));
  const payload = {
    review_status: dom.reviewStatus.value,
    needs_pixel_correction: dom.needsCorrection.checked,
    reviewer: dom.reviewer.value,
    review_notes: dom.reviewNotes.value,
    masks,
  };
  if (manifest.review_heads.includes("shock")) payload.shock_decision = dom.shockDecision.value;
  if (manifest.review_heads.includes("vortex")) payload.vortex_decision = dom.vortexDecision.value;
  if (manifest.review_heads.includes("expansion")) payload.expansion_decision = dom.expansionDecision.value;
  setSaveState("در حال ذخیره…");
  dom.saveReview.disabled = true;
  try {
    const result = await fetchJSON(`/api/frame/${frame.step}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    frame.status = result.review_status;
    for (const head of manifest.review_heads) frame[`${head}_decision`] = payload[`${head}_decision`];
    frame.needs_pixel_correction = payload.needs_pixel_correction ? "yes" : "no";
    frame.reviewer = payload.reviewer.trim();
    frame.review_notes = payload.review_notes.trim();
    frame.saved = true;
    dirty = false;
    const removed = result.removed_inside_geometry;
    const removedCount = Object.values(removed).reduce((sum, value) => sum + value, 0);
    const warning = removedCount > 0
      ? `؛ ${removedCount} px داخل هندسه حذف شد`
      : "";
    setSaveState(`ذخیره شد${warning}`, "saved");
    updateFrameOptions();
  } catch (error) {
    setSaveState(error.message, "error");
  } finally {
    dom.saveReview.disabled = false;
  }
}

function updateFrameOptions() {
  for (const option of dom.frameSelect.options) {
    const frame = manifest.frames.find(x => x.step === Number(option.value));
    option.textContent = `${frame.saved ? "●" : "○"} ${frame.step} · ${frame.status}`;
  }
}

function bindEvents() {
  dom.prevFrame.addEventListener("click", () => loadFrame(frameIndex - 1));
  dom.nextFrame.addEventListener("click", () => loadFrame(frameIndex + 1));
  dom.frameSelect.addEventListener("change", () => {
    const index = manifest.frames.findIndex(x => x.step === Number(dom.frameSelect.value));
    loadFrame(index);
  });
  dom.evidenceSelect.addEventListener("change", loadEvidence);
  dom.contrast.addEventListener("input", () => dom.contrastValue.value = Number(dom.contrast.value).toFixed(2));
  dom.contrast.addEventListener("change", loadEvidence);
  dom.opacity.addEventListener("input", () => {
    dom.opacityValue.value = Number(dom.opacity.value).toFixed(2);
    render();
  });
  dom.zoom.addEventListener("input", applyZoom);
  dom.brush.addEventListener("input", () => dom.brushValue.value = `${dom.brush.value} px`);
  dom.layerButtons.addEventListener("click", event => {
    const button = event.target.closest("[data-layer]");
    if (button) selectLayer(button.dataset.layer);
  });
  dom.drawMode.addEventListener("click", () => selectMode("draw"));
  dom.eraseMode.addEventListener("click", () => selectMode("erase"));
  dom.undo.addEventListener("click", undo);
  dom.redo.addEventListener("click", redo);
  dom.canvas.addEventListener("pointerdown", beginStroke);
  dom.canvas.addEventListener("pointermove", moveStroke);
  dom.canvas.addEventListener("pointerup", endStroke);
  dom.canvas.addEventListener("pointercancel", endStroke);
  document.querySelectorAll("[data-visible]").forEach(box => {
    box.addEventListener("change", () => {
      visible[box.dataset.visible] = box.checked;
      render();
    });
  });
  dom.saveReview.addEventListener("click", saveReview);
  [dom.reviewer, dom.reviewStatus, dom.shockDecision, dom.vortexDecision,
    dom.expansionDecision,
    dom.needsCorrection, dom.reviewNotes].forEach(control =>
    control.addEventListener("change", () => setDirty())
  );
  window.addEventListener("beforeunload", event => {
    if (!dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  window.addEventListener("keydown", event => {
    if (event.target.matches("input, textarea, select")) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      event.shiftKey ? redo() : undo();
      return;
    }
    const shortcuts = {
      "1": "shock", "2": "shock_centerline", "3": "vortex_core",
      "4": "expansion_fan", "5": "expansion_mixed", "6": "shock_ignore",
      "7": "vortex_ignore", "8": "expansion_ignore",
    };
    if (shortcuts[event.key] && editable.includes(shortcuts[event.key])) {
      selectLayer(shortcuts[event.key]);
    }
    if (event.key.toLowerCase() === "d") selectMode("draw");
    if (event.key.toLowerCase() === "e") selectMode("erase");
  });
}

async function init() {
  try {
    manifest = await fetchJSON("/api/manifest");
    editable = manifest.editable_layers || manifest.layers.filter(layer => layer !== "geometry");
    manifest.review_heads ||= ["shock", "vortex"];
    document.querySelectorAll("[data-layer]").forEach(control => {
      control.hidden = !editable.includes(control.dataset.layer);
    });
    document.querySelectorAll("[data-visible]").forEach(control => {
      control.closest("label").hidden = !manifest.layers.includes(control.dataset.visible);
    });
    document.querySelectorAll("[data-review-head]").forEach(control => {
      control.hidden = !manifest.review_heads.includes(control.dataset.reviewHead);
    });
    activeLayer = editable.includes("shock") ? "shock" : editable[0];
    selectLayer(activeLayer);
    for (const frame of manifest.frames) {
      const option = document.createElement("option");
      option.value = frame.step;
      dom.frameSelect.appendChild(option);
    }
    updateFrameOptions();
    bindEvents();
    await loadFrame(0, true);
  } catch (error) {
    setSaveState(error.message, "error");
    console.error(error);
  }
}

init();
