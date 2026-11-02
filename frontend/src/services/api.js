const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

export function resolveMediaUrl(path) {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  return `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`;
}

async function handleResponse(res) {
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const err = await res.json();
      if (err && err.detail) {
        message = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
      }
    } catch {
      const text = await res.text();
      if (text) message = text;
    }
    throw new Error(message);
  }
  return res.json();
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  return handleResponse(res);
}

export async function fetchOverview() {
  const res = await fetch(`${API_BASE}/api/overview`);
  return handleResponse(res);
}

export async function fetchModelInfo(modelName = 'yolov8m') {
  const url = modelName
    ? `${API_BASE}/api/model/info?model_name=${encodeURIComponent(modelName)}`
    : `${API_BASE}/api/model/info`;
  const res = await fetch(url);
  return handleResponse(res);
}

export async function detectUploadedImage({
  file,
  conf,
  iou,
  densityRadiusPx,
  densityMinNeighbors,
  modelName = 'yolov8m',
  autoEnhance = true,
  forceClahe = false,
  containmentDedup = true,
  proximityThresholdPx = 50.0,
}) {
  const form = new FormData();
  form.append('file', file);
  form.append('conf', String(conf));
  form.append('iou', String(iou));
  form.append('model_name', String(modelName));
  form.append('auto_enhance', String(autoEnhance));
  form.append('force_clahe', String(forceClahe));
  form.append('containment_dedup', String(containmentDedup));
  form.append('proximity_threshold_px', String(proximityThresholdPx));
  if (densityRadiusPx && Number(densityRadiusPx) > 0) {
    form.append('density_radius_px', String(densityRadiusPx));
  }
  form.append('density_min_neighbors', String(densityMinNeighbors));

  const res = await fetch(`${API_BASE}/api/detect/image`, {
    method: 'POST',
    body: form,
  });
  return handleResponse(res);
}

export async function detectSampleImage({
  sampleName,
  conf,
  iou,
  densityRadiusPx,
  densityMinNeighbors,
  modelName = 'yolov8m',
  autoEnhance = true,
  forceClahe = false,
  containmentDedup = true,
  proximityThresholdPx = 50.0,
}) {
  const form = new FormData();
  form.append('conf', String(conf));
  form.append('iou', String(iou));
  form.append('model_name', String(modelName));
  form.append('auto_enhance', String(autoEnhance));
  form.append('force_clahe', String(forceClahe));
  form.append('containment_dedup', String(containmentDedup));
  form.append('proximity_threshold_px', String(proximityThresholdPx));
  if (densityRadiusPx && Number(densityRadiusPx) > 0) {
    form.append('density_radius_px', String(densityRadiusPx));
  }
  form.append('density_min_neighbors', String(densityMinNeighbors));

  const res = await fetch(`${API_BASE}/api/detect/sample/${encodeURIComponent(sampleName)}`, {
    method: 'POST',
    body: form,
  });
  return handleResponse(res);
}

export async function fetchSampleVideos() {
  const res = await fetch(`${API_BASE}/api/samples/videos`);
  return handleResponse(res);
}

export async function fetchVideoProgress(jobId) {
  const res = await fetch(`${API_BASE}/api/analyze/video/progress/${encodeURIComponent(jobId)}`);
  return handleResponse(res);
}

export async function analyzeUploadedVideo({
  file,
  conf,
  iou,
  trackerType = 'bytetrack',
  modelName = 'yolov8m',
  autoEnhance = true,
  containmentDedup = true,
  proximityThresholdPx = 50.0,
  maxLost = 30,
  inactivityWindowS = 10.0,
  inactivityThreshPx,
  inactivityMetric = 'extent',
  densityMinNeighbors = 6,
  maxFrames,
  jobId,
}) {
  const form = new FormData();
  form.append('file', file);
  form.append('conf', String(conf));
  form.append('iou', String(iou));
  form.append('tracker_type', String(trackerType));
  form.append('model_name', String(modelName));
  form.append('auto_enhance', String(autoEnhance));
  form.append('containment_dedup', String(containmentDedup));
  form.append('proximity_threshold_px', String(proximityThresholdPx));
  form.append('max_lost', String(maxLost));
  form.append('inactivity_window_s', String(inactivityWindowS));
  if (inactivityThreshPx !== '' && inactivityThreshPx !== null && inactivityThreshPx !== undefined) {
    form.append('inactivity_thresh_px', String(inactivityThreshPx));
  }
  form.append('inactivity_metric', inactivityMetric);
  form.append('density_min_neighbors', String(densityMinNeighbors));
  if (maxFrames && Number(maxFrames) > 0) {
    form.append('max_frames', String(maxFrames));
  }
  if (jobId) {
    form.append('job_id', jobId);
  }

  const res = await fetch(`${API_BASE}/api/analyze/video`, {
    method: 'POST',
    body: form,
  });
  return handleResponse(res);
}

export async function analyzeSampleVideo({
  sampleName,
  conf,
  iou,
  trackerType = 'bytetrack',
  modelName = 'yolov8m',
  autoEnhance = true,
  containmentDedup = true,
  proximityThresholdPx = 50.0,
  maxLost = 30,
  inactivityWindowS = 10.0,
  inactivityThreshPx,
  inactivityMetric = 'extent',
  densityMinNeighbors = 6,
  maxFrames,
  jobId,
}) {
  const form = new FormData();
  form.append('conf', String(conf));
  form.append('iou', String(iou));
  form.append('tracker_type', String(trackerType));
  form.append('model_name', String(modelName));
  form.append('auto_enhance', String(autoEnhance));
  form.append('containment_dedup', String(containmentDedup));
  form.append('proximity_threshold_px', String(proximityThresholdPx));
  form.append('max_lost', String(maxLost));
  form.append('inactivity_window_s', String(inactivityWindowS));
  if (inactivityThreshPx !== '' && inactivityThreshPx !== null && inactivityThreshPx !== undefined) {
    form.append('inactivity_thresh_px', String(inactivityThreshPx));
  }
  form.append('inactivity_metric', inactivityMetric);
  form.append('density_min_neighbors', String(densityMinNeighbors));
  if (maxFrames && Number(maxFrames) > 0) {
    form.append('max_frames', String(maxFrames));
  }
  if (jobId) {
    form.append('job_id', jobId);
  }

  const res = await fetch(`${API_BASE}/api/analyze/video/sample/${encodeURIComponent(sampleName)}`, {
    method: 'POST',
    body: form,
  });
  return handleResponse(res);
}

