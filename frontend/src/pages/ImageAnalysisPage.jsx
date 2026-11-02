import React, { useState, useRef } from 'react';
import {
  detectUploadedImage,
  detectSampleImage,
  resolveMediaUrl,
} from '../services/api';

export default function ImageAnalysisPage({ sampleImages = [], onAnalysisCompleted }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedSample, setSelectedSample] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [dragging, setDragging] = useState(false);

  const [modelName, setModelName] = useState('yolov8m');
  const [autoEnhance, setAutoEnhance] = useState(true);
  const [containmentDedup, setContainmentDedup] = useState(true);

  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [densityRadiusPx, setDensityRadiusPx] = useState('');
  const [densityMinNeighbors, setDensityMinNeighbors] = useState(6);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [viewMode, setViewMode] = useState('annotated');

  const fileInputRef = useRef(null);

  const handleFilePicked = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setSelectedSample('');
    setError('');
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleSamplePicked = (sampleName) => {
    setSelectedSample(sampleName);
    setSelectedFile(null);
    setError('');
    setPreviewUrl(resolveMediaUrl(`/api/samples/image/${encodeURIComponent(sampleName)}`));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFilePicked(file);
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!selectedFile && !selectedSample) {
      setError('Please select an image file or choose a repository test image first.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      let data;
      if (selectedFile) {
        data = await detectUploadedImage({
          file: selectedFile,
          conf,
          iou,
          densityRadiusPx,
          densityMinNeighbors,
          modelName,
          autoEnhance,
          containmentDedup,
        });
      } else {
        data = await detectSampleImage({
          sampleName: selectedSample,
          conf,
          iou,
          densityRadiusPx,
          densityMinNeighbors,
          modelName,
          autoEnhance,
          containmentDedup,
        });
      }
      setResult(data);
      setViewMode('annotated');
      if (onAnalysisCompleted) onAnalysisCompleted();
    } catch (err) {
      setError(err.message || 'Image detection failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel-grid">
      <div className="card">
        <h2>Image Detection Controls</h2>
        <form onSubmit={handleAnalyze}>
          <div
            className={`dropzone ${dragging ? 'dragging' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.bmp,.webp"
              style={{ display: 'none' }}
              onChange={(e) => handleFilePicked(e.target.files?.[0])}
            />
            <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
              {selectedFile
                ? `Selected: ${selectedFile.name}`
                : selectedSample
                ? `Sample: ${selectedSample}`
                : 'Click or drag & drop an image'}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Supports PNG, JPG, JPEG, BMP, WEBP (up to 35 MB)
            </div>
          </div>

          {sampleImages.length > 0 && (
            <div className="form-group">
              <label>
                <span>Or pick a repository test image (img_test/)</span>
              </label>
              <div className="sample-chips">
                {sampleImages.slice(0, 8).map((name) => (
                  <button
                    key={name}
                    type="button"
                    className={`chip ${selectedSample === name ? 'active' : ''}`}
                    onClick={() => handleSamplePicked(name)}
                  >
                    {name}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="form-group">
            <label>
              <span>YOLOv8 Detection Model</span>
            </label>
            <select
              className="input"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              style={{ width: '100%', background: 'var(--surface)' }}
            >
              <option value="yolov8m">YOLOv8m Baseline (Accurate, 49.6 MB)</option>
              <option value="yolov8n">YOLOv8n Lightweight (Fast, 5.9 MB)</option>
            </select>
          </div>

          <div className="form-group" style={{ display: 'flex', gap: '1.2rem', flexWrap: 'wrap', margin: '0.75rem 0' }}>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', cursor: 'pointer', fontSize: '0.85rem' }}>
              <input
                type="checkbox"
                checked={autoEnhance}
                onChange={(e) => setAutoEnhance(e.target.checked)}
              />
              <span>Auto-CLAHE (Low-Light Lift)</span>
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', cursor: 'pointer', fontSize: '0.85rem' }}>
              <input
                type="checkbox"
                checked={containmentDedup}
                onChange={(e) => setContainmentDedup(e.target.checked)}
              />
              <span>Containment Dedup</span>
            </label>
          </div>

          <div className="form-group">
            <label>
              <span>Confidence Threshold (conf)</span>
              <strong>{Number(conf).toFixed(2)}</strong>
            </label>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={conf}
              onChange={(e) => setConf(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          <div className="form-group">
            <label>
              <span>NMS IoU Threshold</span>
              <strong>{Number(iou).toFixed(2)}</strong>
            </label>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={iou}
              onChange={(e) => setIou(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          <div className="form-group">
            <label>
              <span>KDTree Density Radius (px, blank = auto)</span>
            </label>
            <input
              type="number"
              className="input"
              placeholder="Auto (2× median box diagonal)"
              min="10"
              max="2000"
              value={densityRadiusPx}
              onChange={(e) => setDensityRadiusPx(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>
              <span>High-Density Min Chickens in Radius</span>
              <strong>{densityMinNeighbors}</strong>
            </label>
            <input
              type="number"
              className="input"
              min="2"
              max="50"
              value={densityMinNeighbors}
              onChange={(e) => setDensityMinNeighbors(parseInt(e.target.value || '6', 10))}
            />
          </div>

          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'Running YOLOv8 Inference...' : 'Analyse Image'}
          </button>
        </form>
      </div>

      <div>
        {error && <div className="alert error">{error}</div>}

        {result && (
          <div className="grid-cards">
            <div className="metric-card">
              <div className="metric-label">Detected Chickens</div>
              <div className="metric-value">{result.chicken_count}</div>
              <div className="metric-sub">
                {result.image_dimensions.width}×{result.image_dimensions.height} px
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Mean Confidence</div>
              <div className="metric-value">
                {result.confidence_summary.mean !== null
                  ? `${(result.confidence_summary.mean * 100).toFixed(1)}%`
                  : '—'}
              </div>
              <div className="metric-sub">
                {result.confidence_summary.min !== null
                  ? `Range: ${(result.confidence_summary.min * 100).toFixed(0)}% – ${(
                      result.confidence_summary.max * 100
                    ).toFixed(0)}%`
                  : 'No detections'}
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Dense Clusters</div>
              <div className="metric-value">{result.density.dense_cluster_count}</div>
              <div className="metric-sub">
                {result.density.dense_chickens} chickens in dense zones (r=
                {result.density.radius_px ?? '—'} px)
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Inter-Bird Spacing (NND)</div>
              <div className="metric-value">
                {result.interbird_distance?.mean_nearest_neighbor_px !== null &&
                result.interbird_distance?.mean_nearest_neighbor_px !== undefined
                  ? `${result.interbird_distance.mean_nearest_neighbor_px} px`
                  : '—'}
              </div>
              <div className="metric-sub">
                Close contact pairs: {result.interbird_distance?.close_contact_pair_count ?? 0} (≤{result.interbird_distance?.proximity_threshold_px ?? 50} px)
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Inference Time</div>
              <div className="metric-value">{result.inference_time_ms} ms</div>
              <div className="metric-sub">Total pipeline: {result.processing_time_ms} ms</div>
            </div>
          </div>
        )}

        {result && (
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', margin: '0.6rem 0' }}>
            <span className="badge green">Model: {result.model_used || result.parameters?.model_name}</span>
            <span className="badge">Mean Brightness: {result.brightness}</span>
            {result.clahe_applied ? (
              <span className="badge orange">CLAHE: Auto-Applied (Dark Lighting)</span>
            ) : (
              <span className="badge">CLAHE: Not Triggered (Normal Lighting)</span>
            )}
            {result.dedup_removed > 0 ? (
              <span className="badge green">Dedup: {result.dedup_removed} duplicates suppressed</span>
            ) : (
              <span className="badge">Dedup: 0 duplicates</span>
            )}
          </div>
        )}

        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <h2 style={{ margin: 0 }}>
              {result ? `Visual Inspection — ${result.source}` : 'Image Preview'}
            </h2>
            {result && previewUrl && (
              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <button
                  type="button"
                  className={`chip ${viewMode === 'annotated' ? 'active' : ''}`}
                  onClick={() => setViewMode('annotated')}
                >
                  Annotated Detection
                </button>
                <button
                  type="button"
                  className={`chip ${viewMode === 'original' ? 'active' : ''}`}
                  onClick={() => setViewMode('original')}
                >
                  Original Image
                </button>
              </div>
            )}
          </div>

          <div className="media-frame">
            {result && viewMode === 'annotated' ? (
              <img src={resolveMediaUrl(result.annotated_image_url)} alt="Annotated poultry detections" />
            ) : previewUrl ? (
              <img src={previewUrl} alt="Selected input preview" />
            ) : (
              <div className="empty-state">
                Select or drop an image on the left to preview and run YOLOv8 chicken detection.
              </div>
            )}
          </div>

          {result && (
            <div style={{ marginTop: '0.65rem', fontSize: '0.79rem', color: 'var(--text-muted)' }}>
              <strong>Legend:</strong> Green boxes = detected chickens; Orange boxes = chickens with ≥
              {result.density.min_neighbors} birds within {result.density.radius_px} px radius (KDTree centroid
              crowding). {result.density.unit_note}
            </div>
          )}
        </div>

        {result && (
          <div className="card">
            <h2>Spatial Density &amp; Detection Details ({result.detections.length})</h2>
            <div style={{ fontSize: '0.84rem', marginBottom: '0.85rem', color: 'var(--text-muted)' }}>
              Relative image density: <strong>{result.density.chickens_per_megapixel}</strong> chickens/MP • Box area
              coverage: <strong>{(result.density.box_area_coverage_ratio * 100).toFixed(1)}%</strong> • Max local
              neighbours within radius: <strong>{result.density.max_neighbor_count}</strong>
            </div>

            {result.density.dense_clusters.length > 0 && (
              <div style={{ marginBottom: '1rem' }}>
                <h3>Dense Regions ({result.density.dense_clusters.length})</h3>
                <div className="sample-chips">
                  {result.density.dense_clusters.map((cl, idx) => (
                    <span key={idx} className="badge orange">
                      Cluster #{idx + 1}: {cl.size} chickens @ [{cl.region_xyxy.join(', ')}] px
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Class</th>
                    <th>Confidence</th>
                    <th>Centroid (x, y) px</th>
                    <th>Box [x, y, w, h] px</th>
                    <th>Neighbours in Radius</th>
                    <th>Zone</th>
                  </tr>
                </thead>
                <tbody>
                  {result.detections.map((d) => (
                    <tr key={d.id}>
                      <td>{d.id + 1}</td>
                      <td>{d.class_name}</td>
                      <td>{(d.confidence * 100).toFixed(1)}%</td>
                      <td>
                        ({d.centroid[0]}, {d.centroid[1]})
                      </td>
                      <td>[{d.bbox_xywh.join(', ')}]</td>
                      <td>{d.local_neighbors}</td>
                      <td>
                        {d.in_dense_zone ? (
                          <span className="badge orange">High Density</span>
                        ) : (
                          <span className="badge green">Normal</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

