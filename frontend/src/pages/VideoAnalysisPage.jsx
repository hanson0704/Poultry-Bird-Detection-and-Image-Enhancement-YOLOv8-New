import React, { useState, useEffect, useRef } from 'react';
import {
  analyzeUploadedVideo,
  analyzeSampleVideo,
  fetchSampleVideos,
  fetchVideoProgress,
  resolveMediaUrl,
} from '../services/api';

export default function VideoAnalysisPage({ onAnalysisCompleted }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedSample, setSelectedSample] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [sampleVideos, setSampleVideos] = useState([]);

  const [trackerType, setTrackerType] = useState('bytetrack');
  const [modelName, setModelName] = useState('yolov8m');
  const [autoEnhance, setAutoEnhance] = useState(true);
  const [containmentDedup, setContainmentDedup] = useState(true);
  const [proximityThresholdPx, setProximityThresholdPx] = useState(50.0);

  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [maxLost, setMaxLost] = useState(30);
  const [inactivityWindowS, setInactivityWindowS] = useState(10.0);
  const [inactivityThreshPx, setInactivityThreshPx] = useState('');
  const [inactivityMetric, setInactivityMetric] = useState('extent');
  const [densityMinNeighbors, setDensityMinNeighbors] = useState(6);
  const [maxFrames, setMaxFrames] = useState(150);

  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [filterFlaggedOnly, setFilterFlaggedOnly] = useState(false);

  const fileInputRef = useRef(null);
  const pollTimerRef = useRef(null);

  useEffect(() => {
    fetchSampleVideos()
      .then((res) => setSampleVideos(res.samples || []))
      .catch(() => {});
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  const handleFilePicked = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setSelectedSample('');
    setError('');
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleSamplePicked = (name) => {
    setSelectedSample(name);
    setSelectedFile(null);
    setError('');
    setPreviewUrl('');
  };

  const startProgressPolling = (jobId) => {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    setProgress({ frames_done: 0, total_frames: 0, percent: 0 });
    pollTimerRef.current = setInterval(async () => {
      try {
        const p = await fetchVideoProgress(jobId);
        if (p && p.status !== 'unknown') {
          setProgress(p);
        }
      } catch {
        // ignore transient poll errors
      }
    }, 800);
  };

  const stopProgressPolling = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!selectedFile && !selectedSample) {
      setError('Please select a video file to upload or choose a repository sample video.');
      return;
    }

    const jobId = `job_${Date.now()}`;
    setLoading(true);
    setError('');
    startProgressPolling(jobId);

    try {
      const commonParams = {
        conf,
        iou,
        trackerType,
        modelName,
        autoEnhance,
        containmentDedup,
        proximityThresholdPx,
        maxLost,
        inactivityWindowS,
        inactivityThreshPx,
        inactivityMetric,
        densityMinNeighbors,
        maxFrames,
        jobId,
      };
      const data = selectedFile
        ? await analyzeUploadedVideo({ file: selectedFile, ...commonParams })
        : await analyzeSampleVideo({ sampleName: selectedSample, ...commonParams });

      setResult(data);
      if (onAnalysisCompleted) onAnalysisCompleted();
    } catch (err) {
      setError(err.message || 'Video analysis failed');
    } finally {
      stopProgressPolling();
      setLoading(false);
      setProgress(null);
    }
  };

  const tracksToShow = result
    ? result.track_summaries.filter((t) =>
        filterFlaggedOnly ? t.inactivity.low_mobility_indicator : true
      )
    : [];

  return (
    <div className="panel-grid">
      <div className="card">
        <h2>Video Tracking &amp; Behaviour Controls</h2>
        <form onSubmit={handleAnalyze}>
          <div className="dropzone" onClick={() => fileInputRef.current?.click()}>
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.avi,.mov,.mkv,.webm"
              style={{ display: 'none' }}
              onChange={(e) => handleFilePicked(e.target.files?.[0])}
            />
            <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
              {selectedFile
                ? `Upload: ${selectedFile.name}`
                : selectedSample
                ? `Sample: ${selectedSample}`
                : 'Click to select a video file'}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              MP4, AVI, MOV, MKV, WEBM (streamed to disk, up to 500 MB)
            </div>
          </div>

          {sampleVideos.length > 0 && (
            <div className="form-group">
              <label>
                <span>Or select a repository video clip</span>
              </label>
              <select
                className="select"
                value={selectedSample}
                onChange={(e) => handleSamplePicked(e.target.value)}
              >
                <option value="">-- Choose repository sample video --</option>
                {sampleVideos.map((v) => (
                  <option key={v.name} value={v.name}>
                    {v.name} ({v.size_mb} MB)
                  </option>
                ))}
              </select>
              <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                Note: Repository videos in Tracker_Distance_Measure/ have prior green annotations burned in.
              </div>
            </div>
          )}

          <div className="form-group">
            <label>
              <span>Tracking Algorithm</span>
            </label>
            <select
              className="select"
              value={trackerType}
              onChange={(e) => setTrackerType(e.target.value)}
            >
              <option value="bytetrack">ByteTrack (Kalman Filter Velocity, Recommended)</option>
              <option value="centroid">CentroidTracker (Original Baseline Greedy)</option>
            </select>
          </div>

          <div className="form-group">
            <label>
              <span>YOLOv8 Detection Model</span>
            </label>
            <select
              className="select"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
            >
              <option value="yolov8m">YOLOv8m Baseline (Accurate, 49.6 MB)</option>
              <option value="yolov8n">YOLOv8n Lightweight (Fast, 5.9 MB)</option>
            </select>
          </div>

          <div className="form-group" style={{ display: 'flex', gap: '1.2rem', flexWrap: 'wrap', margin: '0.65rem 0' }}>
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
              <span>Inter-Bird Close Proximity Threshold</span>
              <strong>{proximityThresholdPx} px</strong>
            </label>
            <input
              type="range"
              min="10"
              max="200"
              step="5"
              value={proximityThresholdPx}
              onChange={(e) => setProximityThresholdPx(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>
              Distance in pixels below which two chickens are considered in close physical contact or potential pecking.
            </div>
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
              <span>Tracker Max Lost Frames</span>
              <strong>{maxLost} frames</strong>
            </label>
            <input
              type="number"
              className="input"
              min="1"
              max="60"
              value={maxLost}
              onChange={(e) => setMaxLost(parseInt(e.target.value || '5', 10))}
            />
          </div>

          <div className="form-group">
            <label>
              <span>Inactivity Window Duration (seconds)</span>
              <strong>{inactivityWindowS}s</strong>
            </label>
            <input
              type="number"
              className="input"
              min="1"
              max="120"
              step="0.5"
              value={inactivityWindowS}
              onChange={(e) => setInactivityWindowS(parseFloat(e.target.value || '10'))}
            />
          </div>

          <div className="form-group">
            <label>
              <span>Low-Mobility Threshold (px, blank = auto)</span>
            </label>
            <input
              type="number"
              className="input"
              placeholder="Auto (5 px per 480p scale)"
              min="0"
              step="0.5"
              value={inactivityThreshPx}
              onChange={(e) => setInactivityThreshPx(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>
              <span>Inactivity Window Metric</span>
            </label>
            <select
              className="select"
              value={inactivityMetric}
              onChange={(e) => setInactivityMetric(e.target.value)}
            >
              <option value="extent">Spatial Extent (max radius from window mean)</option>
              <option value="path">Cumulative Path Length (sum of frame steps)</option>
            </select>
          </div>

          <div className="form-group">
            <label>
              <span>Max Frames to Process (blank = entire video)</span>
            </label>
            <input
              type="number"
              className="input"
              placeholder="Entire video"
              min="10"
              max="10000"
              value={maxFrames}
              onChange={(e) => setMaxFrames(e.target.value)}
            />
            <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Tip: 260 frames @ 25 FPS = 10.4s (enough for a full 10s inactivity window).
            </div>
          </div>

          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'Processing Video Frames...' : 'Run Video Behaviour Analysis'}
          </button>

          {loading && progress && (
            <div style={{ marginTop: '0.85rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                <span>
                  Frame {progress.frames_done}
                  {progress.total_frames ? ` / ${progress.total_frames}` : ''}
                </span>
                <strong>{progress.percent ?? 0}%</strong>
              </div>
              <div className="progress-bar-wrap">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${Math.min(100, progress.percent || 5)}%` }}
                />
              </div>
            </div>
          )}
        </form>
      </div>

      <div>
        {error && <div className="alert error">{error}</div>}

        {result?.warnings?.map((w, idx) => (
          <div key={idx} className="alert warning">
            <strong>Notice:</strong> {w}
          </div>
        ))}

        {result && (
          <>
            <div className="alert info">
              <strong>{result.inactivity_summary.label}:</strong> {result.inactivity_summary.disclaimer}
            </div>

            <div className="grid-cards">
              <div className="metric-card">
                <div className="metric-label">Total Unique Tracks</div>
                <div className="metric-value">{result.total_unique_tracks}</div>
                <div className="metric-sub">
                  Mean {result.counts_summary.mean_chickens_per_frame} birds/frame (max{' '}
                  {result.counts_summary.max_chickens_in_frame})
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Movement (Pixels)</div>
                <div className="metric-value">{result.movement_summary.mean_distance_px} px</div>
                <div className="metric-sub">
                  Mean speed: {result.movement_summary.mean_speed_px_per_s} px/s • Max:{' '}
                  {result.movement_summary.max_distance_px} px
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Inactivity / Low-Mobility</div>
                <div className="metric-value">
                  {result.inactivity_summary.tracks_with_low_mobility} /{' '}
                  {result.inactivity_summary.tracks_evaluated}
                </div>
                <div className="metric-sub">
                  ≤{result.inactivity_summary.threshold_px} px over {result.inactivity_summary.window_seconds}s (
                  {result.inactivity_summary.metric})
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Dense Crowding</div>
                <div className="metric-value">{result.density_summary.max_dense_chickens}</div>
                <div className="metric-sub">
                  Peak dense birds • {result.density_summary.frames_with_dense_clusters} frames with clusters
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Inter-Bird Spacing (NND)</div>
                <div className="metric-value">
                  {result.interbird_distance_summary?.mean_flock_nearest_neighbor_px !== null &&
                  result.interbird_distance_summary?.mean_flock_nearest_neighbor_px !== undefined
                    ? `${result.interbird_distance_summary.mean_flock_nearest_neighbor_px} px`
                    : '—'}
                </div>
                <div className="metric-sub">
                  Peak close pairs: {result.interbird_distance_summary?.max_close_contact_pairs ?? 0} (≤{result.interbird_distance_summary?.proximity_threshold_px ?? 50} px)
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Processing Speed</div>
                <div className="metric-value">{result.processing_fps} FPS</div>
                <div className="metric-sub">
                  {result.frames_processed} frames ({result.duration_s}s) in {result.processing_time_s}s
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', margin: '0.65rem 0' }}>
              <span className="badge green">Tracker: {result.tracker_used === 'bytetrack' ? 'ByteTrack (Kalman)' : 'CentroidTracker'}</span>
              <span className="badge">Model: {result.model_used}</span>
              <span className="badge">Avg NND: {result.interbird_distance_summary?.mean_flock_nearest_neighbor_px ?? '—'} px</span>
              <span className="badge orange">Peak Close Contacts: {result.interbird_distance_summary?.max_close_contact_pairs ?? 0} pairs</span>
            </div>
          </>
        )}

        <div className="card">
          <h2>
            {result
              ? `Processed Video Playback — ${result.source} (${result.codec})`
              : 'Video Preview'}
          </h2>
          <div className="media-frame">
            {result ? (
              <video
                key={result.processed_video_url}
                src={resolveMediaUrl(result.processed_video_url)}
                controls
                autoPlay
                preload="metadata"
              />
            ) : previewUrl ? (
              <video key={previewUrl} src={previewUrl} controls preload="metadata" />
            ) : (
              <div className="empty-state">
                Upload a video or pick a sample clip on the left and click &ldquo;Run Video Behaviour Analysis&rdquo;.
              </div>
            )}
          </div>
          {result && (
            <div style={{ marginTop: '0.65rem', fontSize: '0.79rem', color: 'var(--text-muted)' }}>
              <strong>Overlay Legend:</strong> Green = active tracked chicken; Orange = chicken inside a high-density
              cluster; Red = flagged by the <strong>Inactivity / Low-Mobility Indicator</strong>.{' '}
              {result.tracking_note}
            </div>
          )}
        </div>

        {result && (
          <div className="card">
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '0.85rem',
                flexWrap: 'wrap',
                gap: '0.5rem',
              }}
            >
              <h2 style={{ margin: 0 }}>
                Per-Track Movement &amp; Inactivity Summary ({tracksToShow.length} shown)
              </h2>
              <label style={{ fontSize: '0.83rem', display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={filterFlaggedOnly}
                  onChange={(e) => setFilterFlaggedOnly(e.target.checked)}
                />
                Show only Low-Mobility Indicator tracks ({result.inactivity_summary.tracks_with_low_mobility})
              </label>
            </div>

            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Track ID</th>
                    <th>Frames</th>
                    <th>Duration (s)</th>
                    <th>Observations</th>
                    <th>Total Distance (px)</th>
                    <th>Mean Speed (px/s)</th>
                    <th>Jump Filters</th>
                    <th>Last Window (px)</th>
                    <th>Mobility Status</th>
                  </tr>
                </thead>
                <tbody>
                  {tracksToShow.map((t) => {
                    const ina = t.inactivity;
                    return (
                      <tr key={t.track_id}>
                        <td>
                          <strong>ID {t.track_id}</strong>
                        </td>
                        <td>
                          {t.first_frame}–{t.last_frame}
                        </td>
                        <td>{t.duration_s}s</td>
                        <td>{t.observations}</td>
                        <td>{t.total_distance_px} px</td>
                        <td>{t.mean_speed_px_per_s} px/s</td>
                        <td>{t.rejected_jumps}</td>
                        <td>{ina.last_window_value_px !== null ? `${ina.last_window_value_px} px` : '—'}</td>
                        <td>
                          {ina.low_mobility_indicator ? (
                            <span className="badge red">
                              Low-Mobility ({ina.low_mobility_windows} win)
                            </span>
                          ) : ina.final_status === 'active' ? (
                            <span className="badge green">Active</span>
                          ) : (
                            <span className="badge muted">{ina.final_status}</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

