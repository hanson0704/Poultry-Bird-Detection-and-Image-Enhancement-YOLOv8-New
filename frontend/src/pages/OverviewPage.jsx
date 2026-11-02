import React from 'react';
import { resolveMediaUrl } from '../services/api';

export default function OverviewPage({ overview, error, loading, onNavigate, onRefresh }) {
  const health = overview?.health;
  const lastImg = overview?.latest_image_analysis;
  const lastVid = overview?.latest_video_analysis;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.1rem' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.3rem' }}>System &amp; Analysis Overview</h2>
          <p style={{ margin: '0.2rem 0 0', fontSize: '0.84rem', color: 'var(--text-muted)' }}>
            Live backend status and real telemetry from your most recent image and video analyses.
          </p>
        </div>
        <button className="btn btn-secondary" style={{ width: 'auto' }} onClick={onRefresh} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh Status'}
        </button>
      </div>

      {error && (
        <div className="alert error">
          <strong>Backend unreachable:</strong> {error}. Make sure the FastAPI server is running on port 8000.
        </div>
      )}

      <div className="grid-cards">
        <div className="metric-card">
          <div className="metric-label">Backend Status</div>
          <div className="metric-value">
            {health ? (health.status === 'ok' ? 'Connected' : 'Degraded') : 'Offline'}
          </div>
          <div className="metric-sub">
            {health ? `Python ${health.python_version} • FastAPI` : 'No response from /api/health'}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">YOLOv8 Detector</div>
          <div className="metric-value">{health?.model_loaded ? 'Loaded (best.pt)' : 'Not Loaded'}</div>
          <div className="metric-sub">
            {health?.model_error ? health.model_error : `Ultralytics ${health?.ultralytics_version || '—'}`}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Inference Device</div>
          <div className="metric-value">{health?.inference_device || '—'}</div>
          <div className="metric-sub">
            {health?.cuda_available ? health.cuda_device_name : 'CPU inference mode'}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Latest Chicken Count</div>
          <div className="metric-value">
            {lastImg ? lastImg.chicken_count : lastVid ? lastVid.mean_chickens_per_frame : '—'}
          </div>
          <div className="metric-sub">
            {lastImg
              ? `From image: ${lastImg.source}`
              : lastVid
              ? `Mean/frame in video: ${lastVid.source}`
              : 'No analysis performed in this session yet'}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Dense Regions (Latest)</div>
          <div className="metric-value">
            {lastImg ? lastImg.dense_cluster_count : lastVid ? lastVid.max_dense_chickens : '—'}
          </div>
          <div className="metric-sub">
            {lastImg
              ? `${lastImg.dense_chickens} chickens in dense clusters`
              : lastVid
              ? 'Peak chickens in dense clusters'
              : 'Run an image or video analysis'}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Inactivity / Low-Mobility</div>
          <div className="metric-value">{lastVid ? lastVid.tracks_with_low_mobility : '—'}</div>
          <div className="metric-sub">
            {lastVid
              ? `Out of ${lastVid.total_unique_tracks} total tracks`
              : 'Requires video trajectory analysis'}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '1.25rem' }}>
        <div className="card">
          <h2>Latest Image Analysis</h2>
          {!lastImg ? (
            <div className="empty-state">
              <p>No image analysis has been executed yet.</p>
              <button
                className="btn"
                style={{ width: 'auto', marginTop: '0.5rem' }}
                onClick={() => onNavigate('image')}
              >
                Open Image Analysis
              </button>
            </div>
          ) : (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', marginBottom: '0.75rem' }}>
                <span><strong>Source:</strong> {lastImg.source}</span>
                <span><strong>Detected:</strong> {lastImg.chicken_count} chickens</span>
                <span><strong>Dense clusters:</strong> {lastImg.dense_cluster_count}</span>
              </div>
              <div className="media-frame">
                <img src={resolveMediaUrl(lastImg.annotated_image_url)} alt="Latest annotated detection" />
              </div>
              <div style={{ marginTop: '0.75rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                Processed in {lastImg.processing_time_ms} ms at {lastImg.timestamp}
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <h2>Latest Video Behaviour Analysis</h2>
          {!lastVid ? (
            <div className="empty-state">
              <p>No video behaviour analysis has been executed yet.</p>
              <button
                className="btn"
                style={{ width: 'auto', marginTop: '0.5rem' }}
                onClick={() => onNavigate('video')}
              >
                Open Video Analysis
              </button>
            </div>
          ) : (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <span><strong>Source:</strong> {lastVid.source}</span>
                <span><strong>Unique Tracks:</strong> {lastVid.total_unique_tracks}</span>
                <span><strong>Low-Mobility Tracks:</strong> {lastVid.tracks_with_low_mobility}</span>
              </div>
              <div className="media-frame">
                <video src={resolveMediaUrl(lastVid.processed_video_url)} controls preload="metadata" />
              </div>
              <div style={{ marginTop: '0.75rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                {lastVid.frames_processed} frames ({lastVid.duration_s}s) processed in {lastVid.processing_time_s}s at {lastVid.timestamp}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

