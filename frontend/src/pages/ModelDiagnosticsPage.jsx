import React, { useEffect, useState } from 'react';
import { fetchModelInfo, fetchHealth } from '../services/api';

export default function ModelDiagnosticsPage() {
  const [health, setHealth] = useState(null);
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadDiagnostics = async () => {
    setLoading(true);
    setError('');
    try {
      const [h, m] = await Promise.all([fetchHealth(), fetchModelInfo()]);
      setHealth(h);
      setInfo(m);
    } catch (err) {
      setError(err.message || 'Failed to load model diagnostics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDiagnostics();
  }, []);

  const model = info?.model;
  const training = info?.training_metrics;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.1rem' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.3rem' }}>Model &amp; Runtime Diagnostics</h2>
          <p style={{ margin: '0.2rem 0 0', fontSize: '0.84rem', color: 'var(--text-muted)' }}>
            Verified checkpoint metadata inspected directly from backend/weights/best.pt and active PyTorch runtime.
          </p>
        </div>
        <button className="btn btn-secondary" style={{ width: 'auto' }} onClick={loadDiagnostics} disabled={loading}>
          {loading ? 'Querying...' : 'Refresh Diagnostics'}
        </button>
      </div>

      {error && <div className="alert error">{error}</div>}

      <div className="grid-cards">
        <div className="metric-card">
          <div className="metric-label">Weights File</div>
          <div className="metric-value">{model?.weights_file || '—'}</div>
          <div className="metric-sub">{model ? `${model.weights_size_mb} MB on disk` : 'Unavailable'}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Base Architecture</div>
          <div className="metric-value">{model?.base_model || 'YOLOv8'}</div>
          <div className="metric-sub">Run: {model?.run_name || '—'}</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Active Device</div>
          <div className="metric-value">{model?.device || health?.inference_device || '—'}</div>
          <div className="metric-sub">
            CUDA Available: {health?.cuda_available ? 'Yes' : 'No'}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">GPU Hardware</div>
          <div className="metric-value" style={{ fontSize: '1.15rem' }}>
            {health?.cuda_device_name || 'CPU Only'}
          </div>
          <div className="metric-sub">PyTorch {health?.torch_version || '—'}</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '1.25rem' }}>
        <div className="card">
          <h2>Active Checkpoint Metadata ({model?.weights_file || 'best.pt'})</h2>
          {model ? (
            <table className="data-table">
              <tbody>
                <tr>
                  <th>Task</th>
                  <td>{model.task}</td>
                </tr>
                <tr>
                  <th>Classes ({model.num_classes})</th>
                  <td>
                    {Object.entries(model.class_names || {}).map(([id, name]) => (
                      <span key={id} className="badge green" style={{ marginRight: '0.4rem' }}>
                        ID {id}: {name}
                      </span>
                    ))}
                  </td>
                </tr>
                <tr>
                  <th>Training Run Name</th>
                  <td>{model.run_name || '—'}</td>
                </tr>
                <tr>
                  <th>Training Image Size (imgsz)</th>
                  <td>{model.train_imgsz ?? '—'}</td>
                </tr>
                <tr>
                  <th>Configured Epochs</th>
                  <td>{model.train_epochs ?? '—'}</td>
                </tr>
                <tr>
                  <th>Saved with Ultralytics</th>
                  <td>v{model.ultralytics_version_saved || '—'}</td>
                </tr>
                <tr>
                  <th>Checkpoint Timestamp</th>
                  <td>{model.checkpoint_date || '—'}</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <div className="empty-state">Model metadata unavailable.</div>
          )}
        </div>

        <div className="card">
          <h2>Runtime Environment</h2>
          {health ? (
            <table className="data-table">
              <tbody>
                <tr>
                  <th>Service Health</th>
                  <td>
                    <span className={`badge ${health.status === 'ok' ? 'green' : 'red'}`}>
                      {health.status.toUpperCase()}
                    </span>
                  </td>
                </tr>
                <tr>
                  <th>Operating System</th>
                  <td>{health.platform}</td>
                </tr>
                <tr>
                  <th>Python Interpreter</th>
                  <td>Python {health.python_version} (backend/.venv)</td>
                </tr>
                <tr>
                  <th>PyTorch Version</th>
                  <td>{health.torch_version}</td>
                </tr>
                <tr>
                  <th>Ultralytics Version</th>
                  <td>{health.ultralytics_version}</td>
                </tr>
                <tr>
                  <th>OpenCV Version</th>
                  <td>{health.opencv_version}</td>
                </tr>
                <tr>
                  <th>Behaviour Modules</th>
                  <td>
                    CentroidTracker (motrackers) • SciPy KDTree Density • Euclidean TDP • Sliding-Window Inactivity
                  </td>
                </tr>
              </tbody>
            </table>
          ) : (
            <div className="empty-state">Backend health status unavailable.</div>
          )}
        </div>
      </div>

      <div className="card">
        <h2>Installed YOLOv8 Model Registry</h2>
        <div className="data-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Key</th>
                <th>Weights Checkpoint</th>
                <th>Description</th>
                <th>File Size</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {info?.available_models?.map((m) => (
                <tr key={m.key}>
                  <td><strong>{m.key}</strong></td>
                  <td><code>{m.filename}</code></td>
                  <td>{m.description}</td>
                  <td>{m.file_size_mb ? `${m.file_size_mb} MB` : '—'}</td>
                  <td>
                    {m.installed ? (
                      <span className="badge green">{m.default ? 'Active Default' : 'Ready'}</span>
                    ) : (
                      <span className="badge red">Not Found</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h2>Training Evaluation Log (Optional results.csv)</h2>
        {!training || !training.available ? (
          <div className="alert info" style={{ marginBottom: 0 }}>
            <strong>No results.csv found:</strong> {training?.note || 'Training log not present.'} No precision,
            recall, or mAP metrics are fabricated. If you copy <code>results.csv</code> from your Colab training run into{' '}
            <code>backend/weights/results.csv</code>, this section will automatically display the recorded epoch
            metrics.
          </div>
        ) : (
          <div>
            <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>
              Loaded {training.epochs_recorded} recorded epochs from <code>backend/weights/results.csv</code>.
            </p>
            <div className="data-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    {Object.keys(training.latest_epoch || {}).map((k) => (
                      <th key={k}>{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {training.history.slice(-10).map((row, idx) => (
                    <tr key={idx}>
                      {Object.values(row).map((v, i) => (
                        <td key={i}>{typeof v === 'number' ? Number(v).toFixed(4) : String(v)}</td>
                      ))}
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

