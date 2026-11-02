import React, { useEffect, useState, useCallback } from 'react';
import { fetchOverview } from './services/api';
import OverviewPage from './pages/OverviewPage';
import ImageAnalysisPage from './pages/ImageAnalysisPage';
import VideoAnalysisPage from './pages/VideoAnalysisPage';
import ModelDiagnosticsPage from './pages/ModelDiagnosticsPage';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [loadingOverview, setLoadingOverview] = useState(true);
  const [overviewError, setOverviewError] = useState('');

  const loadOverview = useCallback(async () => {
    setLoadingOverview(true);
    setOverviewError('');
    try {
      const data = await fetchOverview();
      setOverview(data);
    } catch (err) {
      setOverviewError(err.message || 'Could not reach backend API');
    } finally {
      setLoadingOverview(false);
    }
  }, []);

  useEffect(() => {
    loadOverview();
  }, [loadOverview]);

  const health = overview?.health;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-badge">YOLOv8m + MOT</span>
          <div>
            <h1>PoultryVision AI — Detection &amp; Behaviour Analysis</h1>
            <p>YOLOv8 Chicken Detection • Centroid Tracking • KDTree Crowding • Low-Mobility Indicator</p>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="Main navigation">
          <button
            type="button"
            className={`nav-btn ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            type="button"
            className={`nav-btn ${activeTab === 'image' ? 'active' : ''}`}
            onClick={() => setActiveTab('image')}
          >
            Image Analysis
          </button>
          <button
            type="button"
            className={`nav-btn ${activeTab === 'video' ? 'active' : ''}`}
            onClick={() => setActiveTab('video')}
          >
            Video Analysis
          </button>
          <button
            type="button"
            className={`nav-btn ${activeTab === 'diagnostics' ? 'active' : ''}`}
            onClick={() => setActiveTab('diagnostics')}
          >
            Model Diagnostics
          </button>
        </nav>

        <div className="status-pills">
          <span className={`pill ${health?.status === 'ok' ? 'ok' : overviewError ? 'err' : 'warn'}`}>
            API: {health?.status === 'ok' ? 'Online' : overviewError ? 'Offline' : 'Checking...'}
          </span>
          <span className={`pill ${health?.model_loaded ? 'ok' : 'warn'}`}>
            Device: {health?.inference_device || '—'}
          </span>
        </div>
      </header>

      <main className="main-container">
        {activeTab === 'overview' && (
          <OverviewPage
            overview={overview}
            error={overviewError}
            loading={loadingOverview}
            onNavigate={setActiveTab}
            onRefresh={loadOverview}
          />
        )}
        {activeTab === 'image' && (
          <ImageAnalysisPage
            sampleImages={overview?.sample_images || []}
            onAnalysisCompleted={loadOverview}
          />
        )}
        {activeTab === 'video' && (
          <VideoAnalysisPage onAnalysisCompleted={loadOverview} />
        )}
        {activeTab === 'diagnostics' && <ModelDiagnosticsPage />}
      </main>
    </div>
  );
}

