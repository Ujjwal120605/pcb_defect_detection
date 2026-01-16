import { useState } from 'react'

function App() {
  return (
    <div className="glass-panel">
      <div className="hero-content">
        <h1>PCB Inspection AI</h1>
        <p className="subtitle">
          Advanced PCBA and component defect detection system powered by YOLOv8.
          Automate your quality control with real-time inference and high-precision visual analysis.
        </p>

        <div className="action-area">
          <a href="https://pcbdefectdetection-ansvf2zgpoqrbb4fypvryy.streamlit.app/" className="cta-button">
            Get Started
          </a>
        </div>

        <div className="features-grid">
          <div className="feature-card">
            <h3>🚀 Real-time</h3>
            <span>Live webcam inference with ultra-low latency.</span>
          </div>
          <div className="feature-card">
            <h3>🎯 Accurate</h3>
            <span>Precision defect detection using YOLOv8.</span>
          </div>
          <div className="feature-card">
            <h3>📊 Analytics</h3>
            <span>Comprehensive dashboard and reporting.</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
