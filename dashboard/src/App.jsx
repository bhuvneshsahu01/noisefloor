import { useState } from 'react';
import { Activity, ShieldCheck, TrendingUp, Settings, Box } from 'lucide-react';
import Overview from './components/Overview';
import AgentMonitor from './components/AgentMonitor';
import SPRTChart from './components/SPRTChart';

function App() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <ShieldCheck size={28} className="gradient-text" color="var(--accent-cyan)" />
          <span>Noisefloor Cloud</span>
        </div>
        
        <nav>
          <div 
            className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            <Activity size={18} /> Overview
          </div>
          <div 
            className={`nav-item ${activeTab === 'agents' ? 'active' : ''}`}
            onClick={() => setActiveTab('agents')}
          >
            <Box size={18} /> Agent Monitor
          </div>
          <div 
            className={`nav-item ${activeTab === 'sprt' ? 'active' : ''}`}
            onClick={() => setActiveTab('sprt')}
          >
            <TrendingUp size={18} /> SPRT Live
          </div>
          
          <div style={{ marginTop: 'auto', paddingTop: '40px' }}>
            <div className="nav-item">
              <Settings size={18} /> Settings
            </div>
          </div>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="header">
          <h2 style={{ fontSize: '18px', fontWeight: 600 }}>
            {activeTab === 'overview' && 'Statistical Reliability Platform'}
            {activeTab === 'agents' && 'Autonomous Agent Traces'}
            {activeTab === 'sprt' && 'Wald\'s Sequential Test Boundaries'}
          </h2>
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', color: 'var(--status-success)' }}>● Live Connection</span>
            <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-purple))' }} />
          </div>
        </header>

        <div className="dashboard-grid">
          {activeTab === 'overview' && <Overview />}
          {activeTab === 'agents' && <AgentMonitor />}
          {activeTab === 'sprt' && <SPRTChart />}
        </div>
      </main>
    </div>
  );
}

export default App;
