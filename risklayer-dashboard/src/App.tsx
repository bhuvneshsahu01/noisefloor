import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar, AreaChart, Area } from 'recharts';
import { Shield, Activity, Database, Settings, Terminal, Zap, CheckCircle2, XCircle, BarChart3, TrendingUp } from 'lucide-react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [sprtData, setSprtData] = useState<any[]>([]);
  const [traces, setTraces] = useState<any[]>([]);
  const [stats, setStats] = useState({
    evaluations: 0,
    costSaved: '$0.00',
    interventions: 0,
    avgLatency: '42ms'
  });

  // Quant Overfitting CSCV data
  const [pboData, setPboData] = useState<any[]>([]);
  // Bootstrap prompt comparison curve
  const [bootstrapData, setBootstrapData] = useState<any[]>([]);

  useEffect(() => {
    // Generate initial mock data for the chart to look good immediately
    const initialSprt = [];
    let logLambda = 0;
    for (let i = 0; i < 20; i++) {
      logLambda += (Math.random() > 0.4 ? 0.25 : -0.35);
      initialSprt.push({
        samples: i,
        logLambda: Number(logLambda.toFixed(2)),
        h0Boundary: -2.99,
        h1Boundary: 2.99
      });
    }
    setSprtData(initialSprt);

    // Initial mock traces
    setTraces([
      { id: 1, action: "Execute SQL Query", time: "10:45:01 AM", conformalScore: 0.85, status: "BLOCKED" },
      { id: 2, action: "Read Context DB", time: "10:44:50 AM", conformalScore: 0.12, status: "APPROVED" },
      { id: 3, action: "Generate Summary", time: "10:44:12 AM", conformalScore: 0.05, status: "APPROVED" },
      { id: 4, action: "Call Bash Exec", time: "10:43:05 AM", conformalScore: 0.99, status: "BLOCKED" }
    ]);

    setStats({
      evaluations: 14250,
      costSaved: '$1,204.50',
      interventions: 24,
      avgLatency: '38ms'
    });

    // Mock PBO (Probability of Backtest Overfitting) distribution
    setPboData([
      { rank: "0-10%", count: 2 },
      { rank: "10-20%", count: 5 },
      { rank: "20-30%", count: 12 },
      { rank: "30-40%", count: 18 },
      { rank: "40-50%", count: 25 },
      { rank: "50-60%", count: 14 },
      { rank: "60-70%", count: 8 },
      { rank: "70-80%", count: 4 },
      { rank: "80-90%", count: 1 },
      { rank: "90-100%", count: 0 }
    ]);

    // Mock Bootstrap curve for Prompt upgrade comparison (Model A vs B)
    setBootstrapData([
      { delta: -0.04, density: 10 },
      { delta: -0.03, density: 25 },
      { delta: -0.02, density: 55 },
      { delta: -0.01, density: 120 },
      { delta: 0.00, density: 240 },
      { delta: 0.01, density: 410 },
      { delta: 0.02, density: 580 },
      { delta: 0.03, density: 630 },
      { delta: 0.04, density: 490 },
      { delta: 0.05, density: 290 },
      { delta: 0.06, density: 140 },
      { delta: 0.07, density: 45 },
      { delta: 0.08, density: 15 }
    ]);

    // Connect to SPRT Math Stream
    const sprtEventSource = new EventSource('http://localhost:8000/api/v1/stream/sprt');
    sprtEventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setSprtData(prev => {
          const newData = [...prev, data];
          if (newData.length > 50) newData.shift();
          return newData;
        });
        
        // Update stats
        setStats(s => ({
          ...s,
          evaluations: s.evaluations + 1,
        }));
      } catch (e) {
        console.error("Error parsing SPRT stream:", e);
      }
    };

    // Connect to Agent Trace Stream
    const agentEventSource = new EventSource('http://localhost:8000/api/v1/stream/agents');
    agentEventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setTraces(prev => {
          const newTraces = [data, ...prev];
          if (newTraces.length > 8) newTraces.pop();
          return newTraces;
        });
        
        if (data.status === 'BLOCKED') {
          setStats(s => ({
            ...s,
            interventions: s.interventions + 1,
          }));
        }
      } catch (e) {
         console.error("Error parsing Agent stream:", e);
      }
    };

    return () => {
      sprtEventSource.close();
      agentEventSource.close();
    };
  }, []);

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <Shield className="logo-icon" />
          <span className="logo-text">RiskLayer</span>
        </div>
        
        <nav className="nav-menu">
          <div className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}><Activity size={18} /> Overview</div>
          <div className={`nav-item ${activeTab === 'agent_traces' ? 'active' : ''}`} onClick={() => setActiveTab('agent_traces')}><Terminal size={18} /> Agent Traces</div>
          <div className={`nav-item ${activeTab === 'datasets' ? 'active' : ''}`} onClick={() => setActiveTab('datasets')}><Database size={18} /> Datasets</div>
          <div className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`} onClick={() => setActiveTab('settings')}><Settings size={18} /> Settings</div>
        </nav>
      </aside>

      <main className="main-content">
        <header className="header">
          <h1>Enterprise Risk Control</h1>
          <div className="status-badge">
            <div className="status-dot"></div>
            System Online (Streaming)
          </div>
        </header>

        {activeTab === 'overview' && (
          <>
            <div className="stats-grid">
          <div className="stat-card">
            <span className="stat-title">Total Evaluations</span>
            <span className="stat-value">{stats.evaluations.toLocaleString()}</span>
          </div>
          <div className="stat-card">
            <span className="stat-title">Routing Cost Saved</span>
            <span className="stat-value" style={{color: 'var(--accent-green)'}}>{stats.costSaved}</span>
          </div>
          <div className="stat-card">
            <span className="stat-title">Agent Interventions</span>
            <span className="stat-value" style={{color: 'var(--accent-red)'}}>{stats.interventions}</span>
          </div>
          <div className="stat-card">
            <span className="stat-title">Avg Engine Latency</span>
            <span className="stat-value" style={{color: 'var(--accent-blue)'}}><Zap size={16} style={{display: 'inline', marginRight: '4px'}}/>{stats.avgLatency}</span>
          </div>
        </div>

        <div className="charts-grid">
          <div className="chart-card">
            <div className="chart-header">
              <span className="chart-title">Real-Time SPRT Evaluation Walk</span>
              <span className="live-indicator"><div className="status-dot" style={{backgroundColor: 'var(--accent-red)', boxShadow: 'none'}}></div> LIVE</span>
            </div>
            <div style={{ width: '100%', height: 300 }}>
              <ResponsiveContainer>
                <LineChart data={sprtData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="samples" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" domain={[-4, 4]} />
                  <Tooltip contentStyle={{ backgroundColor: '#151518', border: '1px solid #27272a' }} />
                  <ReferenceLine y={2.99} label="H1 (Accept)" stroke="#10b981" strokeDasharray="3 3" />
                  <ReferenceLine y={-2.99} label="H0 (Reject)" stroke="#ef4444" strokeDasharray="3 3" />
                  <Line type="stepAfter" dataKey="logLambda" stroke="#3b82f6" strokeWidth={3} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>


        </div>

        {/* Row 3: Quant Analytics & AI Eval Rigor */}
        <div className="charts-grid-two">
          <div className="chart-card">
            <div className="chart-header">
              <span className="chart-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <BarChart3 size={18} color="var(--accent-purple)" />
                Probability of Backtest Overfitting (PBO CSCV)
              </span>
            </div>
            <div style={{ width: '100%', height: 250 }}>
              <ResponsiveContainer>
                <BarChart data={pboData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="rank" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip contentStyle={{ backgroundColor: '#151518', border: '1px solid #27272a' }} />
                  <Bar dataKey="count" fill="var(--accent-purple)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="chart-card">
            <div className="chart-header">
              <span className="chart-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <TrendingUp size={18} color="var(--accent-blue)" />
                Bootstrap Upgrade Confidence Intervals (Model A vs B)
              </span>
            </div>
            <div style={{ width: '100%', height: 250 }}>
              <ResponsiveContainer>
                <AreaChart data={bootstrapData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="delta" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip contentStyle={{ backgroundColor: '#151518', border: '1px solid #27272a' }} />
                  <ReferenceLine x={0.00} stroke="var(--accent-red)" strokeDasharray="3 3" label="No Improvement" />
                  <Area type="monotone" dataKey="density" stroke="var(--accent-blue)" fill="rgba(59, 130, 246, 0.2)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
          </>
        )}
        {activeTab === 'agent_traces' && (
          <div className="tab-content">
            <div className="tab-header">
              <h2>Agent Traces</h2>
              <p>Live stream of V2 autonomous agent tool calls and conformal boundaries.</p>
            </div>
            <div className="chart-card">
              <div className="agent-traces">
                {traces.map(trace => (
                  <div key={trace.id} className={`trace-item ${trace.status === 'BLOCKED' ? 'blocked' : ''}`}>
                    <div className="trace-info">
                      <span className="trace-action">{trace.action}</span>
                      <span className="trace-time">{trace.time}</span>
                    </div>
                    <div className="trace-score" style={{color: trace.status === 'BLOCKED' ? 'var(--accent-red)' : 'var(--accent-green)'}}>
                      {trace.status === 'BLOCKED' ? <XCircle size={16} style={{display: 'inline', marginRight: '4px', verticalAlign: 'text-bottom'}}/> : <CheckCircle2 size={16} style={{display: 'inline', marginRight: '4px', verticalAlign: 'text-bottom'}}/>}
                      Score: {trace.conformalScore.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'datasets' && (
          <div className="tab-content">
            <div className="tab-header">
              <h2>Calibration Datasets</h2>
              <p>Historical datasets used to train the conformal predictors.</p>
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Dataset Name</th>
                  <th>Samples</th>
                  <th>Last Updated</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>SQL Injection Context</td>
                  <td>14,592</td>
                  <td>Today, 08:30 AM</td>
                  <td style={{color: 'var(--accent-green)'}}>Active</td>
                </tr>
                <tr>
                  <td>PII Data Leakage</td>
                  <td>8,105</td>
                  <td>Yesterday</td>
                  <td style={{color: 'var(--accent-green)'}}>Active</td>
                </tr>
                <tr>
                  <td>Prompt Injection (Jailbreak)</td>
                  <td>22,410</td>
                  <td>Oct 12</td>
                  <td style={{color: 'var(--accent-green)'}}>Active</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="tab-content">
            <div className="tab-header">
              <h2>Engine Settings</h2>
              <p>Configure risk thresholds and API integration keys.</p>
            </div>
            <div className="settings-form">
              <div className="form-group">
                <label>Target Alpha Risk (α)</label>
                <input className="form-input" type="number" defaultValue="0.10" step="0.01" />
              </div>
              <div className="form-group">
                <label>SPRT H1 Upper Bound</label>
                <input className="form-input" type="number" defaultValue="2.99" step="0.01" />
              </div>
              <div className="form-group">
                <label>Groq API Key (Cheap Route)</label>
                <input className="form-input" type="password" defaultValue="gsk_********************" />
              </div>
              <div className="form-group">
                <label>OpenRouter API Key (Premium Route)</label>
                <input className="form-input" type="password" defaultValue="sk-or-v1-**************" />
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

export default App;
