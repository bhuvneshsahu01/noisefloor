import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { AlertCircle, CheckCircle2 } from 'lucide-react';

export default function Overview() {
  const [data, setData] = useState([]);
  const [feed, setFeed] = useState([]);

  useEffect(() => {
    // SSE Stream for SPRT
    const sprtEventSource = new EventSource("http://localhost:8000/api/v1/stream/sprt");
    sprtEventSource.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      setData(prev => {
        // Keep the last 50 samples
        const newData = [...prev, parsed];
        return newData.length > 50 ? newData.slice(newData.length - 50) : newData;
      });
    };

    // SSE Stream for Agent Monitor
    const agentEventSource = new EventSource("http://localhost:8000/api/v1/stream/agents");
    agentEventSource.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      setFeed(prev => {
        // Keep the last 8 events
        const newFeed = [parsed, ...prev];
        return newFeed.length > 8 ? newFeed.slice(0, 8) : newFeed;
      });
    };

    return () => {
      sprtEventSource.close();
      agentEventSource.close();
    };
  }, []);

  return (
    <>
      <div className="glass-panel kpi-card">
        <div className="kpi-title">API Cost Savings</div>
        <div className="kpi-value gradient-text">$12,450</div>
        <div className="kpi-change change-positive">↑ 24% this week</div>
      </div>
      <div className="glass-panel kpi-card">
        <div className="kpi-title">Active Evaluations</div>
        <div className="kpi-value">48</div>
        <div className="kpi-change change-positive">Across 12 Repositories</div>
      </div>
      <div className="glass-panel kpi-card">
        <div className="kpi-title">Agent Interventions</div>
        <div className="kpi-value" style={{ color: 'var(--status-danger)' }}>14</div>
        <div className="kpi-change change-negative">High risk boundaries hit</div>
      </div>

      <div className="glass-panel chart-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '16px', fontWeight: 600 }}>Live SPRT Evaluator (Claude 3.5 Sonnet vs Baseline)</h3>
          <span style={{ fontSize: '12px', background: 'var(--status-success)', color: '#000', padding: '4px 12px', borderRadius: '12px', fontWeight: 600 }}>SHIP VERDICT REACHED</span>
        </div>
        <ResponsiveContainer width="100%" height="85%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="samples" stroke="#8b8b9b" />
            <YAxis stroke="#8b8b9b" />
            <Tooltip contentStyle={{ background: '#0A0A0F', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
            <ReferenceLine y={2.99} label={{ position: 'top', value: 'Accept H1 (Ship)', fill: '#00f2fe' }} stroke="#00f2fe" strokeDasharray="3 3" />
            <ReferenceLine y={-2.99} label={{ position: 'bottom', value: 'Accept H0 (Reject)', fill: '#ff3366' }} stroke="#ff3366" strokeDasharray="3 3" />
            <Line type="stepAfter" dataKey="logLambda" stroke="#b026ff" strokeWidth={3} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="glass-panel feed-card">
        <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Live Conformal Feed</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {feed.map((item, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', padding: '12px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
              {item.status === 'APPROVED' ? <CheckCircle2 color="var(--status-success)" size={20} /> : <AlertCircle color="var(--status-danger)" size={20} />}
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                  <span style={{ fontWeight: 600 }}>{item.action}</span>
                  <span style={{ color: 'var(--text-secondary)' }}>{item.time}</span>
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Uncertainty Score: {item.conformalScore} ({item.status})
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
