import { useEffect, useState } from 'react';
import { AlertCircle, CheckCircle2, ShieldAlert } from 'lucide-react';

export default function AgentMonitor() {
  const [feed, setFeed] = useState([]);

  useEffect(() => {
    const agentEventSource = new EventSource("http://localhost:8000/api/v1/stream/agents");
    agentEventSource.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      setFeed(prev => {
        const newFeed = [parsed, ...prev];
        return newFeed.length > 8 ? newFeed.slice(0, 8) : newFeed;
      });
    };

    return () => agentEventSource.close();
  }, []);

  return (
    <div className="glass-panel" style={{ gridColumn: 'span 12', minHeight: '600px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '24px' }}>
        <div>
          <h3 style={{ fontSize: '20px', fontWeight: 600, color: 'var(--text-primary)' }}>Autonomous Agent Traces</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '4px' }}>Monitoring trajectory drift using Conformal Risk Control.</p>
        </div>
        <div style={{ padding: '8px 16px', background: 'rgba(255,51,102,0.1)', border: '1px solid var(--status-danger)', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--status-danger)', fontWeight: 600 }}>
          <ShieldAlert size={18} /> High Risk Detected
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {feed.map((item, i) => (
          <div key={i} style={{ 
            display: 'flex', alignItems: 'center', gap: '16px', padding: '16px', 
            background: item.status === 'BLOCKED' ? 'rgba(255,51,102,0.05)' : 'rgba(0,255,136,0.05)', 
            borderRadius: '12px', border: `1px solid ${item.status === 'BLOCKED' ? 'rgba(255,51,102,0.2)' : 'rgba(0,255,136,0.2)'}` 
          }}>
            {item.status === 'APPROVED' ? <CheckCircle2 color="var(--status-success)" size={24} /> : <AlertCircle color="var(--status-danger)" size={24} />}
            
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: '16px', fontWeight: 600 }}>{item.action}</div>
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>Step #{8 - i} • {item.time}</div>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>Conformal Bound</div>
              <div style={{ fontSize: '18px', fontWeight: 700, color: item.status === 'BLOCKED' ? 'var(--status-danger)' : 'var(--text-primary)' }}>
                {item.conformalScore.toFixed(2)}
              </div>
            </div>
            
            <div style={{ width: '120px', textAlign: 'center', padding: '6px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: 700, background: item.status === 'BLOCKED' ? 'var(--status-danger)' : 'var(--status-success)', color: '#000' }}>
              {item.status}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
