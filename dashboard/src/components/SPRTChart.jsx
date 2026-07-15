import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

export default function SPRTChart() {
  const [data, setData] = useState([]);

  useEffect(() => {
    const sprtEventSource = new EventSource("http://localhost:8000/api/v1/stream/sprt");
    sprtEventSource.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      setData(prev => {
        // When n resets to 0 in the backend, clear our chart
        if (parsed.samples === 0) return [parsed];
        const newData = [...prev, parsed];
        return newData.length > 50 ? newData.slice(newData.length - 50) : newData;
      });
    };

    return () => sprtEventSource.close();
  }, []);

  return (
    <div className="glass-panel" style={{ gridColumn: 'span 12', height: '600px', display: 'flex', flexDirection: 'column' }}>
      <div style={{ marginBottom: '24px' }}>
        <h3 style={{ fontSize: '20px', fontWeight: 600, color: 'var(--text-primary)' }}>Wald's Sequential Probability Ratio Test</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '4px' }}>Log-likelihood random walk between acceptance (H1) and rejection (H0) boundaries.</p>
      </div>
      
      <div style={{ flex: 1, minHeight: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="samples" stroke="#8b8b9b" label={{ value: 'Number of Samples (N)', position: 'bottom', fill: '#8b8b9b' }} />
            <YAxis stroke="#8b8b9b" label={{ value: 'Log Likelihood Ratio (Λ)', angle: -90, position: 'insideLeft', fill: '#8b8b9b' }} domain={[-4, 4]} />
            <Tooltip contentStyle={{ background: '#0A0A0F', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
            <ReferenceLine y={2.99} label={{ position: 'top', value: 'Upper Boundary (H1: Target Rate Achieved)', fill: '#00f2fe' }} stroke="#00f2fe" strokeDasharray="3 3" strokeWidth={2} />
            <ReferenceLine y={-2.99} label={{ position: 'bottom', value: 'Lower Boundary (H0: Baseline Rate Retained)', fill: '#ff3366' }} stroke="#ff3366" strokeDasharray="3 3" strokeWidth={2} />
            <Line type="stepAfter" dataKey="logLambda" stroke="url(#colorUv)" strokeWidth={4} dot={true} isAnimationActive={false} />
            <defs>
              <linearGradient id="colorUv" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="var(--accent-cyan)" />
                <stop offset="100%" stopColor="var(--accent-purple)" />
              </linearGradient>
            </defs>
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
