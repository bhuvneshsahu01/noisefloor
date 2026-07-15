export const generateSPRTWalk = () => {
  let logLambda = 0;
  const data = [];
  let n = 0;
  
  // Drift towards H1 (Acceptance)
  while (logLambda > -2.99 && logLambda < 2.99 && n < 50) {
    // 60% chance of success (H1), 40% chance of failure (H0)
    const score = Math.random() > 0.4 ? 1 : 0; 
    
    // Log likelihood update (mock formula)
    if (score === 1) {
      logLambda += 0.25;
    } else {
      logLambda -= 0.35;
    }
    
    data.push({
      samples: n,
      logLambda: Number(logLambda.toFixed(2)),
      h0Boundary: -2.99,
      h1Boundary: 2.99
    });
    n++;
  }
  
  return data;
};

export const generateAgentFeed = () => {
  const actions = [
    "Execute SQL Query", 
    "Read File System", 
    "Call External API", 
    "Generate Summarization",
    "Send Email"
  ];
  
  return Array.from({ length: 8 }).map((_, i) => {
    const isDangerous = Math.random() > 0.7;
    return {
      id: `step-${i}`,
      time: new Date(Date.now() - i * 15000).toLocaleTimeString(),
      action: actions[Math.floor(Math.random() * actions.length)],
      conformalScore: Number((Math.random() * (isDangerous ? 0.9 : 0.4)).toFixed(2)),
      status: isDangerous ? 'BLOCKED' : 'APPROVED'
    };
  });
};
