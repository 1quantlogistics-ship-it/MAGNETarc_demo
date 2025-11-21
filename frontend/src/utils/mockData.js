// Mock data generators for development without backend

export const generateMockDesign = (index = 0) => {
  const timestamp = Date.now() - index * 60000;
  const designId = `design_${timestamp}`;

  const params = {
    length_overall: 12 + Math.random() * 6,
    beam: 1.5 + Math.random() * 1,
    draft: 0.3 + Math.random() * 0.4,
    hull_spacing: 2 + Math.random() * 2,
    displacement_kg: 500 + Math.random() * 500,
  };

  const performance = {
    overall: 60 + Math.random() * 40,
    stability: 70 + Math.random() * 30,
    speed: 65 + Math.random() * 35,
    efficiency: 70 + Math.random() * 30,
    drag_coefficient: 0.1 + Math.random() * 0.3,
    metacentric_height: 0.3 + Math.random() * 0.5,
  };

  return {
    design_id: designId,
    timestamp: new Date(timestamp).toISOString(),
    parameters: params,
    performance,
    mesh_url: `/api/mesh/${designId}`,
    hypothesis_type: ['exploration', 'exploitation', 'novelty', 'optimization'][Math.floor(Math.random() * 4)],
    isBreakthrough: Math.random() > 0.9,
    isParetoOptimal: Math.random() > 0.7,
  };
};

export const generateMockHypothesis = () => {
  const types = ['exploration', 'exploitation', 'novelty', 'optimization'];
  const statements = [
    'Increasing hull spacing improves stability but reduces speed',
    'Wider beam ratios enhance metacentric height',
    'Draft optimization balances displacement and drag',
    'Counter-intuitive hull configurations may yield breakthroughs',
  ];

  return {
    id: `hyp_${Date.now()}`,
    type: types[Math.floor(Math.random() * types.length)],
    statement: statements[Math.floor(Math.random() * statements.length)],
    confidence: 0.5 + Math.random() * 0.5,
    designsGenerated: Math.floor(Math.random() * 50),
    timestamp: new Date().toISOString(),
  };
};

export const generateMockPrinciple = () => {
  const categories = ['Stability', 'Speed', 'Efficiency', 'Trade-offs', 'Failures'];
  const statements = [
    'Hull spacing between 2.5-3.2m provides optimal stability',
    'Beam ratios above 8:1 reduce drag coefficient significantly',
    'Draft values below 0.5m maintain displacement efficiency',
    'Increasing length decreases stability but improves speed',
    'Excessive displacement leads to reduced maneuverability',
  ];

  return {
    id: `principle_${Date.now()}_${Math.random()}`,
    category: categories[Math.floor(Math.random() * categories.length)],
    statement: statements[Math.floor(Math.random() * statements.length)],
    confidence: 0.7 + Math.random() * 0.3,
    evidence: Math.floor(10 + Math.random() * 40),
    discovered: new Date(Date.now() - Math.random() * 86400000 * 30).toISOString(),
  };
};

export const generateMockAgentMessage = () => {
  const agents = [
    { id: 'explorer', name: 'Explorer' },
    { id: 'architect', name: 'Architect' },
    { id: 'critic', name: 'Critic' },
    { id: 'supervisor', name: 'Supervisor' },
  ];

  const messages = [
    'Proposing exploration in unconventional hull spacing range',
    'Design feasibility check passed for current hypothesis',
    'Identified potential stability issue in recent design',
    'Recommending focus on efficiency optimization',
    'New design principle discovered from batch analysis',
  ];

  const agent = agents[Math.floor(Math.random() * agents.length)];

  return {
    agentId: agent.id,
    agentName: agent.name,
    message: messages[Math.floor(Math.random() * messages.length)],
    timestamp: new Date().toISOString(),
  };
};

export const generateInitialMockState = () => {
  return {
    meshes: Array.from({ length: 50 }, (_, i) => generateMockDesign(i)),
    agents: [
      { id: 'explorer', name: 'Explorer', status: 'active', confidence: 0.85, actionsCount: 147, domain: ['novelty', 'exploration'] },
      { id: 'architect', name: 'Architect', status: 'thinking', confidence: 0.91, actionsCount: 142, domain: ['design', 'feasibility'] },
      { id: 'critic', name: 'Critic', status: 'idle', confidence: 0.78, actionsCount: 156, domain: ['evaluation', 'analysis'] },
      { id: 'supervisor', name: 'Supervisor', status: 'active', confidence: 0.94, actionsCount: 143, domain: ['decision', 'strategy'] },
    ],
    agentMessages: Array.from({ length: 20 }, generateMockAgentMessage),
    performanceMetrics: {
      overall: 75 + Math.random() * 15,
      stability: 80 + Math.random() * 15,
      speed: 70 + Math.random() * 20,
      efficiency: 75 + Math.random() * 15,
    },
    resources: {
      gpu0: 60 + Math.random() * 30,
      gpu1: 55 + Math.random() * 35,
      memory: 50 + Math.random() * 30,
      cpu: 40 + Math.random() * 40,
    },
  };
};

// Simulate WebSocket message stream
export const mockWebSocketStream = (callback, interval = 5000) => {
  const messageTypes = [
    'agent_message',
    'batch_results',
    'hypothesis_generated',
    'new_mesh',
    'new_principle',
    'resource_update',
  ];

  const sendRandomMessage = () => {
    const type = messageTypes[Math.floor(Math.random() * messageTypes.length)];
    let data = { type, timestamp: new Date().toISOString() };

    switch (type) {
      case 'agent_message':
        data = { ...data, ...generateMockAgentMessage() };
        break;
      case 'batch_results':
        data = { ...data, designs: Array.from({ length: 5 }, (_, i) => generateMockDesign(i)) };
        break;
      case 'hypothesis_generated':
        data = { ...data, hypothesis: generateMockHypothesis() };
        break;
      case 'new_mesh':
        data = { ...data, design: generateMockDesign() };
        break;
      case 'new_principle':
        data = { ...data, principle: generateMockPrinciple() };
        break;
      case 'resource_update':
        data = {
          ...data,
          resources: {
            gpu0: 60 + Math.random() * 30,
            gpu1: 55 + Math.random() * 35,
            memory: 50 + Math.random() * 30,
            cpu: 40 + Math.random() * 40,
          },
        };
        break;
    }

    callback(data);
  };

  const intervalId = setInterval(sendRandomMessage, interval);
  return () => clearInterval(intervalId);
};
