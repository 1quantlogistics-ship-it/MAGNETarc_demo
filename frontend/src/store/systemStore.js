import { create } from 'zustand';

export const useSystemStore = create((set, get) => ({
  // System state
  isOnline: false,
  systemMode: 'Autonomous',
  currentCycle: 0,

  // Meshes and designs
  meshes: [],
  currentMesh: null,

  // Agents
  agents: [],
  agentMessages: [],

  // Performance metrics
  performanceMetrics: {
    overall: 0,
    stability: 0,
    speed: 0,
    efficiency: 0,
  },

  // Resources
  resources: {
    gpu0: 0,
    gpu1: 0,
    memory: 0,
    cpu: 0,
  },

  // Knowledge base
  knowledgeBase: {
    principles: [],
    hypotheses: [],
    discoveries: 0,
  },

  // Notifications
  notifications: [],

  // Actions
  setOnline: (status) => set({ isOnline: status }),

  setSystemMode: (mode) => set({ systemMode: mode }),

  incrementCycle: () => set((state) => ({ currentCycle: state.currentCycle + 1 })),

  addMesh: (mesh) => set((state) => ({
    meshes: [mesh, ...state.meshes].slice(0, 100), // Keep last 100
    currentMesh: mesh,
  })),

  setCurrentMesh: (mesh) => set({ currentMesh: mesh }),

  updateAgent: (agentId, updates) => set((state) => ({
    agents: state.agents.map((agent) =>
      agent.id === agentId ? { ...agent, ...updates } : agent
    ),
  })),

  addAgentMessage: (message) => set((state) => ({
    agentMessages: [
      {
        ...message,
        timestamp: message.timestamp || new Date().toISOString(),
      },
      ...state.agentMessages,
    ].slice(0, 500), // Keep last 500
  })),

  updatePerformanceMetrics: (metrics) => set((state) => ({
    performanceMetrics: { ...state.performanceMetrics, ...metrics },
  })),

  updateResources: (resources) => set((state) => ({
    resources: { ...state.resources, ...resources },
  })),

  addPrinciple: (principle) => set((state) => ({
    knowledgeBase: {
      ...state.knowledgeBase,
      principles: [principle, ...state.knowledgeBase.principles],
      discoveries: state.knowledgeBase.discoveries + 1,
    },
  })),

  addHypothesis: (hypothesis) => set((state) => ({
    knowledgeBase: {
      ...state.knowledgeBase,
      hypotheses: [hypothesis, ...state.knowledgeBase.hypotheses],
    },
  })),

  addNotification: (notification) => set((state) => ({
    notifications: [
      {
        ...notification,
        id: notification.id || Date.now().toString(),
        timestamp: notification.timestamp || new Date().toISOString(),
        read: false,
      },
      ...state.notifications,
    ],
  })),

  markNotificationRead: (id) => set((state) => ({
    notifications: state.notifications.map((n) =>
      n.id === id ? { ...n, read: true } : n
    ),
  })),

  clearNotifications: () => set({ notifications: [] }),
}));
