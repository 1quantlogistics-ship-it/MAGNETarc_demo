import React from 'react';
import { useSystemStore } from '../store/systemStore';
import Card from '../components/shared/Card';
import Badge from '../components/shared/Badge';
import { Activity, Brain, Eye, Hammer, BookOpen, Zap } from 'lucide-react';

const agentIcons = {
  explorer: Eye,
  architect: Hammer,
  critic: Brain,
  supervisor: Zap,
  historian: BookOpen,
  executor: Activity,
};

export default function AgentMonitor() {
  const { agents, agentMessages } = useSystemStore();

  const mockAgents = agents.length > 0 ? agents : [
    { id: 'explorer', name: 'Explorer', status: 'active', confidence: 0.85, actionsCount: 147, domain: ['novelty', 'exploration'] },
    { id: 'architect', name: 'Architect', status: 'thinking', confidence: 0.91, actionsCount: 142, domain: ['design', 'feasibility'] },
    { id: 'critic', name: 'Critic', status: 'idle', confidence: 0.78, actionsCount: 156, domain: ['evaluation', 'analysis'] },
    { id: 'supervisor', name: 'Supervisor', status: 'active', confidence: 0.94, actionsCount: 143, domain: ['decision', 'strategy'] },
    { id: 'historian', name: 'Historian', status: 'idle', confidence: 0.88, actionsCount: 134, domain: ['knowledge', 'patterns'] },
    { id: 'executor', name: 'Executor', status: 'active', confidence: 0.82, actionsCount: 145, domain: ['simulation', 'execution'] },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold gradient-text mb-2">Agent Monitor</h1>
        <p className="text-gray-400">Multi-agent system activity and performance</p>
      </div>

      {/* Agent Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {mockAgents.map((agent) => {
          const Icon = agentIcons[agent.id] || Activity;
          return (
            <Card key={agent.id}>
              {/* Agent Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-gradient-to-br from-cyan to-cyan-dark rounded-full flex items-center justify-center">
                    <Icon className="w-6 h-6 text-navy-900" />
                  </div>
                  <div>
                    <h3 className="font-semibold">{agent.name}</h3>
                    <Badge variant={
                      agent.status === 'active' ? 'success' :
                      agent.status === 'thinking' ? 'warning' :
                      'info'
                    }>
                      {agent.status}
                    </Badge>
                  </div>
                </div>
              </div>

              {/* Agent Metrics */}
              <div className="space-y-3 mb-4">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Actions Taken</span>
                  <span className="font-mono text-cyan">{agent.actionsCount}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Avg Confidence</span>
                  <span className="font-mono text-success">{(agent.confidence * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <p className="text-sm text-gray-400 mb-2">Current Confidence</p>
                  <div className="h-2 bg-navy-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan to-success"
                      style={{ width: `${agent.confidence * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Domain Expertise */}
              <div>
                <p className="text-xs text-gray-400 mb-2">Domain Expertise</p>
                <div className="flex flex-wrap gap-2">
                  {agent.domain?.map((d, idx) => (
                    <span key={idx} className="px-2 py-1 bg-navy-700 rounded text-xs capitalize">
                      {d}
                    </span>
                  ))}
                </div>
              </div>

              {/* Current Activity */}
              <div className="mt-4 pt-4 border-t border-navy-700">
                <p className="text-xs text-gray-400 mb-2">Current Activity</p>
                <p className="text-sm">
                  {agent.status === 'active' ? 'Processing hypothesis evaluation...' :
                   agent.status === 'thinking' ? 'Analyzing design feasibility...' :
                   'Awaiting next cycle'}
                </p>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Activity Log */}
      <Card title="Activity Log">
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {agentMessages.slice(0, 20).map((msg, idx) => (
            <div key={idx} className="glass-card p-3 rounded-lg">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan to-cyan-dark flex items-center justify-center flex-shrink-0">
                  {agentIcons[msg.agentId] && React.createElement(agentIcons[msg.agentId], { className: 'w-4 h-4 text-navy-900' })}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-sm">{msg.agentName}</span>
                    <span className="text-xs text-gray-400">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                  </div>
                  <p className="text-sm text-gray-300">{msg.message}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
