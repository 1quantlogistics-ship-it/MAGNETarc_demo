import { useState, useEffect } from 'react';
import { useSystemStore } from '../store/systemStore';
import Card from '../components/shared/Card';
import PerformanceGauge from '../components/charts/PerformanceGauge';
import Sparkline from '../components/charts/Sparkline';
import VesselViewer3D from '../components/3d/VesselViewer3D';
import { generateMockDesign, generateMockAgentMessage } from '../utils/mockData';
import { TrendingUp, TrendingDown, Zap } from 'lucide-react';

export default function Dashboard() {
  const { currentMesh, performanceMetrics, currentCycle, resources, agentMessages } =
    useSystemStore();
  const [performanceHistory, setPerformanceHistory] = useState([]);
  const [mockMesh] = useState(generateMockDesign());

  useEffect(() => {
    // Generate mock performance history
    const history = Array.from({ length: 20 }, (_, i) => 60 + Math.random() * 40);
    setPerformanceHistory(history);
  }, []);

  const mockAgents = agentMessages.length > 0
    ? agentMessages
    : Array.from({ length: 10 }, generateMockAgentMessage);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold gradient-text mb-2">System Dashboard</h1>
        <p className="text-gray-400">Real-time overview of autonomous naval research system</p>
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Hero - 3D Vessel Viewer */}
        <Card className="lg:col-span-2" title="Current Best Design">
          <VesselViewer3D
            meshUrl={currentMesh?.mesh_url || mockMesh.mesh_url}
            performanceData={currentMesh?.performance || mockMesh.performance}
            height="500px"
          />
          <div className="mt-4 grid grid-cols-4 gap-3">
            {Object.entries(mockMesh.parameters).slice(0, 4).map(([key, value]) => (
              <div key={key} className="glass-card p-3 rounded-lg">
                <p className="text-xs text-gray-400 capitalize">
                  {key.replace(/_/g, ' ')}
                </p>
                <p className="text-lg font-bold text-cyan">{value.toFixed(2)}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Live Research Cycle */}
        <Card title="Live Research Cycle">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm text-gray-400">Current Cycle</p>
              <p className="text-3xl font-bold text-cyan">#{currentCycle}</p>
            </div>
            <Zap className="w-12 h-12 text-success animate-pulse" />
          </div>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Progress</span>
                <span className="text-cyan">67%</span>
              </div>
              <div className="h-2 bg-navy-700 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-cyan to-success w-2/3" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="glass-card p-2 rounded">
                <p className="text-gray-400">Hypothesis</p>
                <p className="font-semibold">Exploration</p>
              </div>
              <div className="glass-card p-2 rounded">
                <p className="text-gray-400">Designs</p>
                <p className="font-semibold text-cyan">12/20</p>
              </div>
            </div>
          </div>
        </Card>

        {/* Performance Metrics */}
        <Card title="Performance Metrics" className="lg:col-span-1">
          <div className="grid grid-cols-2 gap-6 mb-6">
            <PerformanceGauge
              score={performanceMetrics.overall || mockMesh.performance.overall}
              label="Overall"
              size="md"
            />
            <PerformanceGauge
              score={performanceMetrics.stability || mockMesh.performance.stability}
              label="Stability"
              size="md"
            />
          </div>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Speed</span>
                <span className="font-mono text-cyan">
                  {(performanceMetrics.speed || mockMesh.performance.speed).toFixed(1)}
                </span>
              </div>
              <div className="h-2 bg-navy-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan to-success"
                  style={{ width: `${mockMesh.performance.speed}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Efficiency</span>
                <span className="font-mono text-cyan">
                  {(performanceMetrics.efficiency || mockMesh.performance.efficiency).toFixed(1)}
                </span>
              </div>
              <div className="h-2 bg-navy-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan to-success"
                  style={{ width: `${mockMesh.performance.efficiency}%` }}
                />
              </div>
            </div>
          </div>
        </Card>

        {/* Performance Trend */}
        <Card title="Performance Trend" className="lg:col-span-2">
          <div className="h-32 mb-4">
            <Sparkline data={performanceHistory} color="#00d9ff" height={120} />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-xs text-gray-400 mb-1">Current</p>
              <p className="text-2xl font-bold text-cyan">
                {performanceHistory[performanceHistory.length - 1]?.toFixed(1)}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-gray-400 mb-1">Average</p>
              <p className="text-2xl font-bold text-cyan">
                {(performanceHistory.reduce((a, b) => a + b, 0) / performanceHistory.length).toFixed(1)}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-gray-400 mb-1">Peak</p>
              <p className="text-2xl font-bold text-success">
                {Math.max(...performanceHistory).toFixed(1)}
              </p>
            </div>
          </div>
        </Card>

        {/* Agent Activity Feed */}
        <Card title="Agent Activity Feed" className="lg:col-span-2">
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {mockAgents.slice(0, 10).map((msg, idx) => (
              <div key={idx} className="glass-card p-3 rounded-lg flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan to-cyan-dark flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-bold text-navy-900">
                    {msg.agentName[0]}
                  </span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-sm">{msg.agentName}</span>
                    <span className="text-xs text-gray-400">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-300">{msg.message}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* System Resources */}
        <Card title="System Resources">
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">GPU 0</span>
                <span className="font-mono text-warning">{resources.gpu0.toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-navy-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-warning"
                  style={{ width: `${resources.gpu0}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">GPU 1</span>
                <span className="font-mono text-warning">{resources.gpu1.toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-navy-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-warning"
                  style={{ width: `${resources.gpu1}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Memory</span>
                <span className="font-mono text-cyan">{resources.memory.toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-navy-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan"
                  style={{ width: `${resources.memory}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">CPU</span>
                <span className="font-mono text-success">{resources.cpu.toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-navy-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-success"
                  style={{ width: `${resources.cpu}%` }}
                />
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
