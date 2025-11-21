import { useState, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import Card from '../components/shared/Card';
import Badge from '../components/shared/Badge';
import { generateMockDesign } from '../utils/mockData';

export default function ParetoFrontier() {
  const [data, setData] = useState([]);
  const [selectedPoint, setSelectedPoint] = useState(null);

  useEffect(() => {
    const mockData = Array.from({ length: 100 }, () => {
      const design = generateMockDesign();
      return {
        ...design,
        x: design.performance.drag_coefficient,
        y: design.performance.metacentric_height,
      };
    });
    setData(mockData);
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold gradient-text mb-2">Pareto Frontier</h1>
        <p className="text-gray-400">Multi-objective optimization analysis</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-3" title="Design Space Exploration">
          <ResponsiveContainer width="100%" height={500}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#16213e" />
              <XAxis
                type="number"
                dataKey="x"
                name="Drag Coefficient"
                label={{ value: 'Drag Coefficient', position: 'bottom', fill: '#9ca3af' }}
                stroke="#9ca3af"
              />
              <YAxis
                type="number"
                dataKey="y"
                name="Metacentric Height"
                label={{ value: 'Metacentric Height (m)', angle: -90, position: 'left', fill: '#9ca3af' }}
                stroke="#9ca3af"
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="glass-card p-3 rounded-lg">
                        <p className="text-xs font-mono text-gray-400 mb-2">{data.design_id}</p>
                        <div className="space-y-1 text-sm">
                          <div className="flex justify-between gap-4">
                            <span className="text-gray-400">Overall:</span>
                            <span className="text-cyan">{data.performance.overall.toFixed(1)}</span>
                          </div>
                          <div className="flex justify-between gap-4">
                            <span className="text-gray-400">Drag:</span>
                            <span className="text-cyan">{data.x.toFixed(3)}</span>
                          </div>
                          <div className="flex justify-between gap-4">
                            <span className="text-gray-400">Height:</span>
                            <span className="text-cyan">{data.y.toFixed(3)}</span>
                          </div>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter data={data} onClick={setSelectedPoint}>
                {data.map((entry, index) => (
                  <Cell
                    key={index}
                    fill={entry.isParetoOptimal ? '#00ff88' : '#00d9ff'}
                    opacity={entry.isParetoOptimal ? 1 : 0.6}
                    r={entry.isParetoOptimal ? 6 : 4}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>

          <div className="mt-4 flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-success" />
              <span className="text-gray-400">Pareto Optimal</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-cyan opacity-60" />
              <span className="text-gray-400">Non-optimal</span>
            </div>
          </div>
        </Card>

        {selectedPoint && (
          <Card className="lg:col-span-3" title="Selected Design">
            <div className="grid grid-cols-4 gap-4">
              <div className="glass-card p-4 rounded-lg">
                <p className="text-xs text-gray-400 mb-1">Design ID</p>
                <p className="font-mono text-sm text-cyan">{selectedPoint.design_id}</p>
              </div>
              <div className="glass-card p-4 rounded-lg">
                <p className="text-xs text-gray-400 mb-1">Overall Score</p>
                <p className="text-2xl font-bold text-success">
                  {selectedPoint.performance.overall.toFixed(1)}
                </p>
              </div>
              <div className="glass-card p-4 rounded-lg">
                <p className="text-xs text-gray-400 mb-1">Status</p>
                <Badge variant={selectedPoint.isParetoOptimal ? 'success' : 'info'}>
                  {selectedPoint.isParetoOptimal ? 'Pareto Optimal' : 'Non-optimal'}
                </Badge>
              </div>
              <div className="glass-card p-4 rounded-lg">
                <p className="text-xs text-gray-400 mb-1">Type</p>
                <Badge variant="info">{selectedPoint.hypothesis_type}</Badge>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
