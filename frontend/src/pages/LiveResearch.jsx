import { useState, useEffect } from 'react';
import Card from '../components/shared/Card';
import Badge from '../components/shared/Badge';
import Button from '../components/shared/Button';
import VesselViewer3D from '../components/3d/VesselViewer3D';
import { generateMockDesign, generateMockHypothesis } from '../utils/mockData';
import { Play, Pause, SkipForward } from 'lucide-react';

export default function LiveResearch() {
  const [currentDesign, setCurrentDesign] = useState(generateMockDesign());
  const [hypothesis, setHypothesis] = useState(generateMockHypothesis());
  const [isRunning, setIsRunning] = useState(true);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold gradient-text mb-2">Live Research</h1>
        <p className="text-gray-400">Real-time autonomous research cycle monitoring</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left - 3D Viewer */}
        <div className="lg:col-span-2 space-y-6">
          <Card title="Current Design">
            <VesselViewer3D
              meshUrl={currentDesign.mesh_url}
              performanceData={currentDesign.performance}
              height="600px"
            />
          </Card>

          <Card title="Design Parameters">
            <div className="grid grid-cols-4 gap-4">
              {Object.entries(currentDesign.parameters).map(([key, value]) => (
                <div key={key} className="glass-card p-3 rounded-lg">
                  <p className="text-xs text-gray-400 capitalize mb-1">
                    {key.replace(/_/g, ' ')}
                  </p>
                  <p className="text-lg font-bold text-cyan">{value.toFixed(2)}</p>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Right - Controls and Info */}
        <div className="space-y-6">
          <Card title="Current Hypothesis">
            <Badge variant="info" className="mb-3">{hypothesis.type}</Badge>
            <p className="text-lg mb-4">{hypothesis.statement}</p>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Confidence</span>
              <span className="font-mono text-success">
                {(hypothesis.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="h-2 bg-navy-700 rounded-full overflow-hidden mt-2">
              <div
                className="h-full bg-gradient-to-r from-cyan to-success"
                style={{ width: `${hypothesis.confidence * 100}%` }}
              />
            </div>
          </Card>

          <Card title="Experiment Protocol">
            <div className="space-y-3">
              <div className="glass-card p-3 rounded-lg flex items-center justify-between">
                <span className="text-sm">Generate Designs</span>
                <Badge variant="success">Complete</Badge>
              </div>
              <div className="glass-card p-3 rounded-lg flex items-center justify-between">
                <span className="text-sm">Run Simulations</span>
                <Badge variant="warning">In Progress</Badge>
              </div>
              <div className="glass-card p-3 rounded-lg flex items-center justify-between">
                <span className="text-sm">Evaluate Results</span>
                <Badge variant="info">Pending</Badge>
              </div>
            </div>
          </Card>

          <Card title="Controls">
            <div className="space-y-3">
              <Button
                variant="primary"
                icon={isRunning ? Pause : Play}
                onClick={() => setIsRunning(!isRunning)}
                className="w-full justify-center"
              >
                {isRunning ? 'Pause Cycle' : 'Resume Cycle'}
              </Button>
              <Button variant="secondary" icon={SkipForward} className="w-full justify-center">
                Skip to Next Cycle
              </Button>
            </div>
          </Card>

          <Card title="Real-time Results">
            <div className="space-y-2">
              {Array.from({ length: 5 }, (_, i) => (
                <div key={i} className="glass-card p-2 rounded-lg flex justify-between text-sm">
                  <span className="text-gray-400">Design {i + 1}</span>
                  <span className="font-mono text-cyan">
                    {(60 + Math.random() * 40).toFixed(1)}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
