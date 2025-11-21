import { useState } from 'react';
import Card from '../components/shared/Card';
import Button from '../components/shared/Button';
import { Save, RefreshCw } from 'lucide-react';

export default function Settings() {
  const [settings, setSettings] = useState({
    systemMode: 'Autonomous',
    cycleDelay: 30,
    parallelSimulation: true,
    maxConcurrentAgents: 6,
    autoSaveInterval: 60,
    gpuDevice: '0',
    batchSize: 12,
    explorationTemp: 0.7,
    noveltyThreshold: 0.6,
  });

  const handleSave = () => {
    console.log('Saving settings:', settings);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-2">Settings</h1>
          <p className="text-gray-400">Configure system parameters and preferences</p>
        </div>
        <Button icon={Save} onClick={handleSave}>
          Save Changes
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* General Settings */}
        <Card title="General">
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">System Mode</label>
              <select
                className="input"
                value={settings.systemMode}
                onChange={(e) => setSettings({ ...settings, systemMode: e.target.value })}
              >
                <option value="Autonomous">Autonomous</option>
                <option value="Semi-Auto">Semi-Automatic</option>
                <option value="Manual">Manual</option>
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">
                Cycle Delay (seconds)
              </label>
              <input
                type="number"
                className="input"
                value={settings.cycleDelay}
                onChange={(e) => setSettings({ ...settings, cycleDelay: parseInt(e.target.value) })}
              />
            </div>

            <div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.parallelSimulation}
                  onChange={(e) => setSettings({ ...settings, parallelSimulation: e.target.checked })}
                  className="w-4 h-4"
                />
                <span className="text-sm">Enable Parallel Simulation</span>
              </label>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">
                Max Concurrent Agents
              </label>
              <input
                type="number"
                className="input"
                value={settings.maxConcurrentAgents}
                onChange={(e) => setSettings({ ...settings, maxConcurrentAgents: parseInt(e.target.value) })}
              />
            </div>
          </div>
        </Card>

        {/* Physics Settings */}
        <Card title="Physics Engine">
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">GPU Device</label>
              <select
                className="input"
                value={settings.gpuDevice}
                onChange={(e) => setSettings({ ...settings, gpuDevice: e.target.value })}
              >
                <option value="0">GPU 0</option>
                <option value="1">GPU 1</option>
                <option value="auto">Auto</option>
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">Batch Size</label>
              <input
                type="number"
                className="input"
                value={settings.batchSize}
                onChange={(e) => setSettings({ ...settings, batchSize: parseInt(e.target.value) })}
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">Simulation Fidelity</label>
              <select className="input">
                <option value="fast">Fast</option>
                <option value="balanced">Balanced</option>
                <option value="accurate">Accurate</option>
              </select>
            </div>
          </div>
        </Card>

        {/* Exploration Settings */}
        <Card title="Exploration">
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">
                Exploration Temperature: {settings.explorationTemp}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                className="w-full"
                value={settings.explorationTemp}
                onChange={(e) => setSettings({ ...settings, explorationTemp: parseFloat(e.target.value) })}
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Conservative</span>
                <span>Aggressive</span>
              </div>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">
                Novelty Threshold: {settings.noveltyThreshold}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                className="w-full"
                value={settings.noveltyThreshold}
                onChange={(e) => setSettings({ ...settings, noveltyThreshold: parseFloat(e.target.value) })}
              />
            </div>
          </div>
        </Card>

        {/* Data Management */}
        <Card title="Data Management">
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">
                Auto-save Interval (seconds)
              </label>
              <input
                type="number"
                className="input"
                value={settings.autoSaveInterval}
                onChange={(e) => setSettings({ ...settings, autoSaveInterval: parseInt(e.target.value) })}
              />
            </div>

            <div className="space-y-2">
              <Button variant="secondary" className="w-full">
                Export Knowledge Base
              </Button>
              <Button variant="secondary" className="w-full">
                Import Knowledge Base
              </Button>
              <Button variant="ghost" className="w-full text-alert">
                Clear History (Dangerous)
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
