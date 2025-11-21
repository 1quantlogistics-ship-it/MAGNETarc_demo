import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { WebSocketProvider } from './context/WebSocketContext';
import MainLayout from './components/layout/MainLayout';
import Dashboard from './pages/Dashboard';
import LiveResearch from './pages/LiveResearch';
import ParetoFrontier from './pages/ParetoFrontier';
import DesignHistory from './pages/DesignHistory';
import KnowledgeBase from './pages/KnowledgeBase';
import AgentMonitor from './pages/AgentMonitor';
import Settings from './pages/Settings';
import Alerts from './pages/Alerts';

function App() {
  return (
    <WebSocketProvider>
      <Router>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="research" element={<LiveResearch />} />
            <Route path="frontier" element={<ParetoFrontier />} />
            <Route path="history" element={<DesignHistory />} />
            <Route path="knowledge" element={<KnowledgeBase />} />
            <Route path="agents" element={<AgentMonitor />} />
            <Route path="settings" element={<Settings />} />
            <Route path="alerts" element={<Alerts />} />
          </Route>
        </Routes>
      </Router>
    </WebSocketProvider>
  );
}

export default App;
