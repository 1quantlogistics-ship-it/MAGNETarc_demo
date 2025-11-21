import { useState, useEffect } from 'react';
import { useSystemStore } from '../../store/systemStore';
import { useWebSocket } from '../../context/WebSocketContext';
import { Play, Pause, Wifi, WifiOff, Bell, User } from 'lucide-react';
import Badge from '../shared/Badge';

export default function TopNav() {
  const { systemMode, setSystemMode, notifications } = useSystemStore();
  const { isConnected } = useWebSocket();
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const toggleSystemMode = () => {
    setSystemMode(systemMode === 'Autonomous' ? 'Paused' : 'Autonomous');
  };

  const unreadCount = notifications.filter((n) => !n.read).length;

  return (
    <nav className="h-16 glass-card border-b border-navy-700/50 px-6 flex items-center justify-between">
      {/* Left - System controls */}
      <div className="flex items-center gap-4">
        {/* System mode toggle */}
        <div className="flex items-center gap-3">
          <div
            className={`px-4 py-2 rounded-lg border-2 ${
              systemMode === 'Autonomous'
                ? 'border-success bg-success/10 text-success'
                : 'border-warning bg-warning/10 text-warning'
            }`}
          >
            <span className="font-semibold text-sm uppercase tracking-wider">
              {systemMode}
            </span>
          </div>
          <button
            onClick={toggleSystemMode}
            className="p-2 hover:bg-navy-700/50 rounded-lg transition-colors"
            title={systemMode === 'Autonomous' ? 'Pause System' : 'Resume System'}
          >
            {systemMode === 'Autonomous' ? (
              <Pause className="w-5 h-5 text-warning" />
            ) : (
              <Play className="w-5 h-5 text-success" />
            )}
          </button>
        </div>

        {/* Connection status */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-navy-700/30">
          {isConnected ? (
            <>
              <Wifi className="w-4 h-4 text-success" />
              <span className="text-xs text-success">Connected</span>
            </>
          ) : (
            <>
              <WifiOff className="w-4 h-4 text-alert" />
              <span className="text-xs text-alert">Disconnected</span>
            </>
          )}
        </div>
      </div>

      {/* Right - Status and actions */}
      <div className="flex items-center gap-6">
        {/* Current time */}
        <div className="text-sm font-mono text-gray-400">
          {currentTime.toLocaleTimeString()}
        </div>

        {/* Notifications */}
        <button className="relative p-2 hover:bg-navy-700/50 rounded-lg transition-colors">
          <Bell className="w-5 h-5" />
          {unreadCount > 0 && (
            <span className="absolute -top-1 -right-1 w-5 h-5 bg-alert rounded-full text-xs flex items-center justify-center font-bold">
              {unreadCount > 9 ? '9+' : unreadCount}
            </span>
          )}
        </button>

        {/* User profile */}
        <button className="flex items-center gap-2 px-3 py-2 hover:bg-navy-700/50 rounded-lg transition-colors">
          <div className="w-8 h-8 bg-gradient-to-br from-cyan to-cyan-dark rounded-full flex items-center justify-center">
            <User className="w-5 h-5 text-navy-900" />
          </div>
          <span className="text-sm">Operator</span>
        </button>
      </div>
    </nav>
  );
}
