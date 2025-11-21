import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useSystemStore } from '../../store/systemStore';
import {
  LayoutDashboard,
  Activity,
  TrendingUp,
  History,
  BookOpen,
  Users,
  Settings,
  Bell,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/research', icon: Activity, label: 'Live Research' },
  { path: '/frontier', icon: TrendingUp, label: 'Pareto Frontier' },
  { path: '/history', icon: History, label: 'Design History' },
  { path: '/knowledge', icon: BookOpen, label: 'Knowledge Base' },
  { path: '/agents', icon: Users, label: 'Agent Monitor' },
  { path: '/settings', icon: Settings, label: 'Settings' },
  { path: '/alerts', icon: Bell, label: 'Alerts' },
];

export default function Sidebar({ collapsed, setCollapsed }) {
  const { isOnline, currentCycle, resources, knowledgeBase } = useSystemStore();

  return (
    <motion.aside
      animate={{ width: collapsed ? 80 : 280 }}
      className="fixed left-0 top-0 h-screen glass-card border-r border-navy-700/50 z-50"
    >
      <div className="flex flex-col h-full">
        {/* Logo and collapse button */}
        <div className="p-6 border-b border-navy-700/50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-cyan to-cyan-dark rounded-lg flex items-center justify-center">
              <Activity className="w-6 h-6 text-navy-900" />
            </div>
            {!collapsed && (
              <div>
                <h1 className="text-xl font-bold gradient-text">MAGNET</h1>
                <p className="text-xs text-gray-400">Naval Research AI</p>
              </div>
            )}
          </div>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 hover:bg-navy-700/50 rounded-lg transition-colors"
          >
            {collapsed ? (
              <ChevronRight className="w-5 h-5" />
            ) : (
              <ChevronLeft className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* Status indicator */}
        <div className="px-6 py-4 border-b border-navy-700/50">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isOnline ? 'bg-success animate-pulse' : 'bg-gray-500'
              }`}
            />
            {!collapsed && (
              <span className="text-sm text-gray-400">
                {isOnline ? 'System Online' : 'Offline'}
              </span>
            )}
          </div>

          {!collapsed && (
            <div className="mt-3 space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">Cycle</span>
                <span className="font-mono text-cyan">{currentCycle}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">GPU 0</span>
                <span className="font-mono text-warning">{resources.gpu0.toFixed(0)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Principles</span>
                <span className="font-mono text-success">{knowledgeBase.discoveries}</span>
              </div>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto p-4">
          <div className="space-y-2">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                    isActive
                      ? 'bg-cyan text-navy-900 font-semibold'
                      : 'hover:bg-navy-700/50 text-gray-300'
                  }`
                }
                title={collapsed ? item.label : undefined}
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                {!collapsed && <span>{item.label}</span>}
              </NavLink>
            ))}
          </div>
        </nav>

        {/* Footer */}
        {!collapsed && (
          <div className="p-4 border-t border-navy-700/50 text-xs text-gray-400 text-center">
            <p>MAGNET v1.0.0</p>
            <p className="mt-1">Autonomous Naval Research</p>
          </div>
        )}
      </div>
    </motion.aside>
  );
}
