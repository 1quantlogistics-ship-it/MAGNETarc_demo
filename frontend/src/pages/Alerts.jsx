import { useSystemStore } from '../store/systemStore';
import Card from '../components/shared/Card';
import Badge from '../components/shared/Badge';
import { Bell, CheckCircle, AlertTriangle, Info, Trash2 } from 'lucide-react';

export default function Alerts() {
  const { notifications, markNotificationRead, clearNotifications } = useSystemStore();

  const mockNotifications = notifications.length > 0 ? notifications : [
    { id: '1', type: 'breakthrough', message: 'New design principle discovered: Hull spacing optimization', timestamp: new Date().toISOString(), read: false },
    { id: '2', type: 'system', message: 'GPU 1 utilization above 95% for 10 minutes', timestamp: new Date(Date.now() - 300000).toISOString(), read: false },
    { id: '3', type: 'agent', message: 'Explorer agent proposed counter-intuitive hypothesis', timestamp: new Date(Date.now() - 600000).toISOString(), read: true },
    { id: '4', type: 'failure', message: 'Design batch 47 failed stability requirements', timestamp: new Date(Date.now() - 900000).toISOString(), read: true },
  ];

  const getIcon = (type) => {
    switch (type) {
      case 'breakthrough':
        return <CheckCircle className="w-5 h-5 text-success" />;
      case 'failure':
        return <AlertTriangle className="w-5 h-5 text-alert" />;
      case 'system':
        return <Info className="w-5 h-5 text-warning" />;
      default:
        return <Bell className="w-5 h-5 text-cyan" />;
    }
  };

  const unreadCount = mockNotifications.filter(n => !n.read).length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-2">Alerts & Notifications</h1>
          <p className="text-gray-400">
            {unreadCount} unread notification{unreadCount !== 1 ? 's' : ''}
          </p>
        </div>
        <button
          onClick={clearNotifications}
          className="btn-ghost flex items-center gap-2 text-alert"
        >
          <Trash2 className="w-4 h-4" />
          Clear All
        </button>
      </div>

      {/* Unread Notifications */}
      {unreadCount > 0 && (
        <Card title="Unread Notifications">
          <div className="space-y-3">
            {mockNotifications.filter(n => !n.read).map((notif) => (
              <div
                key={notif.id}
                className="glass-card p-4 rounded-lg border-l-2 border-cyan cursor-pointer hover:bg-navy-700/30"
                onClick={() => markNotificationRead(notif.id)}
              >
                <div className="flex items-start gap-3">
                  {getIcon(notif.type)}
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant={
                        notif.type === 'breakthrough' ? 'success' :
                        notif.type === 'failure' ? 'alert' :
                        notif.type === 'system' ? 'warning' :
                        'info'
                      }>
                        {notif.type}
                      </Badge>
                      <span className="text-xs text-gray-400">
                        {new Date(notif.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-sm">{notif.message}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* All Notifications */}
      <Card title="All Notifications">
        <div className="space-y-2">
          {mockNotifications.map((notif) => (
            <div
              key={notif.id}
              className={`glass-card p-4 rounded-lg cursor-pointer transition-all ${
                notif.read ? 'opacity-50' : 'hover:bg-navy-700/30'
              }`}
              onClick={() => !notif.read && markNotificationRead(notif.id)}
            >
              <div className="flex items-start gap-3">
                {getIcon(notif.type)}
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant={
                      notif.type === 'breakthrough' ? 'success' :
                      notif.type === 'failure' ? 'alert' :
                      notif.type === 'system' ? 'warning' :
                      'info'
                    }>
                      {notif.type}
                    </Badge>
                    <span className="text-xs text-gray-400">
                      {new Date(notif.timestamp).toLocaleString()}
                    </span>
                    {!notif.read && (
                      <span className="ml-auto text-xs text-cyan">NEW</span>
                    )}
                  </div>
                  <p className="text-sm">{notif.message}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Notification Settings */}
      <Card title="Notification Preferences">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" defaultChecked className="w-4 h-4" />
              <span className="text-sm">Breakthrough Discoveries</span>
            </label>
          </div>
          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" defaultChecked className="w-4 h-4" />
              <span className="text-sm">System Alerts</span>
            </label>
          </div>
          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" defaultChecked className="w-4 h-4" />
              <span className="text-sm">Agent Actions</span>
            </label>
          </div>
          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" defaultChecked className="w-4 h-4" />
              <span className="text-sm">Failure Patterns</span>
            </label>
          </div>
        </div>
      </Card>
    </div>
  );
}
