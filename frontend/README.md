# MAGNET React Dashboard

Production-ready React dashboard for the MAGNET Autonomous Naval Research System.

## Features

✅ **Complete UI Implementation**
- Dashboard Overview with 3D vessel viewer
- Live Research monitoring with real-time updates
- Pareto Frontier visualization
- Design History with filterable table
- Knowledge Base with design principles
- Agent Monitor showing multi-agent activity
- System Settings configuration
- Alerts & Notifications center

✅ **Advanced Visualizations**
- 3D vessel rendering with React Three Fiber
- Performance gauges and sparklines
- Interactive scatter plots (Pareto frontier)
- Real-time data charts with Recharts

✅ **Real-time Features**
- WebSocket integration with auto-reconnect
- Live agent activity feed
- Real-time performance metrics
- System resource monitoring

✅ **Professional Design**
- Dark theme with naval aesthetic
- Glassmorphism effects
- Smooth animations with Framer Motion
- Responsive layout (mobile/tablet/desktop)
- Accessible keyboard navigation

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first styling
- **React Three Fiber** - 3D rendering
- **Recharts** - Data visualization
- **Framer Motion** - Animations
- **Zustand** - State management
- **React Router v6** - Routing
- **Lucide React** - Icons

## Installation

```bash
# Navigate to frontend directory
cd /Users/bengibson/MAGNETarc_demo/frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Development Server

The dev server runs on `http://localhost:3000` with:
- Hot module replacement (HMR)
- Proxy to backend API at `http://localhost:8000`
- WebSocket proxy for real-time updates

## Project Structure

```
src/
├── components/
│   ├── layout/
│   │   ├── Sidebar.jsx          # Collapsible navigation
│   │   ├── TopNav.jsx            # Top bar with controls
│   │   └── MainLayout.jsx        # Main layout wrapper
│   ├── 3d/
│   │   └── VesselViewer3D.jsx   # React Three Fiber 3D viewer
│   ├── charts/
│   │   ├── Sparkline.jsx         # Mini trend charts
│   │   └── PerformanceGauge.jsx  # Circular gauges
│   └── shared/
│       ├── Card.jsx              # Glass card component
│       ├── Badge.jsx             # Status badges
│       ├── Button.jsx            # Animated buttons
│       └── LoadingSpinner.jsx    # Loading states
├── pages/
│   ├── Dashboard.jsx             # Main overview
│   ├── LiveResearch.jsx          # Real-time research view
│   ├── ParetoFrontier.jsx        # Multi-objective analysis
│   ├── DesignHistory.jsx         # Design archive
│   ├── KnowledgeBase.jsx         # Discovered principles
│   ├── AgentMonitor.jsx          # Agent activity
│   ├── Settings.jsx              # Configuration
│   └── Alerts.jsx                # Notifications
├── context/
│   └── WebSocketContext.jsx     # WebSocket provider
├── store/
│   └── systemStore.js            # Zustand state
├── utils/
│   └── mockData.js               # Mock data generators
├── App.jsx                       # Router setup
├── main.jsx                      # Entry point
└── index.css                     # Global styles
```

## Environment Variables

Create a `.env` file:

```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws/meshes

# Feature Flags
VITE_ENABLE_3D_VIEWER=true
VITE_ENABLE_MOCK_DATA=false

# Performance
VITE_MAX_FPS=30
VITE_3D_QUALITY=high
```

## Backend Integration

The dashboard connects to your existing Python backend:

### REST API Endpoints
- `GET /api/meshes/list` - Fetch mesh list
- `GET /api/mesh/{design_id}` - Download STL file
- `GET /api/mesh/{design_id}/metadata` - Get mesh metadata
- `GET /api/meshes/stats` - System statistics

### WebSocket Connection
- `ws://localhost:8000/ws/meshes` - Real-time mesh updates

### WebSocket Events
The dashboard handles these event types:
- `agent_message` - Agent said something
- `agent_status` - Agent status changed
- `batch_results` - Simulation batch complete
- `hypothesis_generated` - New hypothesis created
- `new_mesh` - New design mesh available
- `new_principle` - Knowledge base updated
- `resource_update` - GPU/memory stats
- `system_alert` - Error or warning

## License

Part of the MAGNET Autonomous Naval Research System.
