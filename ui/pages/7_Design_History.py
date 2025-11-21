"""
Design History Timeline - Real-Time Mesh Updates (Task 3.5)
============================================================

Displays a timeline of all vessel designs with real-time WebSocket updates.
Shows notifications when new designs are generated.

Features:
- WebSocket connection to mesh API
- Real-time notifications for new meshes
- Design history timeline
- Quick preview and navigation to 3D viewer

Author: Agent 3 (3D Visualization Lead)
Task: 3.5 - WebSocket Real-Time Updates
Date: 2025-11-20
"""

import streamlit as st
import requests
import time
from datetime import datetime
from typing import List, Dict, Any

# ============================================================
# Configuration
# ============================================================

MESH_API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/meshes"

# ============================================================
# Custom CSS
# ============================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.gradient-text {
    background: linear-gradient(135deg, #0A84FF 0%, #30D158 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 36px;
    font-weight: 700;
    letter-spacing: -1px;
}

.timeline-item {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 20px;
    border-left: 4px solid #0A84FF;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}

.timeline-item:hover {
    background: rgba(255, 255, 255, 0.08);
    transform: translateX(4px);
}

.timeline-item.new {
    border-left-color: #30D158;
    animation: pulse 2s ease-in-out;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

.design-id {
    font-size: 18px;
    font-weight: 600;
    color: #0A84FF;
    margin-bottom: 8px;
}

.design-meta {
    font-size: 14px;
    color: rgba(255, 255, 255, 0.6);
    margin-bottom: 4px;
}

.notification-badge {
    background: #30D158;
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-left: 8px;
}

.ws-status {
    padding: 8px 16px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 16px;
}

.ws-connected {
    background: rgba(48, 209, 88, 0.2);
    color: #30D158;
    border: 1px solid #30D158;
}

.ws-disconnected {
    background: rgba(255, 69, 58, 0.2);
    color: #FF453A;
    border: 1px solid #FF453A;
}
</style>
"""

# ============================================================
# WebSocket Status Component
# ============================================================

WEBSOCKET_HTML = """
<div id="ws-status-container">
    <div id="ws-status" class="ws-status ws-disconnected">
        ⚠️ WebSocket: Disconnected
    </div>
    <div id="notifications"></div>
</div>

<script>
let ws = null;
let reconnectInterval = null;
let newMeshCount = 0;

function connectWebSocket() {
    const wsUrl = 'ws://localhost:8000/ws/meshes';

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = function(event) {
            console.log('WebSocket connected');
            document.getElementById('ws-status').innerHTML = '✅ WebSocket: Connected (live updates)';
            document.getElementById('ws-status').className = 'ws-status ws-connected';

            // Clear reconnect interval if it exists
            if (reconnectInterval) {
                clearInterval(reconnectInterval);
                reconnectInterval = null;
            }

            // Send ping every 30 seconds to keep connection alive
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send('ping');
                }
            }, 30000);
        };

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            console.log('WebSocket message:', data);

            if (data.event === 'new_mesh') {
                newMeshCount++;
                showNotification(data);

                // Reload page after 2 seconds to show new mesh
                setTimeout(() => {
                    window.parent.location.reload();
                }, 2000);
            } else if (data.event === 'connection_ack') {
                console.log('Connection acknowledged:', data.message);
            }
        };

        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            document.getElementById('ws-status').innerHTML = '⚠️ WebSocket: Error';
            document.getElementById('ws-status').className = 'ws-status ws-disconnected';
        };

        ws.onclose = function(event) {
            console.log('WebSocket closed');
            document.getElementById('ws-status').innerHTML = '⚠️ WebSocket: Disconnected (trying to reconnect...)';
            document.getElementById('ws-status').className = 'ws-status ws-disconnected';

            // Try to reconnect every 5 seconds
            if (!reconnectInterval) {
                reconnectInterval = setInterval(connectWebSocket, 5000);
            }
        };
    } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        document.getElementById('ws-status').innerHTML = '⚠️ WebSocket: Failed to connect';
        document.getElementById('ws-status').className = 'ws-status ws-disconnected';
    }
}

function showNotification(data) {
    const notification = document.createElement('div');
    notification.className = 'timeline-item new';
    notification.innerHTML = `
        <div class="design-id">🆕 New Design: ${data.design_id}</div>
        <div class="design-meta">File Size: ${(data.file_size / 1024).toFixed(1)} KB</div>
        <div class="design-meta">Just generated!</div>
    `;

    const container = document.getElementById('notifications');
    container.insertBefore(notification, container.firstChild);

    // Remove notification after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Connect when page loads
connectWebSocket();
</script>
"""

# ============================================================
# API Functions
# ============================================================

def fetch_mesh_list(limit: int = 100, include_demo: bool = False) -> Dict[str, Any]:
    """Fetch list of available meshes from API."""
    try:
        response = requests.get(
            f"{MESH_API_URL}/api/meshes/list",
            params={"limit": limit, "include_demo": include_demo},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"meshes": [], "total": 0}
    except Exception as e:
        st.error(f"Failed to fetch mesh list: {e}")
        return {"meshes": [], "total": 0}

def fetch_mesh_metadata(design_id: str) -> Dict[str, Any]:
    """Fetch metadata for a specific mesh."""
    try:
        response = requests.get(
            f"{MESH_API_URL}/api/mesh/{design_id}/metadata",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        return {}

# ============================================================
# Main UI
# ============================================================

def main():
    """Main Streamlit app."""

    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="gradient-text">Design History</h1>', unsafe_allow_html=True)
    st.markdown("Real-time timeline of all generated vessel designs")
    st.markdown("---")

    # WebSocket status and notifications
    st.components.v1.html(WEBSOCKET_HTML, height=150, scrolling=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.markdown("### Timeline Options")
    show_demo = st.sidebar.checkbox("Include demo meshes", value=False)
    auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=False)
    limit = st.sidebar.slider("Max designs to show", 10, 100, 50)

    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(10)
        st.rerun()

    # Fetch mesh list
    mesh_data = fetch_mesh_list(limit=limit, include_demo=show_demo)
    meshes = mesh_data.get("meshes", [])

    st.markdown(f"### Timeline ({len(meshes)} designs)")

    if not meshes:
        st.warning("No designs in history yet. Generate some meshes to see them here!")
        st.code("python3 demo_mesh_generation.py", language="bash")
        return

    # Display timeline
    for i, mesh in enumerate(meshes):
        design_id = mesh['design_id']
        file_size_kb = mesh['file_size'] / 1024
        created_time = datetime.fromtimestamp(mesh['created_at'])

        # Calculate time ago
        time_diff = datetime.now() - created_time
        if time_diff.seconds < 60:
            time_ago = f"{time_diff.seconds}s ago"
        elif time_diff.seconds < 3600:
            time_ago = f"{time_diff.seconds // 60}m ago"
        elif time_diff.days == 0:
            time_ago = f"{time_diff.seconds // 3600}h ago"
        else:
            time_ago = f"{time_diff.days}d ago"

        # Highlight recent designs (< 1 minute old)
        is_new = time_diff.seconds < 60
        item_class = "timeline-item new" if is_new else "timeline-item"

        # Design item
        st.markdown(f'<div class="{item_class}">', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f'<div class="design-id">{design_id}', unsafe_allow_html=True)
            if is_new:
                st.markdown('<span class="notification-badge">NEW</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="design-meta">📦 {file_size_kb:.1f} KB</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="design-meta">🕐 {time_ago} ({created_time.strftime("%H:%M:%S")})</div>', unsafe_allow_html=True)

            # Try to fetch metadata
            metadata = fetch_mesh_metadata(design_id)
            if metadata:
                st.markdown(f'<div class="design-meta">🔺 {metadata.get("vertex_count", 0):,} vertices, {metadata.get("face_count", 0):,} faces</div>', unsafe_allow_html=True)

        with col2:
            if st.button(f"View 3D", key=f"view_{i}"):
                st.session_state['selected_mesh'] = design_id
                st.switch_page("pages/6_3D_Vessel_Viewer.py")

        st.markdown('</div>', unsafe_allow_html=True)

    # Summary stats
    st.markdown("---")
    st.markdown("### Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Designs", len(meshes))
    with col2:
        recent_count = sum(1 for m in meshes if (datetime.now() - datetime.fromtimestamp(m['created_at'])).seconds < 3600)
        st.metric("Recent (1h)", recent_count)
    with col3:
        total_size_mb = sum(m['file_size'] for m in meshes) / (1024 * 1024)
        st.metric("Total Size", f"{total_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
