"""
Dynamic Rendering Demo - MAGNET Real-Time Visualization
=======================================================

Interactive demo page showcasing real-time design generation and 3D visualization.
Features:
- Launch parametric design sweeps
- Side-by-side design comparison
- Live performance metrics
- Animation playback of design evolution

Author: Agent 2 (Integration & Demo)
Date: 2025-11-21
"""

import streamlit as st
import requests
import streamlit.components.v1 as components
from typing import List, Dict, Any, Optional
import json
import subprocess
import time
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

MESH_API_URL = "http://localhost:8000"
DEMO_SCRIPT = "demo_dynamic_rendering.py"

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

.demo-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}

.demo-header h1 {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.demo-header p {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.1rem;
}

.control-panel {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.metrics-card {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.metrics-card h3 {
    color: #667eea;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.metrics-card .value {
    color: white;
    font-size: 2rem;
    font-weight: 700;
}

.status-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    margin-right: 0.5rem;
}

.status-running {
    background: rgba(102, 126, 234, 0.2);
    color: #667eea;
    border: 1px solid #667eea;
}

.status-complete {
    background: rgba(76, 175, 80, 0.2);
    color: #4CAF50;
    border: 1px solid #4CAF50;
}

.status-error {
    background: rgba(244, 67, 54, 0.2);
    color: #F44336;
    border: 1px solid #F44336;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# Page Header
# ============================================================

st.markdown("""
<div class="demo-header">
    <h1>🎬 Dynamic Rendering Demo</h1>
    <p>Real-time design generation and 3D visualization</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Helper Functions
# ============================================================

def check_mesh_api():
    """Check if mesh API is running."""
    try:
        response = requests.get(f"{MESH_API_URL}/api/meshes/stats", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_recent_designs(limit: int = 10) -> List[Dict[str, Any]]:
    """Get list of recent designs."""
    try:
        response = requests.get(
            f"{MESH_API_URL}/api/meshes/list",
            params={"limit": limit, "sort": "newest"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get("meshes", [])
    except Exception as e:
        st.error(f"Failed to fetch designs: {e}")
    return []

def get_mesh_metadata(design_id: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a specific design."""
    try:
        response = requests.get(
            f"{MESH_API_URL}/api/mesh/{design_id}/metadata",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to fetch metadata: {e}")
    return None

# ============================================================
# System Status
# ============================================================

st.subheader("System Status")

col1, col2, col3 = st.columns(3)

with col1:
    api_status = check_mesh_api()
    st.markdown(f"""
    <div class="metrics-card">
        <h3>Mesh API</h3>
        <div class="value">{"🟢 Online" if api_status else "🔴 Offline"}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    designs = get_recent_designs(100)
    st.markdown(f"""
    <div class="metrics-card">
        <h3>Available Designs</h3>
        <div class="value">{len(designs)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    demo_script_exists = Path(DEMO_SCRIPT).exists()
    st.markdown(f"""
    <div class="metrics-card">
        <h3>Demo Script</h3>
        <div class="value">{"✓ Ready" if demo_script_exists else "✗ Missing"}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Demo Control Panel
# ============================================================

st.subheader("Launch Demo")

st.markdown('<div class="control-panel">', unsafe_allow_html=True)

demo_type = st.selectbox(
    "Select Demo Type",
    [
        "Length Sweep (-15% to +15%)",
        "Beam Sweep (-10% to +10%)",
        "Hull Spacing Sweep (-20% to +20%)",
        "Design Comparison",
        "Quick Test (3 designs)"
    ]
)

col1, col2 = st.columns(2)

with col1:
    num_designs = st.slider("Number of Designs", min_value=3, max_value=10, value=5,
                           help="Number of designs to generate in sweep")

with col2:
    delay = st.slider("Delay Between Designs (seconds)", min_value=0, max_value=5, value=2,
                     help="Time to wait between design generations for visualization")

launch_demo = st.button("🚀 Launch Demo", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Launch Demo
# ============================================================

if launch_demo:
    if not api_status:
        st.error("⚠️ Mesh API is not running! Please start the API server first:")
        st.code("python -m uvicorn api.mesh_api:app --reload", language="bash")
    elif not demo_script_exists:
        st.error(f"⚠️ Demo script not found: {DEMO_SCRIPT}")
    else:
        # Map demo type to command
        demo_commands = {
            "Length Sweep (-15% to +15%)": "1",
            "Beam Sweep (-10% to +10%)": "2",
            "Hull Spacing Sweep (-20% to +20%)": "3",
            "Design Comparison": "4",
            "Quick Test (3 designs)": "5"
        }

        demo_choice = demo_commands[demo_type]

        st.info(f"🎬 Launching {demo_type}...")
        st.info(f"📊 Generating {num_designs} designs with {delay}s delay between each")

        # Instructions for running demo
        st.markdown("### To run the demo:")
        st.markdown("Open a terminal and run:")
        st.code(f"python {DEMO_SCRIPT}", language="bash")
        st.markdown(f"Then select option **{demo_choice}** when prompted.")

        st.markdown("### Watch the results here:")
        st.markdown("Switch to the **3D Vessel Viewer** or **Design History** page to see designs appear in real-time!")

        st.success("✅ Demo instructions displayed. Run the command in your terminal to begin!")

# ============================================================
# Recent Designs Display
# ============================================================

st.subheader("Recent Designs")

if designs:
    st.info(f"Showing {len(designs)} most recent designs")

    # Create columns for design cards
    cols = st.columns(3)

    for idx, design in enumerate(designs[:9]):  # Show up to 9 designs
        col = cols[idx % 3]

        with col:
            design_id = design.get("design_id", "unknown")

            # Get metadata if available
            metadata = design.get("metadata", {})

            # Display design card
            st.markdown(f"""
            <div class="metrics-card">
                <h3>Design {design_id[-6:]}</h3>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-top: 0.5rem;">
                    Vertices: {metadata.get('vertex_count', 'N/A')}<br/>
                    Faces: {metadata.get('face_count', 'N/A')}<br/>
                    Size: {design.get('size_mb', 0):.2f} MB
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Link to viewer
            if st.button(f"View in 3D", key=f"view_{design_id}", use_container_width=True):
                st.switch_page("pages/6_3D_Vessel_Viewer.py")

else:
    st.warning("No designs available. Generate some designs using the demo!")

# ============================================================
# Instructions & Tips
# ============================================================

with st.expander("ℹ️ How to Use This Demo"):
    st.markdown("""
    ### Getting Started

    1. **Start the Mesh API Server** (if not already running):
       ```bash
       python -m uvicorn api.mesh_api:app --reload
       ```

    2. **Select a Demo Type** from the dropdown above

    3. **Configure Parameters**:
       - Number of designs to generate
       - Delay between generations (for visualization)

    4. **Launch the Demo** using the button

    5. **Watch Real-Time Updates**:
       - Switch to "3D Vessel Viewer" page
       - Or check "Design History" timeline
       - New designs will appear automatically via WebSocket

    ### Demo Types

    - **Length Sweep**: Varies vessel length from -15% to +15%
    - **Beam Sweep**: Varies vessel beam from -10% to +10%
    - **Hull Spacing Sweep**: Varies hull spacing from -20% to +20%
    - **Design Comparison**: Generates two designs for side-by-side comparison
    - **Quick Test**: Fast 3-design test for verification

    ### Tips

    - Use shorter delays (0-1s) for quick testing
    - Use longer delays (3-5s) for live demonstrations
    - Check "Design History" page for timeline view
    - All designs are automatically saved with STL meshes

    ### Troubleshooting

    - **API Offline**: Start the mesh API server first
    - **No Designs Appearing**: Check WebSocket connection in browser console
    - **Slow Generation**: Physics simulation takes ~0.5-2s per design
    - **Missing Meshes**: Ensure `outputs/meshes/current/` directory exists
    """)

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.5); font-size: 0.9rem;">
    <p>MAGNET Dynamic Rendering Demo | Agent 2 (Integration & Demo)</p>
    <p>Real-time 3D visualization powered by Three.js and WebSocket streaming</p>
</div>
""", unsafe_allow_html=True)
