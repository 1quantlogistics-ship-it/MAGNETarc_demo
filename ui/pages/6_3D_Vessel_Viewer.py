"""
3D Vessel Viewer - MAGNET Naval Design Visualization (Task 3.4)
================================================================

Interactive 3D viewer for naval vessel designs using Three.js.
Loads STL meshes from the mesh serving API and displays them with:
- Orbit controls for camera interaction
- Color-coding by performance score
- Lighting and water plane for context
- Real-time mesh loading

Author: Agent 3 (3D Visualization Lead)
Task: 3.4 - React + Three.js Frontend
Date: 2025-11-20
"""

import streamlit as st
import requests
import streamlit.components.v1 as components
from typing import List, Dict, Any
import json

# ============================================================
# Configuration
# ============================================================

MESH_API_URL = "http://localhost:8000"

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

.viewer-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    margin-bottom: 24px;
}

.mesh-info-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 12px;
}

.gradient-text {
    background: linear-gradient(135deg, #0A84FF 0%, #30D158 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 36px;
    font-weight: 700;
    letter-spacing: -1px;
}

.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #0A84FF;
}

.metric-label {
    font-size: 14px;
    color: rgba(255, 255, 255, 0.6);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
"""

# ============================================================
# Three.js Viewer Component
# ============================================================

def create_threejs_viewer(mesh_url: str, design_id: str, metadata: Dict[str, Any] = None) -> str:
    """
    Create HTML/JavaScript for Three.js 3D viewer.

    Args:
        mesh_url: URL to STL mesh file
        design_id: Design identifier
        metadata: Optional mesh metadata for display

    Returns:
        HTML string with Three.js viewer
    """

    # Calculate color based on performance score (if available)
    color = "#0A84FF"  # Default blue
    if metadata and 'performance_score' in metadata:
        score = metadata['performance_score']
        # Green for high scores, red for low scores
        if score >= 0.8:
            color = "#30D158"  # Green
        elif score >= 0.6:
            color = "#FFD60A"  # Yellow
        else:
            color = "#FF453A"  # Red

    html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Vessel Viewer</title>
    <style>
        body {{
            margin: 0;
            overflow: hidden;
            background: #0A0A0A;
        }}
        #canvas-container {{
            width: 100%;
            height: 700px;
            position: relative;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            z-index: 100;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-family: 'Inter', sans-serif;
            font-size: 18px;
            z-index: 99;
        }}
    </style>
</head>
<body>
    <div id="canvas-container">
        <div id="loading">Loading 3D model...</div>
        <div id="info">
            <strong>{design_id}</strong><br>
            Use mouse to rotate, zoom, pan<br>
            Left click: Rotate | Right click: Pan | Scroll: Zoom
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/STLLoader.js"></script>

    <script>
        // Scene setup
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0A0A0A);
        scene.fog = new THREE.Fog(0x0A0A0A, 50, 200);

        // Camera
        const camera = new THREE.PerspectiveCamera(
            45,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        camera.position.set(30, 20, 30);

        // Renderer
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(renderer.domElement);

        // Orbit controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = 5;
        controls.maxDistance = 100;
        controls.maxPolarAngle = Math.PI / 2;  // Prevent going below water

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);

        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight1.position.set(20, 30, 20);
        directionalLight1.castShadow = true;
        directionalLight1.shadow.camera.left = -50;
        directionalLight1.shadow.camera.right = 50;
        directionalLight1.shadow.camera.top = 50;
        directionalLight1.shadow.camera.bottom = -50;
        scene.add(directionalLight1);

        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        directionalLight2.position.set(-20, 20, -20);
        scene.add(directionalLight2);

        // Rim light for better depth perception
        const rimLight = new THREE.DirectionalLight(0x0A84FF, 0.5);
        rimLight.position.set(0, 10, -30);
        scene.add(rimLight);

        // Water plane
        const waterGeometry = new THREE.PlaneGeometry(200, 200);
        const waterMaterial = new THREE.MeshPhongMaterial({{
            color: 0x006994,
            transparent: true,
            opacity: 0.6,
            shininess: 100,
            side: THREE.DoubleSide
        }});
        const water = new THREE.Mesh(waterGeometry, waterMaterial);
        water.rotation.x = -Math.PI / 2;
        water.position.y = 0;
        water.receiveShadow = true;
        scene.add(water);

        // Grid helper (below water)
        const gridHelper = new THREE.GridHelper(100, 50, 0x444444, 0x222222);
        gridHelper.position.y = -0.1;
        scene.add(gridHelper);

        // Load STL mesh
        const loader = new THREE.STLLoader();
        loader.load(
            '{mesh_url}',
            function(geometry) {{
                // Center the geometry
                geometry.center();

                // Compute vertex normals for smooth shading
                geometry.computeVertexNormals();

                // Create material with color-coding
                const material = new THREE.MeshPhongMaterial({{
                    color: '{color}',
                    specular: 0x111111,
                    shininess: 100,
                    flatShading: false
                }});

                const mesh = new THREE.Mesh(geometry, material);
                mesh.castShadow = true;
                mesh.receiveShadow = true;

                // Position mesh so waterline is at y=0
                // Assume hull draft is roughly 1/4 of height
                const bbox = new THREE.Box3().setFromObject(mesh);
                const height = bbox.max.y - bbox.min.y;
                mesh.position.y = height * 0.1;  // Slightly above water

                scene.add(mesh);

                // Hide loading message
                document.getElementById('loading').style.display = 'none';

                // Auto-frame the vessel
                const size = bbox.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = camera.fov * (Math.PI / 180);
                let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
                cameraZ *= 1.5;  // Add some breathing room

                camera.position.set(cameraZ * 0.8, cameraZ * 0.5, cameraZ * 0.8);
                camera.lookAt(mesh.position);
                controls.target.copy(mesh.position);
                controls.update();
            }},
            function(xhr) {{
                const percent = (xhr.loaded / xhr.total * 100).toFixed(0);
                document.getElementById('loading').textContent = `Loading 3D model... ${{percent}}%`;
            }},
            function(error) {{
                console.error('Error loading STL:', error);
                document.getElementById('loading').textContent = 'Error loading 3D model';
                document.getElementById('loading').style.color = '#FF453A';
            }}
        );

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();

            // Gentle water animation
            water.material.opacity = 0.6 + Math.sin(Date.now() * 0.001) * 0.05;

            renderer.render(scene, camera);
        }}
        animate();

        // Handle window resize
        window.addEventListener('resize', function() {{
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }});
    </script>
</body>
</html>
"""
    return html_code

# ============================================================
# API Functions
# ============================================================

def fetch_mesh_list(limit: int = 100, include_demo: bool = True) -> Dict[str, Any]:
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
        st.error(f"Failed to fetch metadata: {e}")
        return {}

def fetch_api_stats() -> Dict[str, Any]:
    """Fetch API statistics."""
    try:
        response = requests.get(f"{MESH_API_URL}/api/meshes/stats", timeout=5)
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
    st.markdown('<h1 class="gradient-text">3D Vessel Viewer</h1>', unsafe_allow_html=True)
    st.markdown("Interactive visualization of MAGNET naval vessel designs")
    st.markdown("---")

    # Fetch API stats
    stats = fetch_api_stats()

    # Display stats in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="mesh-info-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats.get("total_meshes", 0)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Meshes</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="mesh-info-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats.get("current_meshes", 0)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Current</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="mesh-info-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats.get("demo_meshes", 0)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Demo</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="mesh-info-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats.get("total_size_mb", 0):.1f} MB</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Size</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sidebar - Mesh selection
    st.sidebar.markdown("### Mesh Selection")

    # Options
    include_demo = st.sidebar.checkbox("Include demo meshes", value=True)

    # Fetch mesh list
    mesh_data = fetch_mesh_list(limit=100, include_demo=include_demo)
    meshes = mesh_data.get("meshes", [])

    if not meshes:
        st.warning("No meshes available. Please generate meshes first using the physics engine.")
        st.code("python3 demo_mesh_generation.py", language="bash")
        return

    # Mesh selector
    mesh_options = [f"{m['design_id']} ({m['file_size']/1024:.1f} KB)" for m in meshes]
    selected_idx = st.sidebar.selectbox(
        "Select mesh to view:",
        range(len(mesh_options)),
        format_func=lambda i: mesh_options[i]
    )

    selected_mesh = meshes[selected_idx]
    design_id = selected_mesh['design_id']

    # Display mesh info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Mesh Information")
    st.sidebar.markdown(f"**Design ID:** {design_id}")
    st.sidebar.markdown(f"**File Size:** {selected_mesh['file_size']/1024:.2f} KB")

    from datetime import datetime
    created = datetime.fromtimestamp(selected_mesh['created_at'])
    st.sidebar.markdown(f"**Created:** {created.strftime('%Y-%m-%d %H:%M')}")

    # Fetch metadata
    if st.sidebar.button("Refresh Metadata"):
        st.rerun()

    metadata = fetch_mesh_metadata(design_id)

    if metadata:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Geometry")
        st.sidebar.markdown(f"**Vertices:** {metadata.get('vertex_count', 'N/A'):,}")
        st.sidebar.markdown(f"**Faces:** {metadata.get('face_count', 'N/A'):,}")

        if 'volume' in metadata:
            st.sidebar.markdown(f"**Volume:** {metadata['volume']:.2f} m³")
        if 'surface_area' in metadata:
            st.sidebar.markdown(f"**Surface Area:** {metadata['surface_area']:.2f} m²")
        if 'is_watertight' in metadata:
            watertight_icon = "✅" if metadata['is_watertight'] else "⚠️"
            st.sidebar.markdown(f"**Watertight:** {watertight_icon}")

    # Main viewer
    st.markdown('<div class="viewer-container">', unsafe_allow_html=True)

    mesh_url = f"{MESH_API_URL}/api/mesh/{design_id}"
    viewer_html = create_threejs_viewer(mesh_url, design_id, metadata)

    components.html(viewer_html, height=700, scrolling=False)

    st.markdown('</div>', unsafe_allow_html=True)

    # Controls help
    st.markdown("### Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🖱️ **Left Click + Drag** - Rotate view")
    with col2:
        st.markdown("🖱️ **Right Click + Drag** - Pan camera")
    with col3:
        st.markdown("🖱️ **Scroll Wheel** - Zoom in/out")

if __name__ == "__main__":
    main()
