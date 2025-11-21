#!/usr/bin/env python3
"""
Demo: 3D Vessel Viewer (Task 3.4)

Demonstrates the 3D visualization capabilities:
1. Checks mesh API availability
2. Lists available meshes
3. Provides instructions for launching the viewer

Author: Agent 3 (3D Visualization Lead)
Date: 2025-11-20
"""

import requests
import subprocess
import sys
from pathlib import Path

# Configuration
MESH_API_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).parent


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def check_api_status():
    """Check if mesh API is running."""
    print_header("Step 1: Check Mesh API Status")

    try:
        response = requests.get(f"{MESH_API_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Status: {data['status']}")
            print(f"  Version: {data['version']}")
            print(f"  Endpoints: {len(data['endpoints'])}")
            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ API not running")
        print("\nTo start the API:")
        print("  cd /Users/bengibson/MAGNETarc_demo")
        print("  python3 api/mesh_api.py")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def list_available_meshes():
    """List available meshes."""
    print_header("Step 2: Available Meshes")

    try:
        response = requests.get(
            f"{MESH_API_URL}/api/meshes/list",
            params={"limit": 10, "include_demo": True},
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            meshes = data['meshes']

            print(f"Total meshes available: {data['total']}")
            print(f"Showing: {len(meshes)}")
            print()

            if meshes:
                print("Available designs:")
                for i, mesh in enumerate(meshes[:5], 1):
                    size_kb = mesh['file_size'] / 1024
                    print(f"  {i}. {mesh['design_id']} ({size_kb:.1f} KB)")

                return meshes
            else:
                print("⚠ No meshes available")
                print("\nGenerate some meshes first:")
                print("  python3 demo_mesh_generation.py")
                return []
        else:
            print(f"✗ Failed to fetch meshes: {response.status_code}")
            return []

    except Exception as e:
        print(f"✗ Error: {e}")
        return []


def display_mesh_metadata(design_id):
    """Display metadata for a mesh."""
    try:
        response = requests.get(
            f"{MESH_API_URL}/api/mesh/{design_id}/metadata",
            timeout=5
        )

        if response.status_code == 200:
            meta = response.json()

            print(f"\nSample mesh: {design_id}")
            print(f"  Vertices: {meta.get('vertex_count', 0):,}")
            print(f"  Faces: {meta.get('face_count', 0):,}")

            if 'volume' in meta:
                print(f"  Volume: {meta['volume']:.2f} m³")
            if 'surface_area' in meta:
                print(f"  Surface Area: {meta['surface_area']:.2f} m²")
            if 'is_watertight' in meta:
                watertight = "Yes" if meta['is_watertight'] else "No"
                print(f"  Watertight: {watertight}")

    except Exception as e:
        print(f"  ✗ Could not fetch metadata: {e}")


def get_api_stats():
    """Get overall API statistics."""
    print_header("Step 3: API Statistics")

    try:
        response = requests.get(f"{MESH_API_URL}/api/meshes/stats", timeout=5)

        if response.status_code == 200:
            stats = response.json()

            print(f"Total meshes: {stats['total_meshes']}")
            print(f"  Current: {stats['current_meshes']}")
            print(f"  Archived: {stats['archived_meshes']}")
            print(f"  Demo: {stats['demo_meshes']}")
            print(f"  Total size: {stats['total_size_mb']:.2f} MB")
            print(f"  Directory: {stats['mesh_directory']}")

            return True
        else:
            print(f"✗ Failed to fetch stats")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def launch_instructions():
    """Display instructions for launching the viewer."""
    print_header("Step 4: Launch 3D Viewer")

    print("The 3D Vessel Viewer is a Streamlit web application.")
    print()
    print("To launch the viewer:")
    print()
    print("  1. Open a terminal")
    print("  2. Navigate to the project directory:")
    print("     cd /Users/bengibson/MAGNETarc_demo")
    print()
    print("  3. Run the Streamlit app:")
    print("     streamlit run ui/mission_control.py")
    print()
    print("  4. Your browser will open automatically")
    print()
    print("  5. In the sidebar, navigate to:")
    print("     📊 6_3D_Vessel_Viewer")
    print()
    print("Features:")
    print("  • Interactive 3D mesh visualization using Three.js")
    print("  • Orbit controls (rotate, pan, zoom)")
    print("  • Color-coded by performance score")
    print("  • Realistic lighting and water plane")
    print("  • Real-time mesh metadata display")
    print("  • Select from all available meshes")
    print()


def main():
    """Run the demo."""
    print_header("3D Vessel Viewer Demo (Task 3.4)")

    print("This demo verifies that the 3D visualization system is ready:")
    print("  1. Mesh API connectivity")
    print("  2. Available meshes for visualization")
    print("  3. Metadata retrieval")
    print("  4. Instructions for launching the viewer")
    print()

    # Check API
    api_ok = check_api_status()
    if not api_ok:
        print("\n⚠ Please start the mesh API first, then run this demo again.")
        return 1

    # List meshes
    meshes = list_available_meshes()

    if meshes:
        # Show sample metadata
        display_mesh_metadata(meshes[0]['design_id'])

    # Get stats
    get_api_stats()

    # Launch instructions
    launch_instructions()

    # Summary
    print_header("Demo Complete")

    if meshes:
        print("✓ Mesh API is running")
        print(f"✓ {len(meshes)} meshes available for visualization")
        print("✓ 3D Viewer page created: ui/pages/6_3D_Vessel_Viewer.py")
        print()
        print("Ready to view 3D vessels!")
        print()
        print("Quick start:")
        print("  streamlit run ui/mission_control.py")
        return 0
    else:
        print("⚠ No meshes available yet")
        print()
        print("Generate meshes first:")
        print("  python3 demo_mesh_generation.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
