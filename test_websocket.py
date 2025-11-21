#!/usr/bin/env python3
"""
Test WebSocket Real-Time Updates (Task 3.5)

Connects to the mesh API WebSocket endpoint and listens for new mesh events.
Also simulates mesh generation to trigger events.

Author: Agent 3 (3D Visualization Lead)
Date: 2025-11-20
"""

import asyncio
import websockets
import json
import sys
import time
from pathlib import Path

# Configuration
WS_URL = "ws://localhost:8000/ws/meshes"
MESH_DIR = Path("outputs/meshes/current")


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


async def websocket_client():
    """WebSocket client that listens for mesh updates."""
    print_header("WebSocket Client - Listening for New Meshes")

    print(f"Connecting to {WS_URL}...")

    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✓ Connected to WebSocket!")

            # Listen for messages
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)

                    event = data.get('event')

                    if event == 'connection_ack':
                        print(f"\n✓ {data['message']}")
                        print(f"  Active connections: {data['active_connections']}")
                        print("\nWaiting for new mesh events...")
                        print("(Generate a mesh in another terminal to see real-time updates)")
                        print()

                    elif event == 'new_mesh':
                        print("\n" + "🆕 " + "=" * 66)
                        print("NEW MESH DETECTED!")
                        print("=" * 70)
                        print(f"  Design ID: {data['design_id']}")
                        print(f"  File Size: {data['file_size'] / 1024:.2f} KB")
                        print(f"  Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['created_at']))}")
                        print(f"  Mesh URL: http://localhost:8000{data['url']}")
                        print(f"  Metadata URL: http://localhost:8000{data['metadata_url']}")
                        print("=" * 70 + "\n")

                    elif event == 'pong':
                        print("  [Heartbeat: pong received]")

                    else:
                        print(f"  Unknown event: {event}")
                        print(f"  Data: {data}")

                except websockets.exceptions.ConnectionClosed:
                    print("\n✗ Connection closed by server")
                    break
                except Exception as e:
                    print(f"\n✗ Error receiving message: {e}")
                    break

    except ConnectionRefusedError:
        print("✗ Connection refused!")
        print("\nMake sure the mesh API server is running:")
        print("  python3 api/mesh_api.py")
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    return 0


async def send_heartbeat(websocket):
    """Send periodic heartbeat to keep connection alive."""
    while True:
        try:
            await asyncio.sleep(30)
            await websocket.send("ping")
            print("  [Heartbeat: ping sent]")
        except Exception:
            break


def simulate_mesh_generation():
    """Simulate mesh generation by copying an existing mesh with a new name."""
    print_header("Simulate Mesh Generation")

    MESH_DIR.mkdir(parents=True, exist_ok=True)

    # Find an existing mesh to copy
    existing_meshes = list(MESH_DIR.glob("*.stl"))

    if not existing_meshes:
        print("✗ No existing meshes found to copy")
        print("\nGenerate some meshes first:")
        print("  python3 demo_mesh_generation.py")
        return

    source_mesh = existing_meshes[0]
    print(f"Using template: {source_mesh.name}")

    # Create new mesh with timestamp
    design_id = f"design_{int(time.time() * 1000)}"
    new_mesh = MESH_DIR / f"{design_id}.stl"

    # Copy mesh file
    import shutil
    shutil.copy(source_mesh, new_mesh)

    print(f"\n✓ Created new mesh: {design_id}")
    print(f"  File: {new_mesh}")
    print(f"  Size: {new_mesh.stat().st_size / 1024:.2f} KB")
    print()
    print("WebSocket clients should receive notification!")


def main():
    """Run WebSocket test."""
    print_header("WebSocket Real-Time Updates Test (Task 3.5)")

    print("This test demonstrates WebSocket functionality:")
    print("  1. Connect to WebSocket endpoint")
    print("  2. Listen for new mesh events")
    print("  3. Receive real-time notifications")
    print()

    # Check if API is running
    import requests
    try:
        response = requests.get("http://localhost:8000/", timeout=2)
        if response.status_code == 200:
            print("✓ Mesh API is running")
        else:
            print("⚠ Mesh API returned unexpected status")
    except requests.exceptions.ConnectionError:
        print("✗ Mesh API is not running!")
        print("\nStart the API first:")
        print("  python3 api/mesh_api.py")
        return 1

    # Show options
    print()
    print("Options:")
    print("  1. Listen for WebSocket events (recommended)")
    print("  2. Simulate mesh generation")
    print("  3. Both (listen in background, then simulate)")
    print()

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        # Run WebSocket client
        asyncio.run(websocket_client())
    elif choice == "2":
        # Simulate mesh generation
        simulate_mesh_generation()
    elif choice == "3":
        # Both
        print("\nStarting WebSocket listener in background...")
        print("Press Ctrl+C to stop\n")

        # Create task for WebSocket client
        async def run_both():
            # Start WebSocket client task
            client_task = asyncio.create_task(websocket_client())

            # Wait a moment for connection
            await asyncio.sleep(2)

            # Simulate mesh generation after 3 seconds
            print("\nSimulating mesh generation in 3 seconds...")
            await asyncio.sleep(3)

            # Run simulation in sync code
            simulate_mesh_generation()

            # Keep running to see the event
            await asyncio.sleep(5)

            client_task.cancel()

        try:
            asyncio.run(run_both())
        except KeyboardInterrupt:
            print("\n\nStopped by user")
    else:
        print("Invalid choice")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
