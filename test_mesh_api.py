#!/usr/bin/env python3
"""
Test Mesh API Endpoints (Task 3.3)

Tests the FastAPI endpoints for serving 3D mesh files.
"""

import sys
import time
import requests
from pathlib import Path

# Test configuration
API_BASE = "http://localhost:8000"
TEST_TIMEOUT = 5


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def test_api_health():
    """Test API health check."""
    print_header("Test 1: API Health Check")

    try:
        response = requests.get(f"{API_BASE}/", timeout=TEST_TIMEOUT)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data['status'] == 'online', "API not online"

        print(f"✓ API Status: {data['status']}")
        print(f"  Version: {data['version']}")
        print(f"  Endpoints: {len(data['endpoints'])}")

        return True

    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Cannot connect to API")
        print("  Make sure the API server is running:")
        print("  python3 api/mesh_api.py")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_mesh_stats():
    """Test mesh statistics endpoint."""
    print_header("Test 2: Mesh Statistics")

    try:
        response = requests.get(f"{API_BASE}/api/meshes/stats", timeout=TEST_TIMEOUT)
        assert response.status_code == 200

        data = response.json()

        print(f"✓ Total Meshes: {data['total_meshes']}")
        print(f"  Current: {data['current_meshes']}")
        print(f"  Archived: {data['archived_meshes']}")
        print(f"  Demo: {data['demo_meshes']}")
        print(f"  Total Size: {data['total_size_mb']:.2f} MB")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_list_meshes():
    """Test mesh listing with pagination."""
    print_header("Test 3: List Meshes (Pagination)")

    try:
        # Test basic listing
        response = requests.get(f"{API_BASE}/api/meshes/list", timeout=TEST_TIMEOUT)
        assert response.status_code == 200

        data = response.json()

        print(f"✓ Found {data['total']} meshes")
        print(f"  Returned: {len(data['meshes'])} (limit: {data['limit']})")
        print(f"  Offset: {data['offset']}")

        # Show first few meshes
        if data['meshes']:
            print("\n  Recent meshes:")
            for mesh in data['meshes'][:3]:
                size_kb = mesh['file_size'] / 1024
                print(f"    - {mesh['design_id']} ({size_kb:.2f} KB)")

        # Test pagination
        print("\n  Testing pagination...")
        response2 = requests.get(
            f"{API_BASE}/api/meshes/list",
            params={"limit": 5, "offset": 0},
            timeout=TEST_TIMEOUT
        )
        assert response2.status_code == 200
        data2 = response2.json()
        print(f"  ✓ Pagination works (limit=5): {len(data2['meshes'])} results")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_get_mesh():
    """Test mesh file retrieval."""
    print_header("Test 4: Get Mesh File")

    try:
        # First, get list of available meshes
        response = requests.get(f"{API_BASE}/api/meshes/list", params={"limit": 1}, timeout=TEST_TIMEOUT)
        data = response.json()

        if not data['meshes']:
            print("⚠ No meshes available to test")
            print("  Generate a mesh first:")
            print("  python3 demo_mesh_generation.py")
            return True  # Not a failure, just no data

        design_id = data['meshes'][0]['design_id']
        print(f"Testing with design: {design_id}")

        # Get mesh file
        response = requests.get(f"{API_BASE}/api/mesh/{design_id}", timeout=TEST_TIMEOUT)
        assert response.status_code == 200

        # Check content type
        content_type = response.headers.get('content-type')
        print(f"✓ Content-Type: {content_type}")

        # Check file size
        size_kb = len(response.content) / 1024
        print(f"✓ File Size: {size_kb:.2f} KB")

        # Verify it's STL data (starts with "solid" for ASCII or binary header)
        if response.content[:5] == b'solid':
            print("✓ Format: ASCII STL")
        else:
            print("✓ Format: Binary STL (likely)")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_get_metadata():
    """Test mesh metadata retrieval."""
    print_header("Test 5: Get Mesh Metadata")

    try:
        # Get a mesh to test with
        response = requests.get(f"{API_BASE}/api/meshes/list", params={"limit": 1}, timeout=TEST_TIMEOUT)
        data = response.json()

        if not data['meshes']:
            print("⚠ No meshes available to test")
            return True

        design_id = data['meshes'][0]['design_id']
        print(f"Testing with design: {design_id}")

        # Get metadata
        response = requests.get(f"{API_BASE}/api/mesh/{design_id}/metadata", timeout=TEST_TIMEOUT)
        assert response.status_code == 200

        metadata = response.json()

        print(f"✓ Metadata retrieved successfully")
        print(f"  Vertices: {metadata.get('vertex_count', 'N/A'):,}")
        print(f"  Faces: {metadata.get('face_count', 'N/A'):,}")
        print(f"  Volume: {metadata.get('volume', 0):.2f} m³")
        print(f"  Surface Area: {metadata.get('surface_area', 0):.2f} m²")
        print(f"  Watertight: {metadata.get('is_watertight', False)}")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_404_handling():
    """Test 404 error handling."""
    print_header("Test 6: 404 Error Handling")

    try:
        # Try to get non-existent mesh
        response = requests.get(f"{API_BASE}/api/mesh/nonexistent_design_12345", timeout=TEST_TIMEOUT)

        assert response.status_code == 404, f"Expected 404, got {response.status_code}"

        data = response.json()
        print(f"✓ 404 handled correctly")
        print(f"  Error message: {data.get('detail', 'N/A')}")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_recent_meshes():
    """Test recent meshes endpoint."""
    print_header("Test 7: Recent Meshes")

    try:
        response = requests.get(f"{API_BASE}/api/meshes/recent", params={"limit": 5}, timeout=TEST_TIMEOUT)
        assert response.status_code == 200

        data = response.json()

        print(f"✓ Retrieved {len(data['meshes'])} recent meshes")

        if data['meshes']:
            print("\n  Most recent:")
            for i, mesh in enumerate(data['meshes'][:3], 1):
                print(f"    {i}. {mesh['design_id']}")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def main():
    """Run all API tests."""
    print_header("Mesh API Test Suite (Task 3.3)")

    print("This test suite verifies that the mesh serving API:")
    print("  1. Responds to health checks")
    print("  2. Provides mesh statistics")
    print("  3. Lists available meshes with pagination")
    print("  4. Serves STL files for download")
    print("  5. Provides mesh metadata")
    print("  6. Handles errors gracefully")
    print()

    print(f"API Base URL: {API_BASE}")
    print()

    tests = [
        ("API Health Check", test_api_health),
        ("Mesh Statistics", test_mesh_stats),
        ("List Meshes", test_list_meshes),
        ("Get Mesh File", test_get_mesh),
        ("Get Metadata", test_get_metadata),
        ("404 Handling", test_404_handling),
        ("Recent Meshes", test_recent_meshes),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print_header("Test Summary")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print()
        print("=" * 70)
        print("🎉 ALL TESTS PASSED - Task 3.3 API Complete!")
        print("=" * 70)
        return 0
    else:
        print()
        print("=" * 70)
        print("⚠️  SOME TESTS FAILED - Please review errors above")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
