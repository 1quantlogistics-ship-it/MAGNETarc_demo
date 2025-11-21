#!/usr/bin/env python3
"""
Test Physics Pipeline Mesh Integration (Task 3.2)

This script tests that the physics engine now automatically generates
3D meshes when simulating hull designs.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from naval_domain.physics_engine import simulate_design
from naval_domain.hull_parameters import get_baseline_catamaran, get_high_speed_catamaran


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def test_single_simulation():
    """Test single design simulation with mesh generation."""
    print_header("Test 1: Single Design Simulation with Mesh Generation")

    params = get_baseline_catamaran()
    print("Simulating baseline catamaran with mesh generation...")
    print()

    results = simulate_design(params, verbose=True, generate_mesh=True)

    print()
    print("=== RESULTS ===")
    print(f"Valid Design: {results.is_valid}")
    print(f"Overall Score: {results.overall_score:.2f}/100")
    print()
    print(f"Design ID: {results.design_id}")
    print(f"Mesh Path: {results.mesh_path}")

    if results.mesh_metadata:
        print()
        print("Mesh Metadata:")
        print(f"  - Vertices: {results.mesh_metadata['vertex_count']:,}")
        print(f"  - Faces: {results.mesh_metadata['face_count']:,}")
        print(f"  - Volume: {results.mesh_metadata['volume']:.2f} m³")
        print(f"  - Surface Area: {results.mesh_metadata['surface_area']:.2f} m²")

    # Verify mesh file exists
    if results.mesh_path:
        mesh_file = Path(results.mesh_path)
        if mesh_file.exists():
            file_size_kb = mesh_file.stat().st_size / 1024
            print(f"\n✓ Mesh file verified: {file_size_kb:.2f} KB")
        else:
            print(f"\n✗ ERROR: Mesh file not found at {results.mesh_path}")
            return False

    return True


def test_multiple_simulations():
    """Test multiple simulations to verify unique design IDs."""
    print_header("Test 2: Multiple Simulations (Unique Design IDs)")

    designs = [
        ("Baseline", get_baseline_catamaran()),
        ("High-Speed", get_high_speed_catamaran()),
    ]

    design_ids = []
    mesh_paths = []

    for name, params in designs:
        print(f"Simulating {name} catamaran...")
        results = simulate_design(params, verbose=False, generate_mesh=True)

        design_ids.append(results.design_id)
        mesh_paths.append(results.mesh_path)

        print(f"  ✓ Design ID: {results.design_id}")
        print(f"    Mesh: {results.mesh_path}")
        print(f"    Score: {results.overall_score:.2f}/100")
        print()

    # Verify all design IDs are unique
    if len(design_ids) == len(set(design_ids)):
        print("✓ All design IDs are unique")
    else:
        print("✗ ERROR: Duplicate design IDs detected!")
        return False

    # Verify all mesh files exist
    all_exist = True
    for mesh_path in mesh_paths:
        if not Path(mesh_path).exists():
            print(f"✗ ERROR: Mesh file missing: {mesh_path}")
            all_exist = False

    if all_exist:
        print("✓ All mesh files created successfully")

    return all_exist


def test_mesh_disabled():
    """Test simulation with mesh generation disabled."""
    print_header("Test 3: Simulation WITHOUT Mesh Generation")

    params = get_baseline_catamaran()
    print("Simulating with generate_mesh=False...")

    results = simulate_design(params, verbose=False, generate_mesh=False)

    print(f"Design ID: {results.design_id}")
    print(f"Mesh Path: {results.mesh_path}")
    print(f"Mesh Metadata: {results.mesh_metadata}")

    if results.design_id is None and results.mesh_path is None:
        print("\n✓ Mesh generation correctly disabled")
        return True
    else:
        print("\n✗ ERROR: Mesh data present when it shouldn't be")
        return False


def test_results_serialization():
    """Test that PhysicsResults with mesh data can be serialized."""
    print_header("Test 4: Results Serialization (to_dict)")

    params = get_baseline_catamaran()
    results = simulate_design(params, verbose=False, generate_mesh=True)

    try:
        results_dict = results.to_dict()

        print("✓ Results successfully serialized to dictionary")
        print()
        print("Dictionary keys:")
        for key in results_dict.keys():
            print(f"  - {key}")

        # Verify mesh fields are present
        assert 'design_id' in results_dict
        assert 'mesh_path' in results_dict
        assert 'mesh_metadata' in results_dict

        print()
        print("✓ All mesh fields present in serialized output")
        return True

    except Exception as e:
        print(f"✗ ERROR: Serialization failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print_header("Physics Pipeline Mesh Integration Tests (Task 3.2)")

    print("This test suite verifies that the physics engine now:")
    print("  1. Automatically generates 3D meshes during simulation")
    print("  2. Assigns unique design IDs")
    print("  3. Saves meshes to outputs/meshes/current/")
    print("  4. Includes mesh metadata in PhysicsResults")
    print()

    tests = [
        ("Single Simulation", test_single_simulation),
        ("Multiple Simulations", test_multiple_simulations),
        ("Mesh Disabled", test_mesh_disabled),
        ("Results Serialization", test_results_serialization),
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
        print("🎉 ALL TESTS PASSED - Task 3.2 Integration Complete!")
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
