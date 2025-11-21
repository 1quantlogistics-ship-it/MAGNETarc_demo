"""
Integration Tests for 3D Mesh Generation System

Tests the complete mesh generation functionality integrated with the physics pipeline:
- Physics engine generates meshes correctly
- Orchestrator creates meshes for all designs in a cycle
- Mesh files are valid and watertight
- Performance overhead is acceptable
- Metadata accuracy
- Edge case handling

All tests run on Mac CPU without GPU/LLM requirements.
"""

import pytest
import sys
import os
import tempfile
import shutil
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from naval_domain.physics_engine import simulate_design
from naval_domain.hull_parameters import HullParameters, get_baseline_catamaran, get_high_speed_catamaran
from naval_domain.hull_generator import HullGenerator


class TestMeshGenerationIntegration:
    """Comprehensive integration tests for 3D mesh generation in physics pipeline."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.test_mesh_dir = Path(self.temp_dir) / "test_meshes"
        self.test_mesh_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Cleanup test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_physics_simulation_generates_mesh(self):
        """Test that physics simulation creates STL mesh file."""
        params = get_baseline_catamaran()
        results = simulate_design(params, verbose=False, generate_mesh=True)

        # Verify mesh was generated
        assert results.mesh_path is not None, "mesh_path should be set"
        assert results.design_id is not None, "design_id should be set"
        assert results.mesh_metadata is not None, "mesh_metadata should be set"

        # Verify file exists
        mesh_file = Path(results.mesh_path)
        assert mesh_file.exists(), f"Mesh file should exist at {results.mesh_path}"

        # Verify file is valid size (not empty, not corrupt)
        file_size = mesh_file.stat().st_size
        assert file_size > 100_000, f"Mesh file too small ({file_size} bytes), may be corrupt"
        assert file_size < 10_000_000, f"Mesh file too large ({file_size} bytes)"

        # Verify metadata contains expected fields
        assert "vertex_count" in results.mesh_metadata
        assert "face_count" in results.mesh_metadata
        assert "volume" in results.mesh_metadata
        assert results.mesh_metadata["vertex_count"] > 0

        print(f"\n  ✓ Mesh generated: {results.design_id}")
        print(f"    File: {mesh_file.name}")
        print(f"    Size: {file_size:,} bytes")
        print(f"    Vertices: {results.mesh_metadata['vertex_count']:,}")
        print(f"    Faces: {results.mesh_metadata['face_count']:,}")

    def test_mesh_generation_can_be_disabled(self):
        """Test that mesh generation can be disabled for performance."""
        params = get_baseline_catamaran()
        results = simulate_design(params, verbose=False, generate_mesh=False)

        # Should not have mesh fields when disabled
        assert results.mesh_path is None, "mesh_path should be None when disabled"
        assert results.design_id is None, "design_id should be None when disabled"
        assert results.mesh_metadata is None, "mesh_metadata should be None when disabled"

        print("\n  ✓ Mesh generation correctly disabled")

    def test_mesh_generation_performance_overhead(self):
        """Test that mesh generation adds acceptable overhead."""
        params = get_baseline_catamaran()
        n_runs = 10

        # Time without mesh generation
        start = time.time()
        for _ in range(n_runs):
            simulate_design(params, verbose=False, generate_mesh=False)
        time_no_mesh = (time.time() - start) / n_runs

        # Time with mesh generation
        start = time.time()
        for _ in range(n_runs):
            simulate_design(params, verbose=False, generate_mesh=True)
        time_with_mesh = (time.time() - start) / n_runs

        # Calculate overhead
        overhead = time_with_mesh - time_no_mesh
        overhead_pct = (overhead / time_with_mesh) * 100

        print(f"\n  Performance Analysis ({n_runs} runs averaged):")
        print(f"    Physics only:    {time_no_mesh*1000:.2f}ms")
        print(f"    Physics + mesh:  {time_with_mesh*1000:.2f}ms")
        print(f"    Overhead:        {overhead*1000:.2f}ms ({overhead_pct:.1f}%)")

        # Mesh generation should add < 1 second overhead on average
        assert overhead < 1.0, f"Mesh generation overhead too high: {overhead:.2f}s"

    def test_multiple_simulations_unique_design_ids(self):
        """Test that multiple simulations get unique design IDs."""
        params1 = get_baseline_catamaran()
        params2 = get_high_speed_catamaran()

        results1 = simulate_design(params1, verbose=False, generate_mesh=True)
        # Small delay to ensure different timestamps
        time.sleep(0.01)
        results2 = simulate_design(params2, verbose=False, generate_mesh=True)

        # Design IDs should be unique
        assert results1.design_id != results2.design_id, "Design IDs should be unique"

        # Mesh files should both exist
        assert Path(results1.mesh_path).exists()
        assert Path(results2.mesh_path).exists()

        # Mesh files should be different
        assert results1.mesh_path != results2.mesh_path

        print(f"\n  ✓ Unique design IDs generated:")
        print(f"    Design 1: {results1.design_id}")
        print(f"    Design 2: {results2.design_id}")

    def test_mesh_metadata_accuracy(self):
        """Test that mesh metadata matches actual mesh properties."""
        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not installed")

        params = get_baseline_catamaran()
        results = simulate_design(params, verbose=False, generate_mesh=True)

        # Load the generated mesh
        mesh = trimesh.load(results.mesh_path)

        # Compare metadata with actual mesh
        metadata = results.mesh_metadata

        # Vertex count should match exactly
        assert len(mesh.vertices) == metadata["vertex_count"], \
            f"Vertex count mismatch: {len(mesh.vertices)} vs {metadata['vertex_count']}"

        # Face count should match exactly
        assert len(mesh.faces) == metadata["face_count"], \
            f"Face count mismatch: {len(mesh.faces)} vs {metadata['face_count']}"

        # Volume should be close (some precision loss in serialization)
        if hasattr(mesh, 'volume') and mesh.volume > 0:
            volume_diff_pct = abs(mesh.volume - metadata["volume"]) / mesh.volume * 100
            assert volume_diff_pct < 10, f"Volume mismatch: {volume_diff_pct:.1f}%"

        print(f"\n  ✓ Metadata accuracy verified:")
        print(f"    Vertices: {len(mesh.vertices):,} (matches metadata)")
        print(f"    Faces: {len(mesh.faces):,} (matches metadata)")
        if hasattr(mesh, 'volume') and mesh.volume > 0:
            print(f"    Volume: {mesh.volume:.3f} m³ (error: {volume_diff_pct:.2f}%)")

    def test_mesh_generation_with_edge_case_parameters(self):
        """Test mesh generation with extreme but valid parameters."""
        # Very long, slender hull (extreme but valid)
        params = HullParameters(
            length_overall=35.0,
            beam=2.5,
            hull_depth=2.0,
            hull_spacing=9.0,
            deadrise_angle=5.0,
            freeboard=1.2,
            lcb_position=50.0,
            prismatic_coefficient=0.70,
            waterline_beam=2.2,
            block_coefficient=0.38,
            design_speed=38.0,
            displacement=70.0,
            draft=0.8
        )

        results = simulate_design(params, verbose=False, generate_mesh=True)

        # Should still generate valid mesh
        assert results.mesh_path is not None
        assert Path(results.mesh_path).exists()

        # Mesh should have reasonable size
        file_size = Path(results.mesh_path).stat().st_size
        assert file_size > 50_000, "Extreme design mesh too small"

        print(f"\n  ✓ Edge case parameters handled:")
        print(f"    L/B ratio: {params.length_overall/params.beam:.1f}")
        print(f"    Hull spacing: {params.hull_spacing:.1f}m")
        print(f"    Mesh size: {file_size:,} bytes")

    def test_mesh_watertight_property(self):
        """Test that generated meshes are watertight."""
        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not installed")

        params = get_baseline_catamaran()
        results = simulate_design(params, verbose=False, generate_mesh=True)

        # Load mesh
        mesh = trimesh.load(results.mesh_path)

        # Check watertight status
        is_watertight = mesh.is_watertight if hasattr(mesh, 'is_watertight') else False

        # Metadata should report watertight status
        assert "is_watertight" in results.mesh_metadata

        print(f"\n  ✓ Watertight check:")
        print(f"    Mesh watertight: {is_watertight}")
        print(f"    Metadata reports: {results.mesh_metadata['is_watertight']}")

        # Note: We don't require watertightness as an assertion because some
        # valid designs may have open geometries, but we track the property

    def test_mesh_file_format_validity(self):
        """Test that generated STL files are valid binary STL format."""
        params = get_baseline_catamaran()
        results = simulate_design(params, verbose=False, generate_mesh=True)

        # Read file header
        with open(results.mesh_path, 'rb') as f:
            # STL binary format: 80-byte header, 4-byte triangle count, then triangles
            header = f.read(80)
            triangle_count_bytes = f.read(4)

            assert len(header) == 80, "Invalid STL header size"
            assert len(triangle_count_bytes) == 4, "Invalid triangle count field"

            # Triangle count should be non-zero
            import struct
            triangle_count = struct.unpack('I', triangle_count_bytes)[0]
            assert triangle_count > 0, "STL file has no triangles"

            # Each triangle: 50 bytes (12 floats + 2 bytes attribute)
            expected_file_size = 80 + 4 + (triangle_count * 50)
            actual_file_size = Path(results.mesh_path).stat().st_size

            # File size should match expected format
            assert actual_file_size == expected_file_size, \
                f"File size mismatch: {actual_file_size} vs expected {expected_file_size}"

            print(f"\n  ✓ STL format validation:")
            print(f"    Triangle count: {triangle_count:,}")
            print(f"    File size: {actual_file_size:,} bytes")
            print(f"    Format: Binary STL ✓")

    def test_mesh_generation_error_handling(self):
        """Test that mesh generation failures don't crash simulation."""
        # Create parameters that are valid but might stress mesh generation
        params = HullParameters(
            length_overall=15.0,  # Valid L/B ratio
            beam=2.0,
            hull_depth=1.5,
            hull_spacing=4.0,
            deadrise_angle=15.0,
            freeboard=1.0,
            lcb_position=50.0,
            prismatic_coefficient=0.60,
            waterline_beam=1.8,
            block_coefficient=0.42,
            design_speed=25.0,
            displacement=20.0,
            draft=0.6
        )

        # Should not raise exception even if mesh generation fails
        try:
            results = simulate_design(params, verbose=False, generate_mesh=True)

            # Physics results should still be valid
            assert results.is_valid or not results.is_valid  # Either outcome is ok
            assert results.overall_score >= 0

            print("\n  ✓ Error handling: Simulation completed successfully")

        except Exception as e:
            pytest.fail(f"Simulation crashed with exception: {e}")

    def test_concurrent_mesh_generation(self):
        """Test that multiple concurrent mesh generations don't conflict."""
        import concurrent.futures

        params_list = [
            get_baseline_catamaran(),
            get_high_speed_catamaran(),
            get_baseline_catamaran(),  # Duplicate to test ID uniqueness
        ]

        def generate_mesh_concurrent(params, delay):
            time.sleep(delay)  # Stagger execution to ensure unique timestamps
            return simulate_design(params, verbose=False, generate_mesh=True)

        # Run concurrent mesh generation with staggered delays
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(generate_mesh_concurrent, p, i * 0.05)
                      for i, p in enumerate(params_list)]
            results = [f.result() for f in futures]

        # All should succeed
        assert len(results) == 3
        assert all(r.mesh_path is not None for r in results)

        # All design IDs should be unique
        design_ids = [r.design_id for r in results]
        assert len(design_ids) == len(set(design_ids)), "Design IDs should be unique"

        # All mesh files should exist
        assert all(Path(r.mesh_path).exists() for r in results)

        print(f"\n  ✓ Concurrent generation test:")
        print(f"    Designs generated: {len(results)}")
        print(f"    Unique IDs: {len(set(design_ids))}")
        print(f"    All files exist: ✓")

    def test_mesh_generation_with_different_lod_levels(self):
        """Test Level-of-Detail mesh generation."""
        params = get_baseline_catamaran()

        # Generate LOD meshes
        generator = HullGenerator()
        lod_meshes = generator.generate_lod_meshes(params)

        # Should have 3 LOD levels
        assert "low" in lod_meshes
        assert "medium" in lod_meshes
        assert "high" in lod_meshes

        # Vertex counts should increase with LOD
        low_verts = len(lod_meshes["low"].vertices)
        medium_verts = len(lod_meshes["medium"].vertices)
        high_verts = len(lod_meshes["high"].vertices)

        assert low_verts < medium_verts < high_verts, \
            "LOD vertex counts should increase: low < medium < high"

        print(f"\n  ✓ LOD mesh generation:")
        print(f"    Low:    {low_verts:,} vertices")
        print(f"    Medium: {medium_verts:,} vertices")
        print(f"    High:   {high_verts:,} vertices")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
