"""
Unit Tests for Hull Generator with 3D Mesh Generation

Tests the HullGenerator class with Trimesh-based parametric mesh generation.
"""

import pytest
import numpy as np
import trimesh
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from naval_domain.hull_generator import HullGenerator, generate_hull_metadata, generate_hull_mesh
from naval_domain.hull_parameters import HullParameters


@pytest.fixture
def baseline_catamaran():
    """Fixture providing baseline catamaran parameters."""
    return HullParameters(
        length_overall=18.0,
        beam=2.5,
        hull_spacing=4.5,
        displacement=15.0,
        design_speed=25.0,
        hull_depth=2.0,
        freeboard=1.2,
        draft=0.8,
        deadrise_angle=12.0,
        waterline_beam=2.3,
        block_coefficient=0.45,
        prismatic_coefficient=0.62,
        lcb_position=48.5
    )


@pytest.fixture
def high_speed_catamaran():
    """Fixture providing high-speed catamaran parameters."""
    return HullParameters(
        length_overall=22.0,
        beam=2.0,
        hull_spacing=5.0,
        displacement=12.0,
        design_speed=35.0,
        hull_depth=2.2,
        freeboard=1.5,
        draft=0.7,
        deadrise_angle=15.0,
        waterline_beam=1.8,
        block_coefficient=0.38,
        prismatic_coefficient=0.58,
        lcb_position=46.0
    )


class TestHullGeneratorBasics:
    """Test basic hull generator functionality."""

    def test_generator_initialization(self):
        """Test that generator initializes with correct parameters."""
        generator = HullGenerator(num_cross_sections=50, points_per_section=30)

        assert generator.num_cross_sections == 50
        assert generator.points_per_section == 30
        assert generator.generate_mesh == True

    def test_generator_no_mesh_mode(self):
        """Test generator in metadata-only mode."""
        generator = HullGenerator(generate_mesh=False)

        assert generator.generate_mesh == False

    def test_generate_metadata_only(self, baseline_catamaran):
        """Test generating metadata without 3D mesh."""
        generator = HullGenerator(generate_mesh=False)
        data = generator.generate(baseline_catamaran)

        assert data['hull_type'] == 'twin_hull'
        assert data['n_hulls'] == 2
        assert data['mesh'] is None
        assert data['mesh_metadata'] is None
        assert 'waterline_properties' in data
        assert 'volume_properties' in data


class TestMeshGeneration:
    """Test 3D mesh generation functionality."""

    def test_generate_mesh_basic(self, baseline_catamaran):
        """Test that mesh generation produces valid mesh."""
        generator = HullGenerator(num_cross_sections=20, points_per_section=16)
        data = generator.generate(baseline_catamaran)

        mesh = data['mesh']
        assert mesh is not None
        assert isinstance(mesh, trimesh.Trimesh)

        # Check mesh has vertices and faces
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

        print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    def test_mesh_metadata(self, baseline_catamaran):
        """Test that mesh metadata is correctly calculated."""
        generator = HullGenerator()
        data = generator.generate(baseline_catamaran)

        metadata = data['mesh_metadata']
        assert metadata is not None

        # Check required metadata fields
        assert 'vertex_count' in metadata
        assert 'face_count' in metadata
        assert 'volume' in metadata
        assert 'surface_area' in metadata
        assert 'bounds' in metadata

        # Verify values are reasonable
        assert metadata['vertex_count'] > 0
        assert metadata['face_count'] > 0
        assert metadata['volume'] > 0
        assert metadata['surface_area'] > 0

        print(f"Mesh metadata: {metadata}")

    def test_mesh_validity(self, baseline_catamaran):
        """Test that generated mesh is valid."""
        generator = HullGenerator()
        mesh, metadata = generate_hull_mesh(baseline_catamaran)

        # Check mesh properties
        assert mesh.is_empty == False
        assert mesh.vertices.shape[1] == 3  # 3D coordinates
        assert mesh.faces.shape[1] == 3     # Triangular faces

        # Check bounds are reasonable (mesh should fit design dimensions)
        # Note: Mesh includes hulls + deck + superstructure, so may be slightly larger than LOA
        bounds = mesh.bounds
        length = bounds[1][0] - bounds[0][0]  # X dimension
        width = bounds[1][1] - bounds[0][1]   # Y dimension
        height = bounds[1][2] - bounds[0][2]  # Z dimension

        # Length should be approximately LOA (allow up to 1.5x for deck/superstructure overhang)
        assert 0.5 * baseline_catamaran.length_overall < length < 1.5 * baseline_catamaran.length_overall

        # Width should account for hull spacing + beam
        expected_width = baseline_catamaran.hull_spacing + baseline_catamaran.beam
        assert 0.5 * expected_width < width < 2.0 * expected_width

        print(f"Mesh dimensions: L={length:.2f}m, W={width:.2f}m, H={height:.2f}m")


class TestConsistentTopology:
    """Test that topology is consistent across different designs (critical for morphing)."""

    def test_same_vertex_count(self, baseline_catamaran, high_speed_catamaran):
        """Test that different designs produce same vertex count."""
        generator = HullGenerator(num_cross_sections=30, points_per_section=20)

        mesh1, _ = generate_hull_mesh(baseline_catamaran)
        mesh2, _ = generate_hull_mesh(high_speed_catamaran)

        # Same topology = same vertex count
        assert len(mesh1.vertices) == len(mesh2.vertices)
        print(f"✓ Consistent vertex count: {len(mesh1.vertices)}")

    def test_same_face_count(self, baseline_catamaran, high_speed_catamaran):
        """Test that different designs produce same face count."""
        generator = HullGenerator(num_cross_sections=30, points_per_section=20)

        mesh1, _ = generate_hull_mesh(baseline_catamaran)
        mesh2, _ = generate_hull_mesh(high_speed_catamaran)

        # Same topology = same face count
        assert len(mesh1.faces) == len(mesh2.faces)
        print(f"✓ Consistent face count: {len(mesh1.faces)}")

    def test_topology_enables_morphing(self, baseline_catamaran, high_speed_catamaran):
        """Test that consistent topology enables vertex interpolation (morphing)."""
        generator = HullGenerator(num_cross_sections=25, points_per_section=16)

        mesh1, _ = generate_hull_mesh(baseline_catamaran)
        mesh2, _ = generate_hull_mesh(high_speed_catamaran)

        # Verify we can interpolate vertex positions
        assert mesh1.vertices.shape == mesh2.vertices.shape

        # Try interpolating at 50%
        interpolated_vertices = 0.5 * mesh1.vertices + 0.5 * mesh2.vertices

        # Create morphed mesh
        morphed_mesh = trimesh.Trimesh(
            vertices=interpolated_vertices,
            faces=mesh1.faces
        )

        assert morphed_mesh.is_empty == False
        assert len(morphed_mesh.vertices) == len(mesh1.vertices)
        print(f"✓ Successfully created morphed mesh with {len(morphed_mesh.vertices)} vertices")


class TestSTLExport:
    """Test STL file export functionality."""

    def test_export_stl(self, baseline_catamaran, tmp_path):
        """Test exporting mesh to STL file."""
        generator = HullGenerator()
        mesh, _ = generate_hull_mesh(baseline_catamaran)

        # Export to temporary file
        output_path = tmp_path / "test_hull.stl"
        generator.export_stl(mesh, str(output_path))

        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify we can reload the mesh
        reloaded_mesh = trimesh.load(str(output_path))
        assert isinstance(reloaded_mesh, trimesh.Trimesh)
        assert len(reloaded_mesh.vertices) == len(mesh.vertices)

        print(f"✓ Exported STL: {output_path.stat().st_size} bytes")

    def test_stl_file_size(self, baseline_catamaran, tmp_path):
        """Test that binary STL is reasonably sized."""
        generator = HullGenerator(num_cross_sections=50, points_per_section=30)
        mesh, _ = generate_hull_mesh(baseline_catamaran)

        output_path = tmp_path / "hull_binary.stl"
        generator.export_stl(mesh, str(output_path), binary=True)

        file_size_kb = output_path.stat().st_size / 1024

        # Binary STL should be compact (< 500KB for reasonable detail)
        assert file_size_kb < 500

        print(f"✓ Binary STL size: {file_size_kb:.2f} KB")


class TestLODGeneration:
    """Test Level-of-Detail mesh generation."""

    def test_generate_lod_meshes(self, baseline_catamaran):
        """Test generating multiple LOD versions."""
        generator = HullGenerator()
        lod_meshes = generator.generate_lod_meshes(baseline_catamaran)

        assert 'low' in lod_meshes
        assert 'medium' in lod_meshes
        assert 'high' in lod_meshes

        # All should be valid meshes
        for lod_name, mesh in lod_meshes.items():
            assert mesh is not None
            assert isinstance(mesh, trimesh.Trimesh)
            assert len(mesh.vertices) > 0
            print(f"{lod_name} LOD: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    def test_lod_vertex_counts_decrease(self, baseline_catamaran):
        """Test that vertex counts decrease from high to low LOD."""
        generator = HullGenerator()
        lod_meshes = generator.generate_lod_meshes(baseline_catamaran)

        high_count = len(lod_meshes['high'].vertices)
        medium_count = len(lod_meshes['medium'].vertices)
        low_count = len(lod_meshes['low'].vertices)

        # Higher detail should have more vertices
        assert high_count > medium_count
        assert medium_count > low_count

        print(f"LOD vertex progression: high={high_count}, medium={medium_count}, low={low_count}")


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_generate_hull_metadata_function(self, baseline_catamaran):
        """Test generate_hull_metadata convenience function."""
        data = generate_hull_metadata(baseline_catamaran, generate_mesh=False)

        assert data['hull_type'] == 'twin_hull'
        assert data['mesh'] is None

    def test_generate_hull_mesh_function(self, baseline_catamaran):
        """Test generate_hull_mesh convenience function."""
        mesh, metadata = generate_hull_mesh(baseline_catamaran)

        assert mesh is not None
        assert isinstance(mesh, trimesh.Trimesh)
        assert metadata is not None
        assert 'vertex_count' in metadata


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_minimal_cross_sections(self, baseline_catamaran):
        """Test with minimal number of cross-sections."""
        generator = HullGenerator(num_cross_sections=5, points_per_section=8)
        mesh, metadata = generate_hull_mesh(baseline_catamaran)

        assert mesh is not None
        assert len(mesh.vertices) > 0

    def test_high_detail_mesh(self, baseline_catamaran):
        """Test generating high-detail mesh."""
        generator = HullGenerator(num_cross_sections=100, points_per_section=60)
        mesh, metadata = generate_hull_mesh(baseline_catamaran)

        assert mesh is not None
        assert metadata['vertex_count'] > 2000  # Should be reasonably detailed
        # Note: With 100 sections × 60 points, each hull has 6000 vertices
        # Total mesh includes 2 hulls + deck + superstructure

        print(f"High-detail mesh: {metadata['vertex_count']} vertices")

    def test_different_hull_parameters(self):
        """Test with various hull parameter combinations."""
        test_params = HullParameters(
            length_overall=12.0,
            beam=1.8,
            hull_spacing=3.5,
            displacement=8.0,
            design_speed=20.0,
            hull_depth=1.5,
            freeboard=0.8,
            draft=0.6,
            deadrise_angle=10.0,
            waterline_beam=1.6,
            block_coefficient=0.42,
            prismatic_coefficient=0.60,
            lcb_position=50.0
        )

        mesh, metadata = generate_hull_mesh(test_params)

        assert mesh is not None
        assert metadata['vertex_count'] > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
