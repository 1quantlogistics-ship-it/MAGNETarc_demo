#!/usr/bin/env python3
"""
3D Hull Mesh Generation Demonstration

This script demonstrates the parametric 3D mesh generation capabilities
of the HullGenerator class. It generates sample catamaran meshes and
exports them to STL format.

Usage:
    python3 demo_mesh_generation.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from naval_domain.hull_generator import HullGenerator, generate_hull_mesh
from naval_domain.hull_parameters import get_baseline_catamaran, get_high_speed_catamaran
import trimesh


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def generate_and_save_mesh(hull_params, output_name, description):
    """
    Generate a mesh and save it to STL file.

    Args:
        hull_params: HullParameters object
        output_name: Base filename for output
        description: Description of the design
    """
    print(f"Generating {description}...")

    # Create output directory
    output_dir = Path("outputs/meshes/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate mesh
    generator = HullGenerator(num_cross_sections=50, points_per_section=30)
    mesh, metadata = generate_hull_mesh(hull_params)

    # Print mesh statistics
    print(f"  ✓ Mesh generated successfully!")
    print(f"    - Vertices: {metadata['vertex_count']:,}")
    print(f"    - Faces: {metadata['face_count']:,}")
    print(f"    - Volume: {metadata['volume']:.2f} m³")
    print(f"    - Surface Area: {metadata['surface_area']:.2f} m²")
    print(f"    - Bounds: {metadata['bounds']}")

    # Export to STL
    output_path = output_dir / f"{output_name}.stl"
    generator.export_stl(mesh, str(output_path))

    file_size_kb = output_path.stat().st_size / 1024
    print(f"  ✓ Exported to: {output_path}")
    print(f"    File size: {file_size_kb:.2f} KB")

    return mesh, metadata, output_path


def generate_lod_set(hull_params, output_name, description):
    """
    Generate LOD (Level of Detail) meshes.

    Args:
        hull_params: HullParameters object
        output_name: Base filename for outputs
        description: Description of the design
    """
    print(f"Generating LOD meshes for {description}...")

    # Create output directory
    output_dir = Path("outputs/meshes/demo/lod")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate LOD meshes
    generator = HullGenerator()
    lod_meshes = generator.generate_lod_meshes(hull_params)

    # Export each LOD level
    for lod_level, mesh in lod_meshes.items():
        output_path = output_dir / f"{output_name}_{lod_level}.stl"
        generator.export_stl(mesh, str(output_path))

        file_size_kb = output_path.stat().st_size / 1024
        vertex_count = len(mesh.vertices)
        face_count = len(mesh.faces)

        print(f"  ✓ {lod_level.upper()} LOD:")
        print(f"    - Vertices: {vertex_count:,}, Faces: {face_count:,}")
        print(f"    - File: {output_path} ({file_size_kb:.2f} KB)")


def demonstrate_morphing(params1, params2, name1, name2):
    """
    Demonstrate mesh morphing between two designs.

    Args:
        params1: First HullParameters
        params2: Second HullParameters
        name1: Name of first design
        name2: Name of second design
    """
    print(f"Demonstrating morphing between {name1} and {name2}...")

    # Generate both meshes with same topology
    generator = HullGenerator(num_cross_sections=30, points_per_section=20)
    mesh1, _ = generate_hull_mesh(params1)
    mesh2, _ = generate_hull_mesh(params2)

    # Verify topology matches
    assert len(mesh1.vertices) == len(mesh2.vertices), "Topology mismatch!"
    assert len(mesh1.faces) == len(mesh2.faces), "Topology mismatch!"

    print(f"  ✓ Topology verified: {len(mesh1.vertices):,} vertices")

    # Create output directory
    output_dir = Path("outputs/meshes/demo/morph")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate interpolated meshes (morphing sequence)
    morph_steps = 5
    for i in range(morph_steps + 1):
        t = i / morph_steps  # Interpolation parameter (0 to 1)

        # Interpolate vertex positions
        interpolated_vertices = (1 - t) * mesh1.vertices + t * mesh2.vertices

        # Create morphed mesh
        morphed_mesh = trimesh.Trimesh(
            vertices=interpolated_vertices,
            faces=mesh1.faces
        )

        # Export
        output_path = output_dir / f"morph_step_{i:02d}_t{int(t*100):03d}.stl"
        generator.export_stl(morphed_mesh, str(output_path))

        print(f"  ✓ Morph step {i}/{morph_steps} (t={t:.2f}): {output_path}")

    print(f"  ✓ Generated {morph_steps + 1} morphing frames")


def main():
    """Main demonstration function."""
    print_header("3D Hull Mesh Generation Demonstration")

    print("This demonstration will:")
    print("  1. Generate baseline catamaran mesh")
    print("  2. Generate high-speed catamaran mesh")
    print("  3. Create LOD (Level of Detail) versions")
    print("  4. Demonstrate mesh morphing animation")
    print()

    # Get baseline parameters
    baseline = get_baseline_catamaran()
    high_speed = get_high_speed_catamaran()

    # Demo 1: Baseline catamaran
    print_header("1. Baseline Catamaran")
    print(f"Design parameters:")
    print(f"  - Length: {baseline.length_overall}m")
    print(f"  - Beam: {baseline.beam}m")
    print(f"  - Hull Spacing: {baseline.hull_spacing}m")
    print(f"  - Design Speed: {baseline.design_speed} knots")
    print()

    baseline_mesh, baseline_metadata, baseline_path = generate_and_save_mesh(
        baseline,
        "baseline_catamaran",
        "Baseline Catamaran"
    )

    # Demo 2: High-speed catamaran
    print_header("2. High-Speed Catamaran")
    print(f"Design parameters:")
    print(f"  - Length: {high_speed.length_overall}m")
    print(f"  - Beam: {high_speed.beam}m")
    print(f"  - Hull Spacing: {high_speed.hull_spacing}m")
    print(f"  - Design Speed: {high_speed.design_speed} knots")
    print()

    highspeed_mesh, highspeed_metadata, highspeed_path = generate_and_save_mesh(
        high_speed,
        "highspeed_catamaran",
        "High-Speed Catamaran"
    )

    # Demo 3: LOD meshes
    print_header("3. Level of Detail (LOD) Meshes")
    generate_lod_set(baseline, "baseline", "Baseline Catamaran")
    print()
    generate_lod_set(high_speed, "highspeed", "High-Speed Catamaran")

    # Demo 4: Morphing demonstration
    print_header("4. Mesh Morphing Animation")
    demonstrate_morphing(
        baseline,
        high_speed,
        "Baseline",
        "High-Speed"
    )

    # Summary
    print_header("Summary")
    print("✓ Successfully generated all demonstration meshes!")
    print()
    print("Output locations:")
    print(f"  - Main meshes: outputs/meshes/demo/")
    print(f"  - LOD meshes: outputs/meshes/demo/lod/")
    print(f"  - Morph sequence: outputs/meshes/demo/morph/")
    print()
    print("You can view these STL files with:")
    print("  - MeshLab (free, cross-platform)")
    print("  - Blender (free, open source)")
    print("  - Online STL viewers")
    print("  - Your favorite 3D viewing software")
    print()
    print("Next steps:")
    print("  1. View the generated meshes in a 3D viewer")
    print("  2. Integrate mesh generation into physics pipeline (Task 3.2)")
    print("  3. Create FastAPI endpoints for mesh serving (Task 3.3)")
    print("  4. Build React + Three.js frontend (Task 3.4)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
