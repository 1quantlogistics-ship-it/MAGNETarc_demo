"""
Hull Generator for Twin-Hull (Catamaran) Vessels

This module generates 3D hull meshes and geometry metadata from HullParameters.
Uses Trimesh library for parametric mesh generation with consistent topology.

Features:
- Parametric 3D mesh generation (vertices, faces)
- Binary STL export
- Level-of-Detail (LOD) mesh generation
- Consistent topology for morphing animations
- Hull geometry metadata calculation
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
import trimesh

from naval_domain.hull_parameters import HullParameters


class HullGenerator:
    """
    Generates 3D meshes and geometry metadata for catamaran hulls.

    Uses parametric generation to create consistent topology across
    different designs, enabling smooth morphing animations.
    """

    def __init__(self,
                 num_cross_sections: int = 50,
                 points_per_section: int = 30,
                 generate_mesh: bool = True):
        """
        Initialize hull generator with topology parameters.

        Args:
            num_cross_sections: Number of hull slices (longitudinal)
            points_per_section: Points around each cross-section
            generate_mesh: If True, generate 3D meshes; if False, only metadata

        Note: num_cross_sections and points_per_section must be consistent
              across ALL meshes for morphing to work!
        """
        self.num_cross_sections = num_cross_sections
        self.points_per_section = points_per_section
        self.generate_mesh = generate_mesh

    def generate(self, hull_params: HullParameters) -> Dict[str, Any]:
        """
        Generate hull geometry metadata and optionally 3D mesh.

        Args:
            hull_params: Hull parameters defining the design

        Returns:
            Dictionary containing hull geometry metadata:
            {
                'hull_type': 'twin_hull',
                'n_hulls': 2,
                'mesh': trimesh.Trimesh or None,
                'mesh_metadata': {...} or None,
                'waterline_properties': {...},
                'volume_properties': {...},
                'geometric_properties': {...},
                'validation': {...}
            }

        Raises:
            ValueError: If hull parameters are invalid
        """
        # Validate parameters
        hull_params.validate()

        # Generate 3D mesh if requested
        mesh = None
        mesh_metadata = None
        if self.generate_mesh:
            mesh, mesh_metadata = self._generate_3d_mesh(hull_params)

        # Generate metadata
        hull_data = {
            'hull_type': 'twin_hull',
            'n_hulls': 2,
            'mesh': mesh,
            'mesh_metadata': mesh_metadata,
            'waterline_properties': self._calculate_waterline_properties(hull_params),
            'volume_properties': self._calculate_volume_properties(hull_params),
            'geometric_properties': self._calculate_geometric_properties(hull_params),
            'station_properties': self._calculate_station_properties(hull_params),
            'validation': self._validate_geometry(hull_params),
        }

        return hull_data

    # ========================================================================
    # 3D MESH GENERATION METHODS
    # ========================================================================

    def _generate_3d_mesh(self, hull_params: HullParameters) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        """
        Generate complete 3D mesh from hull parameters.

        Returns:
            (mesh, metadata) tuple where mesh is a trimesh.Trimesh object
        """
        # Generate individual components
        port_hull = self._generate_single_hull(hull_params, offset=-hull_params.hull_spacing/2)
        starboard_hull = self._generate_single_hull(hull_params, offset=hull_params.hull_spacing/2)
        deck = self._generate_deck_platform(hull_params)
        superstructure = self._generate_superstructure(hull_params)

        # Combine all parts
        combined = trimesh.util.concatenate([
            port_hull, starboard_hull, deck, superstructure
        ])

        # Center mesh at origin
        combined.vertices -= combined.center_mass

        # Calculate metadata
        metadata = {
            "vertex_count": len(combined.vertices),
            "face_count": len(combined.faces),
            "volume": float(combined.volume) if hasattr(combined, 'volume') else 0.0,
            "surface_area": float(combined.area) if hasattr(combined, 'area') else 0.0,
            "center_of_buoyancy": combined.center_mass.tolist(),
            "bounds": combined.bounds.tolist(),
            "is_watertight": bool(combined.is_watertight) if hasattr(combined, 'is_watertight') else False
        }

        return combined, metadata

    def _generate_single_hull(self, hull_params: HullParameters, offset: float) -> trimesh.Trimesh:
        """
        Generate one hull of the catamaran.

        Args:
            hull_params: Hull design parameters
            offset: Y-axis offset (+ for starboard, - for port)
        """
        sections = []

        for i in range(self.num_cross_sections):
            # Longitudinal position (0 = stern, 1 = bow)
            x_ratio = i / (self.num_cross_sections - 1)
            x = x_ratio * hull_params.length_overall

            # Generate cross-section at this x position
            section = self._generate_cross_section(hull_params, x_ratio, x, offset)
            sections.append(section)

        # Loft sections together into 3D surface
        mesh = self._loft_sections(sections)

        return mesh

    def _generate_cross_section(self, hull_params: HullParameters,
                                x_ratio: float, x: float,
                                y_offset: float) -> np.ndarray:
        """
        Generate a single cross-section profile.

        Args:
            hull_params: Hull parameters
            x_ratio: Normalized position along length (0=stern, 1=bow)
            x: Actual x position in meters
            y_offset: Y-axis offset from centerline

        Returns:
            Array of (x, y, z) coordinates for section perimeter
        """
        # Hull taper (wider at middle, tapered at ends)
        if x_ratio < 0.1:
            # Stern - sharp taper
            width_factor = x_ratio / 0.1
        elif x_ratio > 0.9:
            # Bow - gradual taper
            width_factor = (1.0 - x_ratio) / 0.1
        else:
            # Midship - full width
            width_factor = 1.0

        width = hull_params.beam * width_factor
        draft = hull_params.draft if hull_params.draft else hull_params.hull_depth * 0.5
        deadrise_rad = np.radians(hull_params.deadrise_angle)

        # Generate points around cross-section perimeter
        points = []

        for i in range(self.points_per_section):
            # Parameter along perimeter (0 to 1)
            t = i / self.points_per_section

            if t < 0.25:
                # Bottom center to starboard (with deadrise)
                local_t = t / 0.25
                y_local = local_t * width / 2
                z = -draft + abs(y_local) * np.tan(deadrise_rad)
            elif t < 0.5:
                # Starboard side (vertical)
                local_t = (t - 0.25) / 0.25
                y_local = width / 2
                z = -draft + abs(y_local) * np.tan(deadrise_rad) + local_t * (draft - abs(y_local) * np.tan(deadrise_rad))
            elif t < 0.75:
                # Top starboard to port
                local_t = (t - 0.5) / 0.25
                y_local = width / 2 * (1 - 2 * local_t)
                z = 0
            else:
                # Port side (vertical down)
                local_t = (t - 0.75) / 0.25
                y_local = -width / 2
                z = -local_t * (draft - abs(y_local) * np.tan(deadrise_rad))

            points.append([x, y_local + y_offset, z])

        return np.array(points)

    def _loft_sections(self, sections: List[np.ndarray]) -> trimesh.Trimesh:
        """
        Connect cross-sections into 3D surface mesh.

        Args:
            sections: List of cross-section point arrays

        Returns:
            trimesh.Trimesh object
        """
        vertices = []
        faces = []

        # Flatten all section points into vertex list
        for section in sections:
            vertices.extend(section)

        vertices = np.array(vertices)

        # Generate triangular faces connecting sections
        n_pts = self.points_per_section

        for i in range(len(sections) - 1):
            for j in range(n_pts):
                # Current section vertex indices
                v1 = i * n_pts + j
                v2 = i * n_pts + (j + 1) % n_pts

                # Next section vertex indices
                v3 = (i + 1) * n_pts + j
                v4 = (i + 1) * n_pts + (j + 1) % n_pts

                # Two triangles per quad
                faces.append([v1, v2, v4])
                faces.append([v1, v4, v3])

        faces = np.array(faces)

        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def _generate_deck_platform(self, hull_params: HullParameters) -> trimesh.Trimesh:
        """Generate deck connecting the two hulls."""
        # Simple rectangular platform
        length = hull_params.length_overall * 0.8  # 80% of hull length
        width = hull_params.hull_spacing
        thickness = 0.1

        # Create box mesh
        deck = trimesh.creation.box(extents=[length, width, thickness])

        # Position at waterline
        draft = hull_params.draft if hull_params.draft else hull_params.hull_depth * 0.5
        deck.vertices[:, 2] += (hull_params.freeboard - draft)

        return deck

    def _generate_superstructure(self, hull_params: HullParameters) -> trimesh.Trimesh:
        """Generate simplified deckhouse/cabin."""
        # Small box representing cabin
        cabin_length = hull_params.length_overall * 0.3
        cabin_width = hull_params.hull_spacing * 0.6
        cabin_height = 2.0

        cabin = trimesh.creation.box(extents=[cabin_length, cabin_width, cabin_height])

        # Position on deck
        draft = hull_params.draft if hull_params.draft else hull_params.hull_depth * 0.5
        cabin.vertices[:, 2] += (hull_params.freeboard - draft + cabin_height/2 + 0.1)

        return cabin

    def export_stl(self, mesh: trimesh.Trimesh, path: str, binary: bool = True):
        """
        Export mesh to STL file.

        Args:
            mesh: Trimesh object to export
            path: Output file path
            binary: If True, use binary STL format (more compact)
        """
        mesh.export(path, file_type='stl')

    def generate_lod_meshes(self, hull_params: HullParameters) -> Dict[str, trimesh.Trimesh]:
        """
        Generate multiple Level-of-Detail versions.

        Returns:
            {"low": mesh, "medium": mesh, "high": mesh}
        """
        # High detail
        high_gen = HullGenerator(num_cross_sections=50, points_per_section=30)
        high_data = high_gen.generate(hull_params)
        high_mesh = high_data['mesh']

        # Medium detail
        medium_gen = HullGenerator(num_cross_sections=30, points_per_section=20)
        medium_data = medium_gen.generate(hull_params)
        medium_mesh = medium_data['mesh']

        # Low detail
        low_gen = HullGenerator(num_cross_sections=15, points_per_section=10)
        low_data = low_gen.generate(hull_params)
        low_mesh = low_data['mesh']

        return {
            "high": high_mesh,
            "medium": medium_mesh,
            "low": low_mesh
        }

    # ========================================================================
    # GEOMETRY METADATA CALCULATIONS
    # ========================================================================

    def _calculate_waterline_properties(self, hull_params: HullParameters) -> Dict[str, float]:
        """
        Calculate waterline geometric properties.

        Returns properties at the design waterline including:
        - Length at waterline (LWL)
        - Beam at waterline (BWL) per hull
        - Waterline area per hull
        - Overall beam (hull spacing + 2 × BWL/2)

        Args:
            hull_params: Hull parameters

        Returns:
            Dictionary of waterline properties
        """
        # Length at waterline (assume 98% of LOA for fine ends)
        lwl = hull_params.length_overall * 0.98

        # Beam at waterline (from parameters)
        bwl = hull_params.waterline_beam

        # Waterline area per hull (using prismatic coefficient)
        # AWL ≈ LWL × BWL × Cp_waterline
        # Use prismatic coefficient as approximation
        awl_per_hull = lwl * bwl * hull_params.prismatic_coefficient

        # Overall beam (center-to-center spacing + beam on each side)
        # This is the maximum beam including both hulls
        overall_beam = hull_params.hull_spacing + hull_params.beam

        return {
            'length_waterline': lwl,
            'beam_waterline_per_hull': bwl,
            'waterline_area_per_hull': awl_per_hull,
            'waterline_area_total': 2.0 * awl_per_hull,
            'overall_beam': overall_beam,
            'length_beam_ratio': lwl / bwl,  # Slenderness per hull
        }

    def _calculate_volume_properties(self, hull_params: HullParameters) -> Dict[str, float]:
        """
        Calculate volumetric properties.

        Estimates displaced volume using block coefficient approximation.

        Args:
            hull_params: Hull parameters

        Returns:
            Dictionary of volume properties
        """
        # Use draft if specified, otherwise estimate
        if hull_params.draft is not None:
            draft = hull_params.draft
        else:
            # Rough estimate: draft ≈ hull_depth * 0.5
            draft = hull_params.hull_depth * 0.5

        # Displacement volume per hull
        # V = L × B × T × Cb
        volume_per_hull = (
            hull_params.length_overall *
            hull_params.beam *
            draft *
            hull_params.block_coefficient
        )

        total_volume = 2.0 * volume_per_hull

        # Midship section area per hull
        # Assuming simplified shape: A_m ≈ B × T × C_m
        # Where C_m (midship coefficient) ≈ Cb / Cp
        midship_coefficient = hull_params.block_coefficient / hull_params.prismatic_coefficient
        midship_area = hull_params.beam * draft * midship_coefficient

        return {
            'draft': draft,
            'volume_per_hull': volume_per_hull,
            'total_volume': total_volume,
            'displacement_mass': total_volume * 1.025,  # Tons (seawater density)
            'midship_area_per_hull': midship_area,
            'midship_coefficient': midship_coefficient,
        }

    def _calculate_geometric_properties(self, hull_params: HullParameters) -> Dict[str, Any]:
        """
        Calculate derived geometric properties.

        Includes dimensional ratios, shape characteristics, and
        twin-hull specific properties.

        Args:
            hull_params: Hull parameters

        Returns:
            Dictionary of geometric properties
        """
        # Get ratios from hull parameters
        ratios = hull_params.get_primary_ratios()

        # Calculate additional properties
        draft = hull_params.draft if hull_params.draft else hull_params.hull_depth * 0.5

        # Volumetric properties
        volume_per_hull = (
            hull_params.length_overall *
            hull_params.beam *
            draft *
            hull_params.block_coefficient
        )

        # Length-displacement ratio (naval architecture measure of slenderness)
        # L/∇^(1/3) where ∇ is volume in cubic feet
        # Convert volume to cubic feet (1 m³ = 35.3147 ft³)
        total_volume_cuft = 2.0 * volume_per_hull * 35.3147
        length_disp_ratio = hull_params.length_overall / (total_volume_cuft ** (1.0/3.0))

        # Hull separation factor (ratio of spacing to beam)
        separation_factor = hull_params.hull_spacing / hull_params.beam

        return {
            **ratios,  # Include L/B, B/T, spacing_ratio, slenderness
            'length_displacement_ratio': length_disp_ratio,
            'separation_factor': separation_factor,
            'deadrise_radians': math.radians(hull_params.deadrise_angle),
            'lcb_position_from_stern': hull_params.lcb_position / 100.0 * hull_params.length_overall,
            'freeboard_draft_ratio': hull_params.freeboard / draft if draft > 0 else 0,
            'draft_depth_ratio': draft / hull_params.hull_depth,
        }

    def _calculate_station_properties(self, hull_params: HullParameters, n_stations: int = 11) -> List[Dict[str, float]]:
        """
        Calculate properties at stations along the hull length.

        Stations are cross-sections at regular intervals from bow to stern.
        Standard practice is 11 stations (0-10) with station 5 at midship.

        Args:
            hull_params: Hull parameters
            n_stations: Number of stations (default 11)

        Returns:
            List of dictionaries, one per station, with geometric properties
        """
        stations = []
        draft = hull_params.draft if hull_params.draft else hull_params.hull_depth * 0.5

        for i in range(n_stations):
            # Normalized position along length (0 = aft, 1 = forward)
            x_norm = i / (n_stations - 1)

            # Station position from aft perpendicular
            x_position = x_norm * hull_params.length_overall

            # Section area coefficient using prismatic distribution
            # Assume parabolic distribution peaking at midship (station 5)
            # C_s(x) = C_p × [1 - k × (x - 0.5)²]
            # Where k controls the fullness distribution

            # Distance from midship (normalized)
            x_mid_dist = abs(x_norm - 0.5)

            # Section coefficient (fuller at midship, finer at ends)
            # Use prismatic coefficient as guide
            if hull_params.prismatic_coefficient > 0.65:
                # Fuller hull - less taper
                section_coeff = hull_params.block_coefficient * (1.0 - 1.5 * x_mid_dist**2)
            else:
                # Finer hull - more taper
                section_coeff = hull_params.block_coefficient * (1.0 - 2.5 * x_mid_dist**2)

            section_coeff = max(0.0, section_coeff)  # Ensure non-negative

            # Section area
            section_area = hull_params.beam * draft * section_coeff

            # Beam at this station (tapers at ends)
            if x_norm < 0.1:  # Aft taper
                section_beam = hull_params.beam * (x_norm / 0.1)
            elif x_norm > 0.9:  # Forward taper
                section_beam = hull_params.beam * ((1.0 - x_norm) / 0.1)
            else:  # Parallel middle body
                section_beam = hull_params.beam

            stations.append({
                'station_number': i,
                'position_from_aft': x_position,
                'normalized_position': x_norm,
                'section_area': section_area,
                'section_beam': section_beam,
                'section_coefficient': section_coeff,
            })

        return stations

    def _validate_geometry(self, hull_params: HullParameters) -> Dict[str, Any]:
        """
        Validate geometric feasibility.

        Checks for physical impossibilities or extreme geometries.

        Args:
            hull_params: Hull parameters

        Returns:
            Dictionary with validation results
        """
        warnings = []
        errors = []

        # Check L/B ratio
        lb_ratio = hull_params.length_overall / hull_params.beam
        if lb_ratio < 6.0:
            warnings.append(f"Low L/B ratio ({lb_ratio:.2f}) - may have high resistance")
        elif lb_ratio > 20.0:
            warnings.append(f"High L/B ratio ({lb_ratio:.2f}) - may have structural issues")

        # Check hull spacing
        spacing_ratio = hull_params.hull_spacing / hull_params.length_overall
        if spacing_ratio < 0.20:
            warnings.append(f"Narrow hull spacing ({spacing_ratio:.3f} × LOA) - wave interference likely")
        elif spacing_ratio > 0.50:
            warnings.append(f"Wide hull spacing ({spacing_ratio:.3f} × LOA) - structural challenges")

        # Check prismatic coefficient vs block coefficient
        # Mathematically: Cp = Cb / Cm, so Cp should be less than Cb
        # But in practice Cm > 1.0 is impossible, so Cp ≈ Cb is the limit
        if hull_params.prismatic_coefficient < hull_params.block_coefficient:
            errors.append(
                f"Invalid Cp/Cb relationship: Cp ({hull_params.prismatic_coefficient:.3f}) "
                f"< Cb ({hull_params.block_coefficient:.3f}) implies Cm > 1.0"
            )

        # Check displacement feasibility
        draft = hull_params.draft if hull_params.draft else hull_params.hull_depth * 0.5
        estimated_volume = 2.0 * hull_params.length_overall * hull_params.beam * draft * hull_params.block_coefficient
        estimated_mass = estimated_volume * 1.025  # Tons

        displacement_error = abs(estimated_mass - hull_params.displacement) / hull_params.displacement
        if displacement_error > 0.5:
            warnings.append(
                f"Large displacement mismatch: estimated {estimated_mass:.1f}t "
                f"vs specified {hull_params.displacement:.1f}t ({displacement_error*100:.1f}% error)"
            )

        # Check freeboard adequacy
        if hull_params.freeboard < 0.5:
            warnings.append(f"Low freeboard ({hull_params.freeboard:.2f}m) - safety concern")

        is_valid = len(errors) == 0

        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'n_warnings': len(warnings),
            'n_errors': len(errors),
        }


def generate_hull_metadata(hull_params: HullParameters, generate_mesh: bool = False) -> Dict[str, Any]:
    """
    Convenience function to generate hull metadata and optionally 3D mesh.

    Args:
        hull_params: Hull parameters
        generate_mesh: If True, generate 3D mesh

    Returns:
        Dictionary of hull geometry metadata (and mesh if requested)
    """
    generator = HullGenerator(generate_mesh=generate_mesh)
    return generator.generate(hull_params)


def generate_hull_mesh(hull_params: HullParameters) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Convenience function to generate 3D hull mesh.

    Args:
        hull_params: Hull parameters

    Returns:
        (mesh, metadata) tuple
    """
    generator = HullGenerator(generate_mesh=True)
    data = generator.generate(hull_params)
    return data['mesh'], data['mesh_metadata']


if __name__ == "__main__":
    # Demonstrate hull generator
    from naval_domain.hull_parameters import get_baseline_catamaran, get_high_speed_catamaran
    import json

    print("=" * 70)
    print("HULL GENERATOR DEMONSTRATION")
    print("=" * 70)
    print()

    # Generate baseline catamaran
    baseline = get_baseline_catamaran()
    print("Generating BASELINE CATAMARAN hull...")
    print()

    hull_data = generate_hull_metadata(baseline)

    print("=== WATERLINE PROPERTIES ===")
    for key, value in hull_data['waterline_properties'].items():
        print(f"  {key}: {value:.3f}")

    print()
    print("=== VOLUME PROPERTIES ===")
    for key, value in hull_data['volume_properties'].items():
        print(f"  {key}: {value:.3f}")

    print()
    print("=== GEOMETRIC PROPERTIES ===")
    for key, value in hull_data['geometric_properties'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print()
    print("=== VALIDATION ===")
    validation = hull_data['validation']
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Errors: {validation['n_errors']}")
    print(f"  Warnings: {validation['n_warnings']}")

    if validation['warnings']:
        print("\n  Warnings:")
        for warning in validation['warnings']:
            print(f"    - {warning}")

    if validation['errors']:
        print("\n  Errors:")
        for error in validation['errors']:
            print(f"    - {error}")

    print()
    print("=== STATION PROPERTIES (sample) ===")
    stations = hull_data['station_properties']
    print(f"  Total stations: {len(stations)}")
    print(f"  Sample - Station 0 (aft): {stations[0]}")
    print(f"  Sample - Station 5 (midship): {stations[5]}")
    print(f"  Sample - Station 10 (forward): {stations[10]}")

    print()
    print("=" * 70)
    print()

    # Test high-speed catamaran
    print("Generating HIGH-SPEED CATAMARAN hull...")
    print()
    high_speed = get_high_speed_catamaran()
    hull_data_hs = generate_hull_metadata(high_speed)

    print("=== VALIDATION ===")
    validation_hs = hull_data_hs['validation']
    print(f"  Valid: {validation_hs['is_valid']}")
    print(f"  Warnings: {validation_hs['n_warnings']}")

    if validation_hs['warnings']:
        print("\n  Warnings:")
        for warning in validation_hs['warnings']:
            print(f"    - {warning}")

    print()
    print("=" * 70)
    print()

    # Show JSON export capability
    print("=== JSON EXPORT (sample) ===")
    json_sample = {
        'hull_type': hull_data['hull_type'],
        'waterline_properties': hull_data['waterline_properties'],
        'validation': hull_data['validation'],
    }
    print(json.dumps(json_sample, indent=2))
