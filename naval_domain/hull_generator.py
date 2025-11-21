"""
Hull Generator for Twin-Hull (Catamaran) Vessels

This module generates hull geometry metadata from HullParameters.
For now, this is a simplified implementation that produces geometric
characteristics without full 3D mesh generation.

Future enhancements could include:
- 3D mesh generation (vertices, faces)
- STL/OBJ export
- Hull surface parametrization
- Integration with CAD tools (Rhino, etc.)
"""

from typing import Dict, Any, List, Tuple
import math

from naval_domain.hull_parameters import HullParameters


class HullGenerator:
    """
    Generates hull geometry metadata from parameters.

    This is a simplified generator that produces key geometric
    characteristics without full 3D meshing. Suitable for rapid
    design space exploration and physics simulation.
    """

    def __init__(self, resolution: str = 'low'):
        """
        Initialize hull generator.

        Args:
            resolution: Mesh resolution ('low', 'medium', 'high')
                       Currently unused, reserved for future 3D meshing
        """
        self.resolution = resolution

    def generate(self, hull_params: HullParameters) -> Dict[str, Any]:
        """
        Generate hull geometry metadata.

        Args:
            hull_params: Hull parameters defining the design

        Returns:
            Dictionary containing hull geometry metadata:
            {
                'hull_type': 'twin_hull',
                'n_hulls': 2,
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

        # Generate metadata
        hull_data = {
            'hull_type': 'twin_hull',
            'n_hulls': 2,
            'waterline_properties': self._calculate_waterline_properties(hull_params),
            'volume_properties': self._calculate_volume_properties(hull_params),
            'geometric_properties': self._calculate_geometric_properties(hull_params),
            'station_properties': self._calculate_station_properties(hull_params),
            'validation': self._validate_geometry(hull_params),
        }

        return hull_data

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


def generate_hull_metadata(hull_params: HullParameters) -> Dict[str, Any]:
    """
    Convenience function to generate hull metadata.

    Args:
        hull_params: Hull parameters

    Returns:
        Dictionary of hull geometry metadata
    """
    generator = HullGenerator()
    return generator.generate(hull_params)


if __name__ == "__main__":
    # Demonstrate hull generator
    from hull_parameters import get_baseline_catamaran, get_high_speed_catamaran
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
