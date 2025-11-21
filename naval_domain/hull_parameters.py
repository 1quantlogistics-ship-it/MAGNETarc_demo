"""
Hull Parameters Schema for Twin-Hull (Catamaran) Naval Vessels

This module defines the parameter space for autonomous hull design exploration.
All parameters are validated against physical constraints based on naval architecture
principles and empirical catamaran design data.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple
import json


@dataclass
class HullParameters:
    """
    Defines the complete parameter space for twin-hull vessel design.

    Parameters are organized by category:
    - Primary Dimensions: Overall size and proportions
    - Hull Geometry: Shape characteristics
    - Hydrostatic: Water interaction properties
    - Configuration: Twin-hull specific parameters

    All parameters include physical validation based on naval architecture constraints.
    """

    # === PRIMARY DIMENSIONS ===
    length_overall: float  # LOA in meters (overall length)
    beam: float           # Maximum beam width in meters (single hull)
    hull_depth: float     # Molded depth in meters

    # === TWIN-HULL CONFIGURATION ===
    hull_spacing: float   # Center-to-center distance between hulls in meters

    # === HULL GEOMETRY ===
    deadrise_angle: float      # Deadrise angle at midship in degrees (0-30°)
    freeboard: float           # Height of deck above waterline in meters
    lcb_position: float        # Longitudinal center of buoyancy (% of LOA from stern, 0-100)

    # === HYDROSTATIC PROPERTIES ===
    prismatic_coefficient: float  # Cp - ratio of displaced volume to volume of prism (0.5-0.75)
    waterline_beam: float         # Beam at waterline in meters
    block_coefficient: float      # Cb - ratio of displaced volume to L×B×T (0.35-0.55 for catamarans)

    # === OPERATIONAL PARAMETERS ===
    design_speed: float          # Design speed in knots
    displacement: float          # Design displacement in metric tons

    # === DERIVED/OPTIONAL ===
    draft: Optional[float] = None  # Design draft in meters (can be derived)

    def __post_init__(self):
        """Validate all parameters against physical constraints."""
        self.validate()

    def validate(self) -> None:
        """
        Validates all parameters against naval architecture constraints.

        Raises:
            ValueError: If any parameter violates physical constraints
        """
        errors = []

        # === PRIMARY DIMENSIONS VALIDATION ===
        if not (8.0 <= self.length_overall <= 50.0):
            errors.append(f"length_overall must be 8-50m, got {self.length_overall}m")

        if not (0.5 <= self.beam <= 6.0):
            errors.append(f"beam must be 0.5-6.0m, got {self.beam}m")

        if not (0.5 <= self.hull_depth <= 4.0):
            errors.append(f"hull_depth must be 0.5-4.0m, got {self.hull_depth}m")

        # === TWIN-HULL CONFIGURATION VALIDATION ===
        if not (2.0 <= self.hull_spacing <= 15.0):
            errors.append(f"hull_spacing must be 2-15m, got {self.hull_spacing}m")

        # Hull spacing should be reasonable relative to LOA (typically 0.2-0.5 × LOA)
        spacing_ratio = self.hull_spacing / self.length_overall
        if not (0.15 <= spacing_ratio <= 0.6):
            errors.append(
                f"hull_spacing/LOA ratio must be 0.15-0.6, got {spacing_ratio:.3f}"
            )

        # === HULL GEOMETRY VALIDATION ===
        if not (0.0 <= self.deadrise_angle <= 30.0):
            errors.append(f"deadrise_angle must be 0-30°, got {self.deadrise_angle}°")

        if not (0.3 <= self.freeboard <= 3.0):
            errors.append(f"freeboard must be 0.3-3.0m, got {self.freeboard}m")

        if not (40.0 <= self.lcb_position <= 60.0):
            errors.append(
                f"lcb_position must be 40-60% of LOA, got {self.lcb_position}%"
            )

        # === HYDROSTATIC PROPERTIES VALIDATION ===
        if not (0.50 <= self.prismatic_coefficient <= 0.75):
            errors.append(
                f"prismatic_coefficient must be 0.50-0.75, got {self.prismatic_coefficient}"
            )

        if not (0.35 <= self.block_coefficient <= 0.55):
            errors.append(
                f"block_coefficient must be 0.35-0.55 for catamarans, got {self.block_coefficient}"
            )

        if not (0.3 <= self.waterline_beam <= 5.0):
            errors.append(f"waterline_beam must be 0.3-5.0m, got {self.waterline_beam}m")

        # Waterline beam should be less than or equal to maximum beam
        if self.waterline_beam > self.beam:
            errors.append(
                f"waterline_beam ({self.waterline_beam}m) cannot exceed beam ({self.beam}m)"
            )

        # === OPERATIONAL PARAMETERS VALIDATION ===
        if not (10.0 <= self.design_speed <= 45.0):
            errors.append(f"design_speed must be 10-45 knots, got {self.design_speed} knots")

        if not (5.0 <= self.displacement <= 500.0):
            errors.append(
                f"displacement must be 5-500 metric tons, got {self.displacement} tons"
            )

        # === DERIVED PARAMETERS VALIDATION ===
        if self.draft is not None:
            if not (0.3 <= self.draft <= 3.5):
                errors.append(f"draft must be 0.3-3.5m, got {self.draft}m")

            # Draft should be less than hull depth
            if self.draft >= self.hull_depth:
                errors.append(
                    f"draft ({self.draft}m) must be less than hull_depth ({self.hull_depth}m)"
                )

            # Freeboard check: freeboard = hull_depth - draft (approximately)
            expected_freeboard = self.hull_depth - self.draft
            if abs(self.freeboard - expected_freeboard) > 0.5:
                errors.append(
                    f"Freeboard inconsistency: freeboard={self.freeboard}m, "
                    f"but hull_depth-draft={expected_freeboard:.2f}m"
                )

        # === RATIO VALIDATIONS ===
        # Length-to-beam ratio (L/B) for catamarans typically 8-15 per hull
        lb_ratio = self.length_overall / self.beam
        if not (6.0 <= lb_ratio <= 20.0):
            errors.append(
                f"Length/Beam ratio should be 6-20 for efficiency, got {lb_ratio:.2f}"
            )

        # Beam-to-draft ratio (if draft is specified)
        if self.draft is not None:
            bd_ratio = self.beam / self.draft
            if not (2.0 <= bd_ratio <= 12.0):
                errors.append(
                    f"Beam/Draft ratio should be 2-12 for stability, got {bd_ratio:.2f}"
                )

        if errors:
            raise ValueError(
                f"Hull parameter validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary format."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HullParameters':
        """Create HullParameters from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'HullParameters':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_primary_ratios(self) -> Dict[str, float]:
        """
        Calculate primary naval architecture ratios.

        Returns:
            Dictionary with key design ratios:
            - L/B: Length-to-beam ratio
            - B/T: Beam-to-draft ratio (if draft available)
            - spacing_ratio: Hull spacing / LOA
            - slenderness: LOA / (beam * hull_spacing)^0.5
        """
        ratios = {
            'L/B': self.length_overall / self.beam,
            'spacing_ratio': self.hull_spacing / self.length_overall,
            'slenderness': self.length_overall / (self.beam * self.hull_spacing) ** 0.5,
        }

        if self.draft is not None:
            ratios['B/T'] = self.beam / self.draft

        return ratios

    def summary(self) -> str:
        """
        Generate human-readable summary of hull parameters.

        Returns:
            Multi-line string summary
        """
        ratios = self.get_primary_ratios()

        summary_lines = [
            "=== HULL PARAMETERS SUMMARY ===",
            f"Length Overall: {self.length_overall:.2f}m",
            f"Beam (per hull): {self.beam:.2f}m",
            f"Hull Spacing: {self.hull_spacing:.2f}m",
            f"Depth: {self.hull_depth:.2f}m",
            f"Draft: {self.draft:.2f}m" if self.draft else "Draft: Not specified",
            f"",
            f"Design Speed: {self.design_speed:.1f} knots",
            f"Displacement: {self.displacement:.1f} metric tons",
            f"",
            f"Prismatic Coeff: {self.prismatic_coefficient:.3f}",
            f"Block Coeff: {self.block_coefficient:.3f}",
            f"Deadrise Angle: {self.deadrise_angle:.1f}°",
            f"",
            f"=== KEY RATIOS ===",
            f"L/B: {ratios['L/B']:.2f}",
            f"Spacing/LOA: {ratios['spacing_ratio']:.3f}",
            f"Slenderness: {ratios['slenderness']:.2f}",
        ]

        if 'B/T' in ratios:
            summary_lines.append(f"B/T: {ratios['B/T']:.2f}")

        return "\n".join(summary_lines)


# === BASELINE CONFIGURATIONS ===

def get_baseline_catamaran() -> HullParameters:
    """
    Returns baseline 18m catamaran configuration.

    This represents a proven, efficient twin-hull design suitable for
    coastal operations. Used as reference point for autonomous exploration.

    Based on typical power catamaran characteristics:
    - 18m LOA
    - Medium speed (25 knots)
    - Good stability (wide hull spacing)
    - Efficient hull form (Cp ≈ 0.60)

    Returns:
        HullParameters: Validated baseline configuration
    """
    return HullParameters(
        # Primary dimensions
        length_overall=18.0,      # 18m LOA - medium size catamaran
        beam=2.0,                 # 2.0m beam per hull
        hull_depth=2.2,           # 2.2m depth

        # Twin-hull configuration
        hull_spacing=5.4,         # 5.4m center-to-center (0.3 × LOA)

        # Hull geometry
        deadrise_angle=12.0,      # 12° deadrise - good compromise
        freeboard=1.4,            # 1.4m freeboard (hull_depth - draft)
        lcb_position=48.0,        # LCB at 48% LOA (slightly forward)

        # Hydrostatic properties
        prismatic_coefficient=0.60,  # Moderate Cp for efficiency
        waterline_beam=1.8,          # Slightly less than max beam
        block_coefficient=0.42,      # Typical for catamarans

        # Operational
        design_speed=25.0,        # 25 knots design speed
        displacement=35.0,        # 35 metric tons
        draft=0.8,               # 0.8m draft (B/T = 2.5)
    )


def get_high_speed_catamaran() -> HullParameters:
    """
    Returns high-speed catamaran configuration.

    Optimized for speed with:
    - Slender hulls (high L/B)
    - Higher prismatic coefficient
    - Lower deadrise for reduced wetted area

    Returns:
        HullParameters: High-speed configuration
    """
    return HullParameters(
        length_overall=22.0,
        beam=1.6,                # Slender hulls
        hull_depth=2.5,
        hull_spacing=6.0,
        deadrise_angle=8.0,      # Lower deadrise
        freeboard=1.9,           # 1.9m freeboard
        lcb_position=50.0,
        prismatic_coefficient=0.68,  # Higher Cp for speed
        waterline_beam=1.5,
        block_coefficient=0.38,  # Lower Cb for speed
        design_speed=35.0,       # High speed
        displacement=40.0,
        draft=0.6,               # 0.6m draft (B/T = 2.67)
    )


def get_stability_optimized_catamaran() -> HullParameters:
    """
    Returns stability-optimized catamaran configuration.

    Optimized for stability with:
    - Wide hull spacing
    - Higher beam
    - Lower LCB for better initial stability

    Returns:
        HullParameters: Stability-optimized configuration
    """
    return HullParameters(
        length_overall=16.0,
        beam=2.4,                # Wider hulls
        hull_depth=2.0,
        hull_spacing=6.5,        # Wide spacing (0.41 × LOA)
        deadrise_angle=15.0,     # Higher deadrise for sea-keeping
        freeboard=1.2,           # 1.2m freeboard
        lcb_position=46.0,       # LCB forward for stability
        prismatic_coefficient=0.56,  # Lower Cp
        waterline_beam=2.2,
        block_coefficient=0.45,  # Higher Cb
        design_speed=20.0,       # Moderate speed
        displacement=32.0,
        draft=0.8,               # 0.8m draft (B/T = 3.0)
    )


# === PARAMETER RANGES FOR EXPLORATION ===

PARAMETER_RANGES: Dict[str, Tuple[float, float]] = {
    'length_overall': (8.0, 50.0),
    'beam': (0.5, 6.0),
    'hull_depth': (0.5, 4.0),
    'hull_spacing': (2.0, 15.0),
    'deadrise_angle': (0.0, 30.0),
    'freeboard': (0.3, 3.0),
    'lcb_position': (40.0, 60.0),
    'prismatic_coefficient': (0.50, 0.75),
    'waterline_beam': (0.3, 5.0),
    'block_coefficient': (0.35, 0.55),
    'design_speed': (10.0, 45.0),
    'displacement': (5.0, 500.0),
    'draft': (0.3, 3.5),
}


if __name__ == "__main__":
    # Demonstrate baseline configurations
    print("=" * 60)
    print("BASELINE CATAMARAN")
    print("=" * 60)
    baseline = get_baseline_catamaran()
    print(baseline.summary())
    print("\n")

    print("=" * 60)
    print("HIGH-SPEED CATAMARAN")
    print("=" * 60)
    high_speed = get_high_speed_catamaran()
    print(high_speed.summary())
    print("\n")

    print("=" * 60)
    print("STABILITY-OPTIMIZED CATAMARAN")
    print("=" * 60)
    stability = get_stability_optimized_catamaran()
    print(stability.summary())
    print("\n")

    # Test serialization
    print("=" * 60)
    print("JSON SERIALIZATION TEST")
    print("=" * 60)
    json_str = baseline.to_json()
    print(json_str)

    # Test deserialization
    reconstructed = HullParameters.from_json(json_str)
    print("\nReconstructed successfully:", reconstructed.length_overall == baseline.length_overall)
