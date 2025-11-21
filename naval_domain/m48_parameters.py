"""
M48-Specific Hull Parameters for Magnet Defense M48 Unmanned Surface Vessel

This module defines the parameter space specifically for the M48 platform,
a 48-meter twin-hull catamaran with proven heritage:
- 32,000+ NM of open-sea trials (Pacific, Caribbean, US East Coast)
- Proven 28-30 knots at 50% max payload
- 15,000 NM range at 15-20 knots sustained
- Sea State 9 validated (winds >68 kts)

Design Brief Reference: NAVSEA MASC N00024-25-R-6314 White Paper
Mission: High-Capacity Modular Attack Surface Craft (HC-MASC)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json

from naval_domain.hull_parameters import HullParameters


@dataclass
class M48HullParameters(HullParameters):
    """
    M48-specific hull parameters extending base HullParameters.

    Adds M48-specific attributes for:
    - Payload capacity (containerized mission modules)
    - Proven performance envelope (from 32,000 NM sea trials)
    - Mission equipment (sensors, weapons, autonomy)
    - Propulsion configuration (twin diesel, straight shaft)

    All validation ranges are constrained to M48 reality and NAVSEA requirements.
    """

    # === M48-SPECIFIC MISSION PARAMETERS ===
    payload_capacity: float = 144.0  # Metric tons (4 × 36-ton containers)
    fuel_capacity: float = 50000.0   # Liters (for 15,000 NM range estimation)

    # Mission module configuration
    mission_modules: List[str] = field(default_factory=lambda: ["ISR"])  # ISR, ASW, MCM, etc.

    # Propulsion configuration (M48 proven)
    propulsion_type: str = "twin_diesel_straight_shaft"  # Fixed configuration
    num_engines: int = 2  # One per hull
    generator_power: float = 200.0  # kW total (2 × 100 kW)

    # Structural configuration
    superstructure_removed: bool = True  # Unmanned variant removes upper 2 decks
    faceted_geometry: bool = True  # RCS reduction via faceted hull

    # Performance envelope (from sea trials)
    proven_max_speed: float = 30.0  # Knots (at 50% payload)
    proven_range: float = 15000.0  # Nautical miles (at cruise speed)
    proven_sea_state: int = 9  # WMO Sea State survivability

    def validate(self) -> None:
        """
        M48-specific validation extending base validation.

        Enforces M48 constraints:
        - LOA: 46-50m (centered on 48m proven hull)
        - Beam: 1.8-2.2m per hull
        - Hull spacing: 8-12m (catamaran configuration)
        - Displacement: 90-200 tons (structural + payload)
        - Design speed: 15-30 knots
        """
        errors = []

        # === M48-SPECIFIC DIMENSION CONSTRAINTS ===
        if not (46.0 <= self.length_overall <= 50.0):
            errors.append(
                f"M48 LOA must be 46-50m (proven 48m hull), got {self.length_overall}m"
            )

        if not (1.8 <= self.beam <= 2.2):
            errors.append(
                f"M48 beam must be 1.8-2.2m per hull, got {self.beam}m"
            )

        if not (8.0 <= self.hull_spacing <= 12.0):
            errors.append(
                f"M48 hull spacing must be 8-12m, got {self.hull_spacing}m"
            )

        # === M48 DISPLACEMENT CONSTRAINTS ===
        # Structural weight: ~92 short tons = 83.5 metric tons
        # Total displacement: structural + payload
        min_displacement = 90.0  # Minimum (structural weight)
        max_displacement = 250.0  # Maximum (structural + max payload)

        if not (min_displacement <= self.displacement <= max_displacement):
            errors.append(
                f"M48 displacement must be {min_displacement}-{max_displacement} tons, "
                f"got {self.displacement} tons"
            )

        # === M48 OPERATIONAL CONSTRAINTS ===
        if not (15.0 <= self.design_speed <= 32.0):
            errors.append(
                f"M48 design speed must be 15-32 knots (proven 28-30 kts), "
                f"got {self.design_speed} knots"
            )

        # === M48 PAYLOAD CONSTRAINTS ===
        if not (0.0 <= self.payload_capacity <= 200.0):
            errors.append(
                f"M48 payload capacity must be 0-200 tons, got {self.payload_capacity} tons"
            )

        # Displacement should be >= structural weight + payload
        structural_weight = 83.5  # metric tons (92 short tons)
        min_required_displacement = structural_weight + self.payload_capacity
        if self.displacement < min_required_displacement:
            errors.append(
                f"Displacement ({self.displacement} tons) must be >= "
                f"structural weight ({structural_weight} tons) + "
                f"payload ({self.payload_capacity} tons) = {min_required_displacement} tons"
            )

        # === M48 DRAFT CONSTRAINTS ===
        if self.draft is not None:
            if not (1.0 <= self.draft <= 2.5):
                errors.append(
                    f"M48 draft must be 1.0-2.5m, got {self.draft}m"
                )

        # === M48 HULL FORM CONSTRAINTS ===
        # Semi-displacement catamaran characteristics
        if not (0.55 <= self.prismatic_coefficient <= 0.70):
            errors.append(
                f"M48 Cp should be 0.55-0.70 for semi-displacement, "
                f"got {self.prismatic_coefficient}"
            )

        if not (0.38 <= self.block_coefficient <= 0.50):
            errors.append(
                f"M48 Cb should be 0.38-0.50 for efficiency, got {self.block_coefficient}"
            )

        # === MISSION MODULE VALIDATION ===
        valid_modules = ["ISR", "ASW", "MCM", "SURFACE_WARFARE", "LOGISTICS", "COMMUNICATIONS"]
        for module in self.mission_modules:
            if module not in valid_modules:
                errors.append(
                    f"Invalid mission module '{module}'. Valid: {valid_modules}"
                )

        if errors:
            raise ValueError(
                f"M48 parameter validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Call base validation for standard naval architecture checks
        # (but skip it if we already have M48-specific errors)
        try:
            super().validate()
        except ValueError as e:
            # Filter out errors that conflict with M48-specific validation
            base_errors = str(e).split('\n')[1:]  # Skip first line "validation failed"
            filtered_errors = []

            for error in base_errors:
                # Skip base validation errors for parameters we've already validated
                if not any(keyword in error for keyword in [
                    'length_overall', 'beam', 'hull_spacing',
                    'displacement', 'design_speed', 'draft'
                ]):
                    filtered_errors.append(error)

            if filtered_errors:
                raise ValueError(
                    f"M48 base validation failed:\n" + "\n".join(filtered_errors)
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert M48 parameters to dictionary including M48-specific fields."""
        base_dict = super().to_dict()
        base_dict.update({
            'payload_capacity': self.payload_capacity,
            'fuel_capacity': self.fuel_capacity,
            'mission_modules': self.mission_modules,
            'propulsion_type': self.propulsion_type,
            'num_engines': self.num_engines,
            'generator_power': self.generator_power,
            'superstructure_removed': self.superstructure_removed,
            'faceted_geometry': self.faceted_geometry,
            'proven_max_speed': self.proven_max_speed,
            'proven_range': self.proven_range,
            'proven_sea_state': self.proven_sea_state,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'M48HullParameters':
        """Create M48HullParameters from dictionary."""
        return cls(**data)

    def summary(self) -> str:
        """Generate human-readable summary of M48 parameters."""
        base_summary = super().summary()

        m48_specific = [
            "",
            "=== M48-SPECIFIC CONFIGURATION ===",
            f"Payload Capacity: {self.payload_capacity:.1f} metric tons",
            f"Fuel Capacity: {self.fuel_capacity:.0f} liters",
            f"Mission Modules: {', '.join(self.mission_modules)}",
            f"Propulsion: {self.propulsion_type}",
            f"Generator Power: {self.generator_power:.0f} kW",
            f"Superstructure: {'Removed (unmanned)' if self.superstructure_removed else 'Retained'}",
            f"RCS Reduction: {'Faceted geometry' if self.faceted_geometry else 'Standard'}",
            f"",
            "=== PROVEN PERFORMANCE (32,000 NM SEA TRIALS) ===",
            f"Max Speed: {self.proven_max_speed:.0f} knots (at 50% payload)",
            f"Range: {self.proven_range:.0f} NM (at cruise speed)",
            f"Sea State: {self.proven_sea_state} (winds >{68 if self.proven_sea_state == 9 else 'N/A'} kts)",
        ]

        return base_summary + "\n" + "\n".join(m48_specific)


# === M48 PARAMETER RANGES FOR AUTONOMOUS EXPLORATION ===

M48_PARAMETER_RANGES: Dict[str, tuple] = {
    # Primary dimensions (constrained to M48 envelope)
    'length_overall': (46.0, 50.0),  # Centered on 48m proven hull
    'beam': (1.8, 2.2),  # Per hull
    'hull_depth': (2.0, 3.5),
    'hull_spacing': (8.0, 12.0),

    # Hull geometry
    'deadrise_angle': (8.0, 15.0),  # Semi-displacement range
    'freeboard': (1.5, 2.5),
    'lcb_position': (46.0, 52.0),  # Slightly forward for heavy sensors

    # Hydrostatics (semi-displacement catamaran)
    'prismatic_coefficient': (0.55, 0.70),
    'waterline_beam': (1.6, 2.1),
    'block_coefficient': (0.38, 0.50),

    # Operational
    'design_speed': (15.0, 32.0),  # 15-20 cruise, 28-30 max proven
    'displacement': (90.0, 250.0),  # Structural + payload
    'draft': (1.0, 2.5),

    # M48-specific
    'payload_capacity': (0.0, 200.0),  # Mission modules + fuel tradeoff
    'fuel_capacity': (30000.0, 60000.0),  # Liters
}


if __name__ == "__main__":
    # Demonstration of M48 parameter validation
    print("=" * 70)
    print("M48 HULL PARAMETERS - VALIDATION DEMONSTRATION")
    print("=" * 70)
    print()

    # Valid M48 configuration
    print("Creating valid M48 unmanned baseline...")
    try:
        m48_baseline = M48HullParameters(
            # Primary dimensions (proven 48m hull)
            length_overall=48.0,
            beam=2.0,
            hull_depth=3.0,
            hull_spacing=10.0,

            # Hull geometry
            deadrise_angle=11.0,
            freeboard=1.8,
            lcb_position=49.0,

            # Hydrostatics
            prismatic_coefficient=0.64,
            waterline_beam=1.9,
            block_coefficient=0.43,

            # Operational
            design_speed=28.0,  # Proven max speed
            displacement=150.0,  # Structural + moderate payload
            draft=1.5,

            # M48-specific
            payload_capacity=60.0,  # Mission modules
            fuel_capacity=45000.0,
            mission_modules=["ISR", "COMMUNICATIONS"],
        )

        print("✓ Valid M48 configuration created successfully!")
        print()
        print(m48_baseline.summary())

    except ValueError as e:
        print(f"✗ Validation failed: {e}")

    print()
    print("=" * 70)
    print()

    # Test invalid configuration
    print("Testing invalid M48 configuration (35m hull, not M48)...")
    try:
        invalid_m48 = M48HullParameters(
            length_overall=35.0,  # Too short for M48!
            beam=2.0,
            hull_depth=2.5,
            hull_spacing=8.0,
            deadrise_angle=12.0,
            freeboard=1.5,
            lcb_position=48.0,
            prismatic_coefficient=0.60,
            waterline_beam=1.8,
            block_coefficient=0.42,
            design_speed=25.0,
            displacement=100.0,
            draft=1.2,
        )
        print("✗ Should have failed validation!")

    except ValueError as e:
        print("✓ Correctly rejected invalid configuration:")
        print(f"  {e}")

    print()
    print("=" * 70)
