"""
Baseline Design Library for Twin-Hull Naval Vessels

Provides curated starting designs for autonomous exploration.
All designs are returned as dictionaries (compatible with Agent 2 LLM interface).

Categories:
- General Purpose: Balanced performance across all metrics
- High Speed: Optimized for maximum speed
- Stability: Optimized for high GM and sea-keeping
- Efficiency: Optimized for low power consumption
"""

from typing import Dict, List, Any


# === BASELINE DESIGNS (as dictionaries for Agent 2 compatibility) ===

BASELINE_GENERAL: Dict[str, Any] = {
    'name': 'General Purpose 18m Catamaran',
    'description': 'Balanced design for coastal operations',
    'length_overall': 18.0,
    'beam': 2.0,
    'hull_depth': 2.2,
    'hull_spacing': 5.4,
    'deadrise_angle': 12.0,
    'freeboard': 1.4,
    'lcb_position': 48.0,
    'prismatic_coefficient': 0.60,
    'waterline_beam': 1.8,
    'block_coefficient': 0.42,
    'design_speed': 25.0,
    'displacement': 35.0,
    'draft': 0.8,
}

BASELINE_HIGH_SPEED: Dict[str, Any] = {
    'name': 'High-Speed 22m Catamaran',
    'description': 'Optimized for high-speed performance (35 knots)',
    'length_overall': 22.0,
    'beam': 1.6,
    'hull_depth': 2.5,
    'hull_spacing': 6.0,
    'deadrise_angle': 8.0,
    'freeboard': 1.9,
    'lcb_position': 50.0,
    'prismatic_coefficient': 0.68,
    'waterline_beam': 1.5,
    'block_coefficient': 0.38,
    'design_speed': 35.0,
    'displacement': 40.0,
    'draft': 0.6,
}

BASELINE_STABILITY: Dict[str, Any] = {
    'name': 'Stability-Optimized 16m Catamaran',
    'description': 'Wide hull spacing for maximum stability',
    'length_overall': 16.0,
    'beam': 2.4,
    'hull_depth': 2.0,
    'hull_spacing': 6.5,
    'deadrise_angle': 15.0,
    'freeboard': 1.2,
    'lcb_position': 46.0,
    'prismatic_coefficient': 0.56,
    'waterline_beam': 2.2,
    'block_coefficient': 0.45,
    'design_speed': 20.0,
    'displacement': 32.0,
    'draft': 0.8,
}

BASELINE_EFFICIENCY: Dict[str, Any] = {
    'name': 'Efficiency-Optimized 20m Catamaran',
    'description': 'Low drag for minimum fuel consumption',
    'length_overall': 20.0,
    'beam': 1.8,
    'hull_depth': 2.3,
    'hull_spacing': 5.5,
    'deadrise_angle': 10.0,
    'freeboard': 1.3,
    'lcb_position': 49.0,
    'prismatic_coefficient': 0.62,
    'waterline_beam': 1.7,
    'block_coefficient': 0.40,
    'design_speed': 22.0,
    'displacement': 30.0,
    'draft': 0.7,
}

BASELINE_COMPACT: Dict[str, Any] = {
    'name': 'Compact 12m Catamaran',
    'description': 'Small size for harbor/coastal operations',
    'length_overall': 12.0,
    'beam': 1.8,
    'hull_depth': 1.5,
    'hull_spacing': 3.5,
    'deadrise_angle': 15.0,
    'freeboard': 0.8,
    'lcb_position': 48.0,
    'prismatic_coefficient': 0.58,
    'waterline_beam': 1.6,
    'block_coefficient': 0.44,
    'design_speed': 18.0,
    'displacement': 15.0,
    'draft': 0.7,
}

BASELINE_LARGE: Dict[str, Any] = {
    'name': 'Large 30m Catamaran',
    'description': 'Large vessel for extended operations',
    'length_overall': 30.0,
    'beam': 2.8,
    'hull_depth': 3.0,
    'hull_spacing': 8.5,
    'deadrise_angle': 11.0,
    'freeboard': 1.8,
    'lcb_position': 49.0,
    'prismatic_coefficient': 0.64,
    'waterline_beam': 2.5,
    'block_coefficient': 0.43,
    'design_speed': 28.0,
    'displacement': 120.0,
    'draft': 1.2,
}


# === BASELINE COLLECTIONS ===

ALL_BASELINES: List[Dict[str, Any]] = [
    BASELINE_GENERAL,
    BASELINE_HIGH_SPEED,
    BASELINE_STABILITY,
    BASELINE_EFFICIENCY,
    BASELINE_COMPACT,
    BASELINE_LARGE,
]


# === ACCESS FUNCTIONS ===

def get_all_baselines() -> List[Dict[str, Any]]:
    """
    Get all baseline designs.

    Returns:
        List of all baseline design dicts
    """
    return [design.copy() for design in ALL_BASELINES]


def get_baseline_by_name(name: str) -> Dict[str, Any]:
    """
    Get specific baseline by name.

    Args:
        name: Design name (e.g., 'General Purpose 18m Catamaran')

    Returns:
        Baseline design dict

    Raises:
        ValueError: If name not found
    """
    for design in ALL_BASELINES:
        if design['name'] == name:
            return design.copy()

    raise ValueError(f"Baseline design '{name}' not found")


def get_baseline_general() -> Dict[str, Any]:
    """Get general-purpose baseline (18m)."""
    return BASELINE_GENERAL.copy()


def get_baseline_high_speed() -> Dict[str, Any]:
    """Get high-speed baseline (22m)."""
    return BASELINE_HIGH_SPEED.copy()


def get_baseline_stability() -> Dict[str, Any]:
    """Get stability-optimized baseline (16m)."""
    return BASELINE_STABILITY.copy()


def get_baseline_efficiency() -> Dict[str, Any]:
    """Get efficiency-optimized baseline (20m)."""
    return BASELINE_EFFICIENCY.copy()


def get_baseline_compact() -> Dict[str, Any]:
    """Get compact baseline (12m)."""
    return BASELINE_COMPACT.copy()


def get_baseline_large() -> Dict[str, Any]:
    """Get large baseline (30m)."""
    return BASELINE_LARGE.copy()


def get_design_summary(design: Dict[str, Any]) -> str:
    """
    Generate human-readable summary of a design.

    Args:
        design: Design dict

    Returns:
        Summary string
    """
    return (
        f"{design.get('name', 'Unnamed Design')}\n"
        f"  LOA: {design['length_overall']:.1f}m, "
        f"Beam: {design['beam']:.1f}m, "
        f"Spacing: {design['hull_spacing']:.1f}m\n"
        f"  Speed: {design['design_speed']:.0f} knots, "
        f"Displacement: {design['displacement']:.0f} tons\n"
        f"  {design.get('description', 'No description')}"
    )


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("BASELINE DESIGNS LIBRARY")
    print("=" * 70)
    print()

    print(f"Total Baseline Designs: {len(ALL_BASELINES)}")
    print()

    for design in ALL_BASELINES:
        print(get_design_summary(design))
        print()

    print("=" * 70)
    print()

    # Test retrieval functions
    print("Testing retrieval functions:")
    print()

    general = get_baseline_general()
    print(f"General: {general['name']}")

    high_speed = get_baseline_high_speed()
    print(f"High-Speed: {high_speed['name']}")

    stability = get_baseline_stability()
    print(f"Stability: {stability['name']}")

    print()
    print("=" * 70)
    print()

    # Test dict format (Agent 2 compatibility)
    print("Example design dict (Agent 2 compatible):")
    print()
    import json
    print(json.dumps(BASELINE_GENERAL, indent=2))
