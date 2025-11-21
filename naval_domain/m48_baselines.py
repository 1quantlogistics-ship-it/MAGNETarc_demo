"""
M48 Baseline Design Library

Provides curated M48-specific starting designs for autonomous exploration,
all based on proven Magnet Defense M48 platform with 32,000 NM sea trial validation.

All baselines are variants of the 48-meter twin-hull catamaran optimized for
different NAVSEA HC-MASC mission profiles:

Mission Categories:
1. PROVEN_CREWED: Original crewed variant (32,000 NM sea trials)
2. UNMANNED_BASELINE: Unmanned conversion (remove upper decks, lower CG)
3. EXTENDED_RANGE: Optimized for >15,000 NM endurance missions
4. HIGH_PAYLOAD: 4×36-ton container configuration (144 tons payload)
5. ISR_OPTIMIZED: Sensor platform with MUOS antenna + tracking systems
6. ASW_MCM: Mine warfare / Anti-submarine warfare module carrier

All designs are returned as dictionaries for Agent 2 LLM interface compatibility.
"""

from typing import Dict, List, Any


# === M48 BASELINE DESIGNS ===

M48_PROVEN_CREWED: Dict[str, Any] = {
    'name': 'M48 Proven Crewed Variant',
    'description': 'Original 48m crewed catamaran with 32,000 NM sea trials (baseline reference)',

    # Primary dimensions (proven)
    'length_overall': 48.0,
    'beam': 2.0,
    'hull_depth': 3.2,  # Includes upper deck superstructure
    'hull_spacing': 10.0,

    # Hull geometry (proven)
    'deadrise_angle': 11.0,
    'freeboard': 2.0,  # Higher with superstructure
    'lcb_position': 49.0,

    # Hydrostatics (semi-displacement)
    'prismatic_coefficient': 0.64,
    'waterline_beam': 1.9,
    'block_coefficient': 0.43,

    # Operational (proven performance)
    'design_speed': 28.0,  # 28-30 kts proven at 50% payload
    'displacement': 165.0,  # Includes crew accommodations, research lab
    'draft': 1.6,

    # M48-specific
    'payload_capacity': 70.0,  # Research equipment + supplies
    'fuel_capacity': 45000.0,
    'mission_modules': ["RESEARCH", "CREW_SUPPORT"],
    'superstructure_removed': False,  # Crewed variant retains superstructure
    'faceted_geometry': False,  # Standard hull form
    'proven_max_speed': 30.0,
    'proven_range': 15000.0,
    'proven_sea_state': 9,
}

M48_UNMANNED_BASELINE: Dict[str, Any] = {
    'name': 'M48 Unmanned Baseline',
    'description': 'Unmanned conversion: remove upper 2 decks, lower CG, add faceted RCS reduction',

    # Primary dimensions (same hull)
    'length_overall': 48.0,
    'beam': 2.0,
    'hull_depth': 3.0,  # Reduced (no upper decks)
    'hull_spacing': 10.0,

    # Hull geometry (optimized for unmanned)
    'deadrise_angle': 11.0,
    'freeboard': 1.8,  # Lower CG = improved stability
    'lcb_position': 49.0,

    # Hydrostatics
    'prismatic_coefficient': 0.64,
    'waterline_beam': 1.9,
    'block_coefficient': 0.43,

    # Operational (same as crewed at lighter weight)
    'design_speed': 28.0,
    'displacement': 145.0,  # Lighter (no crew spaces, research lab)
    'draft': 1.5,

    # M48-specific (unmanned advantages)
    'payload_capacity': 60.0,  # Mission modules
    'fuel_capacity': 45000.0,
    'mission_modules': ["ISR", "COMMUNICATIONS"],
    'superstructure_removed': True,  # Weight reduction + stability improvement
    'faceted_geometry': True,  # RCS reduction for contested environments
    'proven_max_speed': 30.0,  # Same propulsion as crewed
    'proven_range': 16000.0,  # Better range (lighter weight)
    'proven_sea_state': 9,
}

M48_EXTENDED_RANGE: Dict[str, Any] = {
    'name': 'M48 Extended Range Variant',
    'description': 'Optimized for >18,000 NM endurance (fuel prioritized over payload)',

    # Primary dimensions
    'length_overall': 48.0,
    'beam': 2.0,
    'hull_depth': 3.0,
    'hull_spacing': 10.0,

    # Hull geometry (optimized for cruise efficiency)
    'deadrise_angle': 10.0,  # Lower drag
    'freeboard': 1.8,
    'lcb_position': 49.5,  # Slightly aft for cruise trim

    # Hydrostatics (low drag)
    'prismatic_coefficient': 0.66,  # Higher Cp for cruise efficiency
    'waterline_beam': 1.85,  # Slightly narrower for lower wetted area
    'block_coefficient': 0.41,  # Lower Cb = less drag

    # Operational (cruise optimized)
    'design_speed': 18.0,  # Cruise speed (max range)
    'displacement': 135.0,  # Minimized for efficiency
    'draft': 1.4,

    # M48-specific (range priority)
    'payload_capacity': 30.0,  # Reduced payload for fuel weight
    'fuel_capacity': 55000.0,  # +22% fuel capacity
    'mission_modules': ["ISR", "COMMUNICATIONS"],  # Persistent surveillance
    'superstructure_removed': True,
    'faceted_geometry': True,
    'proven_max_speed': 30.0,
    'proven_range': 18500.0,  # >23% range improvement
    'proven_sea_state': 9,
}

M48_HIGH_PAYLOAD: Dict[str, Any] = {
    'name': 'M48 High Payload Variant',
    'description': '4×36-ton containerized mission modules (144 tons payload capacity)',

    # Primary dimensions
    'length_overall': 48.0,
    'beam': 2.1,  # Slightly wider for payload stability
    'hull_depth': 3.0,
    'hull_spacing': 11.0,  # Wider spacing for heavy payload stability

    # Hull geometry (stability priority)
    'deadrise_angle': 12.0,  # Higher for sea-keeping under load
    'freeboard': 1.6,  # Lower with heavy payload (deeper draft)
    'lcb_position': 48.5,  # Forward for heavy deck loads

    # Hydrostatics (volume priority)
    'prismatic_coefficient': 0.62,
    'waterline_beam': 2.0,
    'block_coefficient': 0.45,  # Higher for displacement volume

    # Operational (heavy load)
    'design_speed': 24.0,  # Reduced speed with max payload
    'displacement': 235.0,  # Structural (83.5) + payload (144) + fuel
    'draft': 1.9,  # Deeper draft under load

    # M48-specific (max payload)
    'payload_capacity': 144.0,  # 4 × 36-ton containers (threshold requirement)
    'fuel_capacity': 35000.0,  # Reduced fuel for payload weight
    'mission_modules': ["SURFACE_WARFARE", "ASW", "MCM", "LOGISTICS"],
    'superstructure_removed': True,
    'faceted_geometry': True,
    'proven_max_speed': 28.0,  # Lower at max payload
    'proven_range': 8000.0,  # Reduced range with max payload
    'proven_sea_state': 9,
}

M48_ISR_OPTIMIZED: Dict[str, Any] = {
    'name': 'M48 ISR Platform',
    'description': 'Sensor platform with MUOS antenna, missile tracking, elevated surveillance masts',

    # Primary dimensions
    'length_overall': 48.0,
    'beam': 2.0,
    'hull_depth': 3.0,
    'hull_spacing': 10.5,  # Wide spacing for sensor mast stability

    # Hull geometry (sensor platform)
    'deadrise_angle': 11.0,
    'freeboard': 1.9,  # Higher for sensor mounting
    'lcb_position': 48.0,  # Forward for tall aft sensor mast balance

    # Hydrostatics
    'prismatic_coefficient': 0.63,
    'waterline_beam': 1.9,
    'block_coefficient': 0.42,

    # Operational (loiter + sprint)
    'design_speed': 22.0,  # Loiter speed for persistent ISR
    'displacement': 155.0,  # Moderate (sensors + MUOS + tracking systems)
    'draft': 1.55,

    # M48-specific (ISR mission)
    'payload_capacity': 50.0,  # Sensor packages + MUOS antenna + power
    'fuel_capacity': 48000.0,
    'mission_modules': ["ISR", "COMMUNICATIONS", "MISSILE_TRACKING"],
    'superstructure_removed': True,  # But radar arch for sensors
    'faceted_geometry': True,  # RCS reduction for survivability
    'proven_max_speed': 30.0,  # Sprint capability for repositioning
    'proven_range': 14000.0,
    'proven_sea_state': 9,  # Critical for sensor stability in heavy seas
}

M48_ASW_MCM: Dict[str, Any] = {
    'name': 'M48 ASW/MCM Carrier',
    'description': 'Mine warfare & anti-submarine warfare module carrier (LCS mission replacement)',

    # Primary dimensions
    'length_overall': 48.0,
    'beam': 2.0,
    'hull_depth': 3.0,
    'hull_spacing': 10.0,

    # Hull geometry (UUV launch/recovery)
    'deadrise_angle': 11.0,
    'freeboard': 1.7,
    'lcb_position': 49.0,

    # Hydrostatics
    'prismatic_coefficient': 0.64,
    'waterline_beam': 1.9,
    'block_coefficient': 0.43,

    # Operational (mission speed)
    'design_speed': 20.0,  # MCM/ASW operating speed
    'displacement': 170.0,  # ASW sensors + UUV/USV + MCM modules
    'draft': 1.6,

    # M48-specific (ASW/MCM mission)
    'payload_capacity': 80.0,  # Sonar, UUV, MCM modules, launch/recovery systems
    'fuel_capacity': 42000.0,
    'mission_modules': ["ASW", "MCM"],
    'superstructure_removed': True,
    'faceted_geometry': False,  # ASW requires smooth hull (sonar interference)
    'proven_max_speed': 28.0,
    'proven_range': 12000.0,  # Moderate range (mission area operations)
    'proven_sea_state': 9,
}

M48_PICKET_SHIP: Dict[str, Any] = {
    'name': 'M48 Picket Ship / Sensor Node',
    'description': 'Distributed sensor node for fleet operations (enable EMCON for capital ships)',

    # Primary dimensions
    'length_overall': 48.0,
    'beam': 2.0,
    'hull_depth': 3.0,
    'hull_spacing': 10.5,  # Wide for sensor stability

    # Hull geometry (persistent station-keeping)
    'deadrise_angle': 11.0,
    'freeboard': 1.9,
    'lcb_position': 48.5,

    # Hydrostatics
    'prismatic_coefficient': 0.63,
    'waterline_beam': 1.9,
    'block_coefficient': 0.42,

    # Operational (loiter + sprint)
    'design_speed': 15.0,  # Loiter for persistent coverage
    'displacement': 150.0,
    'draft': 1.5,

    # M48-specific (picket mission)
    'payload_capacity': 45.0,  # Radar, EO/IR, SIGINT, datalink
    'fuel_capacity': 50000.0,  # Extended loiter time
    'mission_modules': ["ISR", "COMMUNICATIONS", "EW_PASSIVE"],
    'superstructure_removed': True,
    'faceted_geometry': True,  # Low observability for picket role
    'proven_max_speed': 30.0,  # Sprint to reposition
    'proven_range': 17000.0,  # Extended endurance
    'proven_sea_state': 9,
}


# === BASELINE COLLECTIONS ===

ALL_M48_BASELINES: List[Dict[str, Any]] = [
    M48_PROVEN_CREWED,
    M48_UNMANNED_BASELINE,
    M48_EXTENDED_RANGE,
    M48_HIGH_PAYLOAD,
    M48_ISR_OPTIMIZED,
    M48_ASW_MCM,
    M48_PICKET_SHIP,
]


# === ACCESS FUNCTIONS ===

def get_all_m48_baselines() -> List[Dict[str, Any]]:
    """Get all M48 baseline designs."""
    return [design.copy() for design in ALL_M48_BASELINES]


def get_m48_baseline_by_name(name: str) -> Dict[str, Any]:
    """
    Get specific M48 baseline by name.

    Args:
        name: Design name (e.g., 'M48 Unmanned Baseline')

    Returns:
        M48 baseline design dict

    Raises:
        ValueError: If name not found
    """
    for design in ALL_M48_BASELINES:
        if design['name'] == name:
            return design.copy()

    raise ValueError(f"M48 baseline design '{name}' not found")


def get_m48_proven_crewed() -> Dict[str, Any]:
    """Get original crewed M48 variant (32,000 NM sea trials)."""
    return M48_PROVEN_CREWED.copy()


def get_m48_unmanned_baseline() -> Dict[str, Any]:
    """Get unmanned conversion baseline."""
    return M48_UNMANNED_BASELINE.copy()


def get_m48_extended_range() -> Dict[str, Any]:
    """Get extended range variant (>18,000 NM)."""
    return M48_EXTENDED_RANGE.copy()


def get_m48_high_payload() -> Dict[str, Any]:
    """Get high payload variant (4×36-ton containers)."""
    return M48_HIGH_PAYLOAD.copy()


def get_m48_isr_optimized() -> Dict[str, Any]:
    """Get ISR platform variant (MUOS + tracking sensors)."""
    return M48_ISR_OPTIMIZED.copy()


def get_m48_asw_mcm() -> Dict[str, Any]:
    """Get ASW/MCM variant (mine warfare + anti-submarine)."""
    return M48_ASW_MCM.copy()


def get_m48_picket_ship() -> Dict[str, Any]:
    """Get picket ship / sensor node variant."""
    return M48_PICKET_SHIP.copy()


def get_m48_design_summary(design: Dict[str, Any]) -> str:
    """
    Generate human-readable summary of M48 design.

    Args:
        design: M48 design dict

    Returns:
        Summary string
    """
    mission_modules_str = ', '.join(design.get('mission_modules', []))

    return (
        f"{design.get('name', 'Unnamed M48 Design')}\n"
        f"  {design.get('description', 'No description')}\n"
        f"  LOA: {design['length_overall']:.1f}m, "
        f"Beam: {design['beam']:.1f}m, "
        f"Spacing: {design['hull_spacing']:.1f}m\n"
        f"  Speed: {design['design_speed']:.0f} knots (max {design.get('proven_max_speed', 30):.0f} kts), "
        f"Range: {design.get('proven_range', 15000):.0f} NM\n"
        f"  Displacement: {design['displacement']:.0f} tons, "
        f"Payload: {design.get('payload_capacity', 0):.0f} tons\n"
        f"  Mission: {mission_modules_str}\n"
        f"  RCS Reduction: {'Yes' if design.get('faceted_geometry', False) else 'No'}, "
        f"Unmanned: {'Yes' if design.get('superstructure_removed', True) else 'No'}"
    )


if __name__ == "__main__":
    # Demonstration
    print("=" * 80)
    print("M48 BASELINE DESIGNS LIBRARY")
    print("=" * 80)
    print()

    print(f"Total M48 Baseline Designs: {len(ALL_M48_BASELINES)}")
    print()

    for design in ALL_M48_BASELINES:
        print(get_m48_design_summary(design))
        print()

    print("=" * 80)
    print()

    # Test retrieval functions
    print("Testing retrieval functions:")
    print()

    unmanned = get_m48_unmanned_baseline()
    print(f"Unmanned Baseline: {unmanned['name']}")
    print(f"  Displacement: {unmanned['displacement']} tons")
    print(f"  Payload: {unmanned['payload_capacity']} tons")

    print()

    extended_range = get_m48_extended_range()
    print(f"Extended Range: {extended_range['name']}")
    print(f"  Range: {extended_range['proven_range']} NM")
    print(f"  Fuel: {extended_range['fuel_capacity']} liters")

    print()

    high_payload = get_m48_high_payload()
    print(f"High Payload: {high_payload['name']}")
    print(f"  Payload: {high_payload['payload_capacity']} tons (4×36-ton containers)")
    print(f"  Hull Spacing: {high_payload['hull_spacing']} m (wide for stability)")

    print()
    print("=" * 80)
    print()

    # Test dict format (Agent 2 compatibility)
    print("Example M48 design dict (Agent 2 compatible):")
    print()
    import json
    print(json.dumps(M48_UNMANNED_BASELINE, indent=2))
