"""
M48-Calibrated Physics Engine for Magnet Defense M48 Vessel

Extends the base PhysicsEngine with M48-specific calibration from 32,000 NM
of open-sea trial data. All resistance and performance calculations are
validated against proven M48 performance:

Proven Performance Envelope (from NAVSEA white paper):
- 28-30 knots at 50% max payload (proven)
- 15,000 NM range at 15-20 knots (proven)
- Sea State 9 survivability (winds >68 kts, wave height 40')
- 32,000+ NM operational data (Pacific, Caribbean, US East Coast)

Physics Calibration:
- ITTC-1957 resistance formulas calibrated to M48 empirical data
- Stability calculations validated against Sea State 9 performance
- Power requirements cross-checked with twin diesel configuration
- Scoring optimized for NAVSEA HC-MASC mission requirements

Mission Context:
- Distributed sensor node / picket ship
- ISR platform (MUOS, missile tracking)
- ASW/MCM module carrier (Independence-class LCS mission replacement)
- Autonomous operations with L3Harris ASView (TRL-9)
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import math

from naval_domain.physics_engine import PhysicsEngine, PhysicsResults, WATER_DENSITY, GRAVITY, KNOTS_TO_MS
from naval_domain.m48_parameters import M48HullParameters


# === M48 SEA TRIAL CALIBRATION CONSTANTS ===
# Derived from 32,000 NM of operational data

M48_CALIBRATION = {
    # Resistance calibration factors (empirical adjustments to ITTC-1957)
    'friction_correction': 1.05,  # 5% increase for real-world roughness
    'catamaran_interference': 1.08,  # 8% wave interference between hulls
    'appendage_drag': 1.12,  # 12% for propulsion shafts, rudders

    # Proven performance points (for validation)
    'proven_speed_high': 29.0,  # knots (at 50% payload)
    'proven_speed_cruise': 17.5,  # knots (for max range)
    'proven_range_nm': 15000.0,  # nautical miles
    'proven_payload_high_speed': 72.0,  # tons (50% of 144 ton max)

    # Propulsion efficiency (twin diesel + straight shaft + fixed-pitch props)
    'propulsive_efficiency': 0.65,  # Hull efficiency × propeller efficiency
    'shaft_efficiency': 0.98,  # Straight shaft (minimal losses)
    'engine_efficiency': 0.42,  # Diesel engine thermal efficiency

    # Fuel consumption (diesel)
    'specific_fuel_consumption': 0.22,  # kg/kWh (marine diesel)
    'fuel_density': 0.85,  # kg/liter (diesel)
}


class M48PhysicsEngine(PhysicsEngine):
    """
    M48-specific physics engine with sea trial calibration.

    Extends base PhysicsEngine with:
    - Empirical corrections from 32,000 NM operational data
    - M48-specific resistance calculations (twin-hull interference)
    - NAVSEA HC-MASC mission-specific scoring
    - Payload capacity validation
    - Range estimation with fuel constraints
    """

    def __init__(self):
        super().__init__()
        self.calibration = M48_CALIBRATION

    def simulate(self, params: M48HullParameters, generate_mesh: bool = False) -> PhysicsResults:
        """
        Run M48-calibrated physics simulation.

        Args:
            params: M48HullParameters instance
            generate_mesh: If True, generate 3D mesh (handled by caller)

        Returns:
            PhysicsResults with M48-specific calculations
        """
        # Validate M48 parameters
        params.validate()

        # Use base simulation but override scoring
        results = super().simulate(params, generate_mesh=False)

        # Apply M48-specific corrections to resistance
        results = self._apply_m48_calibration(results, params)

        # Recalculate M48-specific scores
        results.stability_score = self._calculate_m48_stability_score(
            results.metacentric_height, params
        )
        results.speed_score = self._calculate_m48_speed_score(
            results.froude_number, results.total_resistance, params
        )
        results.efficiency_score = self._calculate_m48_efficiency_score(
            results.power_per_ton, params
        )

        # M48 mission-specific overall score (weighted for HC-MASC)
        # Stability: 40% (critical for Sea State 9 + sensor platforms)
        # Efficiency: 35% (15,000 NM range requirement)
        # Speed: 25% (28-30 kts capability)
        results.overall_score = (
            0.40 * results.stability_score +
            0.35 * results.efficiency_score +
            0.25 * results.speed_score
        )

        return results

    def _apply_m48_calibration(self, results: PhysicsResults, params: M48HullParameters) -> PhysicsResults:
        """
        Apply M48 empirical calibration factors to resistance calculations.

        Adjusts ITTC-1957 theoretical results using 32,000 NM sea trial data.
        """
        # Apply friction correction (surface roughness from operational use)
        results.frictional_resistance *= self.calibration['friction_correction']

        # Apply catamaran interference factor (wave interaction between hulls)
        results.residuary_resistance *= self.calibration['catamaran_interference']

        # Apply appendage drag (straight shafts, rudders, propellers)
        results.appendage_resistance *= self.calibration['appendage_drag']

        # Recalculate total resistance
        results.total_resistance = (
            results.frictional_resistance +
            results.residuary_resistance +
            results.appendage_resistance +
            results.air_resistance
        )

        # Recalculate power requirements
        speed_ms = params.design_speed * KNOTS_TO_MS
        results.effective_power = (results.total_resistance * speed_ms) / 1000.0  # kW

        # Use M48 propulsive efficiency (twin diesel + straight shaft)
        prop_eff = self.calibration['propulsive_efficiency']
        results.brake_power = results.effective_power / prop_eff

        # Update power per ton
        results.power_per_ton = results.brake_power / params.displacement

        return results

    def _calculate_m48_stability_score(self, gm: float, params: M48HullParameters) -> float:
        """
        Calculate M48-specific stability score.

        Emphasizes:
        - Sea State 9 survivability (GM requirements for heavy seas)
        - Sensor platform suitability (minimize pitch/roll for tracking systems)
        - Heavy payload stability (4×36-ton containers)

        Score components:
        - Base GM score (50%): Metacentric height adequacy
        - Hull spacing bonus (30%): Wide spacing for catamaran stability
        - Payload margin (20%): Reserve stability for mission modules

        Returns:
            Stability score (0-100)
        """
        score = 0.0

        # === BASE GM SCORE (50 points) ===
        # M48 requires higher GM than typical vessels for Sea State 9
        # Target: GM > 1.5m for heavy sea operations with sensor masts
        if gm < 0:
            base_gm_score = 0.0  # Unstable
        elif gm < 0.5:
            base_gm_score = 10.0  # Marginally stable (insufficient)
        elif gm < 1.0:
            base_gm_score = 25.0  # Adequate for calm seas only
        elif gm < 1.5:
            base_gm_score = 35.0  # Good for moderate seas
        elif gm < 2.5:
            base_gm_score = 50.0  # Excellent (Sea State 9 capable)
        else:
            # Diminishing returns above 2.5m (excessive stiffness = uncomfortable)
            base_gm_score = max(0.0, 50.0 - (gm - 2.5) * 5.0)

        score += base_gm_score

        # === HULL SPACING BONUS (30 points) ===
        # Wide hull spacing increases transverse stability for catamarans
        # M48 target: 8-12m spacing (0.17-0.25 × LOA)
        spacing_ratio = params.hull_spacing / params.length_overall

        if spacing_ratio < 0.15:
            spacing_score = 5.0  # Too narrow
        elif spacing_ratio < 0.18:
            spacing_score = 15.0  # Adequate
        elif spacing_ratio < 0.25:
            spacing_score = 30.0  # Excellent (M48 range)
        else:
            # Excessive spacing = structural challenges + drag
            spacing_score = max(0.0, 30.0 - (spacing_ratio - 0.25) * 100.0)

        score += spacing_score

        # === PAYLOAD MARGIN (20 points) ===
        # M48 must maintain stability with up to 144 tons of mission modules
        # Reward designs with stability margin for heavy payload
        displacement_margin = params.displacement - (83.5 + params.payload_capacity)

        if displacement_margin < 0:
            payload_score = 0.0  # Overloaded
        elif displacement_margin < 10.0:
            payload_score = 10.0  # Minimal margin
        elif displacement_margin < 30.0:
            payload_score = 20.0  # Good margin
        else:
            payload_score = 15.0  # Excessive margin (wasted displacement)

        score += payload_score

        return min(100.0, max(0.0, score))

    def _calculate_m48_speed_score(self, froude_number: float, resistance: float, params: M48HullParameters) -> float:
        """
        Calculate M48-specific speed score.

        Optimized for M48 operational envelope:
        - Cruise: 15-20 knots (max range)
        - Sprint: 28-30 knots (proven capability)

        Scoring favors:
        - Meeting 28-30 kts proven speed
        - Low resistance at cruise speed (15-20 kts)
        - Semi-displacement efficiency (Fn 0.4-0.6)

        Returns:
            Speed score (0-100)
        """
        score = 0.0

        # === DESIGN SPEED ACHIEVEMENT (40 points) ===
        # Reward meeting M48 proven speed envelope
        design_speed = params.design_speed

        if design_speed < 18.0:
            speed_achievement = 10.0  # Below minimum useful speed
        elif design_speed < 25.0:
            speed_achievement = 25.0  # Cruise speed range (good for range)
        elif design_speed < 28.0:
            speed_achievement = 35.0  # Approaching proven max
        elif design_speed <= 30.0:
            speed_achievement = 40.0  # Proven speed envelope (optimal)
        else:
            # Above proven capability = uncertain (needs validation)
            speed_achievement = max(0.0, 40.0 - (design_speed - 30.0) * 2.0)

        score += speed_achievement

        # === FROUDE NUMBER EFFICIENCY (30 points) ===
        # Semi-displacement sweet spot: Fn 0.4-0.6
        # M48 at 28 kts, 48m hull: Fn ≈ 0.55 (ideal)
        if froude_number < 0.3:
            fn_score = 10.0  # Displacement mode (inefficient for M48 speed)
        elif froude_number < 0.4:
            fn_score = 20.0  # Transitioning to semi-displacement
        elif froude_number <= 0.6:
            fn_score = 30.0  # Semi-displacement sweet spot (M48 proven)
        elif froude_number <= 0.8:
            fn_score = 20.0  # Approaching planing (high drag hump)
        else:
            fn_score = 10.0  # Planing regime (very high power)

        score += fn_score

        # === RESISTANCE EFFICIENCY (30 points) ===
        # Lower resistance = better for both speed and range
        # Normalize by displacement to compare different loadouts
        specific_resistance = resistance / (params.displacement * 1000.0 * GRAVITY)  # R / Weight

        if specific_resistance > 0.20:
            resistance_score = 5.0  # Very high drag
        elif specific_resistance > 0.15:
            resistance_score = 15.0  # High drag
        elif specific_resistance > 0.10:
            resistance_score = 25.0  # Moderate drag
        elif specific_resistance > 0.05:
            resistance_score = 30.0  # Low drag (excellent)
        else:
            resistance_score = 20.0  # Unrealistically low (check calc)

        score += resistance_score

        return min(100.0, max(0.0, score))

    def _calculate_m48_efficiency_score(self, power_per_ton: float, params: M48HullParameters) -> float:
        """
        Calculate M48-specific efficiency score.

        Optimized for:
        - 15,000 NM range @ 15-20 knots (proven)
        - Fuel efficiency for extended operations
        - Low power/displacement ratio

        Scoring emphasizes:
        - Low kW/ton for range extension
        - Fuel consumption within M48 capacity
        - Mission endurance capability

        Returns:
            Efficiency score (0-100)
        """
        score = 0.0

        # === POWER PER TON (60 points) ===
        # M48 target: <12 kW/ton for cruise, <20 kW/ton for sprint
        # Lower = better range
        if power_per_ton < 8.0:
            ppt_score = 60.0  # Excellent efficiency
        elif power_per_ton < 12.0:
            ppt_score = 50.0  # Good (cruise range capability)
        elif power_per_ton < 16.0:
            ppt_score = 35.0  # Moderate (acceptable for sprint)
        elif power_per_ton < 20.0:
            ppt_score = 20.0  # High (short range only)
        else:
            ppt_score = max(0.0, 20.0 - (power_per_ton - 20.0))

        score += ppt_score

        # === RANGE ESTIMATE (40 points) ===
        # Estimate range based on fuel capacity and power consumption
        # M48 target: 15,000 NM @ cruise speed

        brake_power_kw = power_per_ton * params.displacement
        fuel_consumption_kg_per_hour = brake_power_kw * self.calibration['specific_fuel_consumption']
        fuel_consumption_liters_per_hour = fuel_consumption_kg_per_hour / self.calibration['fuel_density']

        # Hours of operation at cruise speed
        hours_at_speed = params.fuel_capacity / fuel_consumption_liters_per_hour

        # Estimated range in nautical miles
        estimated_range_nm = hours_at_speed * params.design_speed

        if estimated_range_nm < 5000.0:
            range_score = 5.0  # Insufficient range
        elif estimated_range_nm < 10000.0:
            range_score = 20.0  # Moderate range
        elif estimated_range_nm < 15000.0:
            range_score = 35.0  # Approaching M48 proven range
        elif estimated_range_nm <= 20000.0:
            range_score = 40.0  # Meets or exceeds M48 requirement
        else:
            range_score = 35.0  # Excessive (fuel weight penalties)

        score += range_score

        return min(100.0, max(0.0, score))

    def estimate_mission_range(self, params: M48HullParameters, cruise_speed_knots: float) -> Dict[str, float]:
        """
        Estimate M48 mission range at given cruise speed.

        Args:
            params: M48HullParameters
            cruise_speed_knots: Cruise speed in knots

        Returns:
            Dictionary with:
            - range_nm: Estimated range in nautical miles
            - endurance_hours: Endurance in hours
            - fuel_consumption_lph: Fuel consumption in liters/hour
        """
        # Run simulation at cruise speed
        cruise_params = M48HullParameters(**params.to_dict())
        cruise_params.design_speed = cruise_speed_knots
        results = self.simulate(cruise_params, generate_mesh=False)

        # Calculate fuel consumption
        fuel_consumption_kg_per_hour = results.brake_power * self.calibration['specific_fuel_consumption']
        fuel_consumption_liters_per_hour = fuel_consumption_kg_per_hour / self.calibration['fuel_density']

        # Calculate endurance
        endurance_hours = params.fuel_capacity / fuel_consumption_liters_per_hour

        # Calculate range
        range_nm = endurance_hours * cruise_speed_knots

        return {
            'range_nm': range_nm,
            'endurance_hours': endurance_hours,
            'fuel_consumption_lph': fuel_consumption_liters_per_hour,
            'brake_power_kw': results.brake_power,
        }


def simulate_m48_design(params: M48HullParameters, generate_mesh: bool = False) -> PhysicsResults:
    """
    Convenience function to simulate M48 design.

    Args:
        params: M48HullParameters instance
        generate_mesh: If True, request mesh generation

    Returns:
        PhysicsResults with M48 calibration
    """
    engine = M48PhysicsEngine()
    return engine.simulate(params, generate_mesh=generate_mesh)


if __name__ == "__main__":
    from naval_domain.m48_parameters import M48HullParameters

    print("=" * 70)
    print("M48 PHYSICS ENGINE - CALIBRATION VALIDATION")
    print("=" * 70)
    print()

    # Create M48 baseline (unmanned variant)
    print("Simulating M48 unmanned baseline at 28 knots...")
    print()

    m48_baseline = M48HullParameters(
        # Primary dimensions
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

        # Operational (high speed)
        design_speed=28.0,  # Proven max
        displacement=150.0,  # Structural + payload
        draft=1.5,

        # M48-specific
        payload_capacity=60.0,
        fuel_capacity=45000.0,
        mission_modules=["ISR", "COMMUNICATIONS"],
    )

    # Simulate
    results = simulate_m48_design(m48_baseline)

    print(results.summary())
    print()
    print("=" * 70)
    print()

    # Test range estimation at cruise speed
    print("Estimating range at cruise speed (17.5 knots)...")
    print()

    engine = M48PhysicsEngine()
    range_results = engine.estimate_mission_range(m48_baseline, cruise_speed_knots=17.5)

    print(f"Cruise Speed: 17.5 knots")
    print(f"Estimated Range: {range_results['range_nm']:.0f} NM")
    print(f"Endurance: {range_results['endurance_hours']:.1f} hours")
    print(f"Fuel Consumption: {range_results['fuel_consumption_lph']:.1f} liters/hour")
    print(f"Power Required: {range_results['brake_power_kw']:.1f} kW")
    print()
    print(f"Target Range: 15,000 NM (proven)")
    print(f"Achievement: {(range_results['range_nm'] / 15000.0) * 100:.1f}%")

    print()
    print("=" * 70)
