"""
Physics Engine for Twin-Hull (Catamaran) Naval Vessel Simulation

This module implements CPU-based physics calculations for evaluating hull designs.
All formulas are based on established naval architecture principles and resistance
prediction methods.

Key Calculations:
- Displacement and buoyancy (Archimedes' principle)
- Metacentric height (GM) for stability analysis
- Resistance prediction (ITTC-1957 friction line + residuary resistance)
- Power requirements
- Multi-objective performance scoring

References:
- ITTC-1957 Model-Ship Correlation Line
- Savitsky's planing hull theory (adapted)
- Principles of Naval Architecture (SNAME)
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import math

from hull_parameters import HullParameters


# === PHYSICAL CONSTANTS ===
WATER_DENSITY = 1025.0  # kg/m³ (seawater)
KINEMATIC_VISCOSITY = 1.19e-6  # m²/s (seawater at 15°C)
GRAVITY = 9.81  # m/s²
AIR_DENSITY = 1.225  # kg/m³
KNOTS_TO_MS = 0.51444  # Conversion factor: knots to m/s


@dataclass
class PhysicsResults:
    """
    Results from physics simulation of a hull design.

    Contains all calculated performance metrics and derived properties.
    """

    # === INPUT PARAMETERS (for reference) ===
    hull_params: Dict[str, Any]  # Original hull parameters

    # === HYDROSTATIC PROPERTIES ===
    displacement_volume: float  # m³
    displacement_mass: float    # metric tons
    wetted_surface_area: float  # m² (total for both hulls)
    draft_actual: float        # m (calculated or verified)

    # === STABILITY METRICS ===
    metacentric_height: float  # GM in meters
    transverse_inertia: float  # I_T in m⁴ (second moment of area)
    volumetric_centroid: float # KB in meters (center of buoyancy height)

    # === RESISTANCE COMPONENTS (at design speed) ===
    froude_number: float           # Fn = V / sqrt(g * L)
    reynolds_number: float         # Rn = V * L / ν
    frictional_resistance: float   # N (ITTC-1957)
    residuary_resistance: float    # N (wave + form drag)
    appendage_resistance: float    # N (rudders, propulsion)
    air_resistance: float          # N (aerodynamic)
    total_resistance: float        # N (sum of all components)

    # === POWER REQUIREMENTS ===
    effective_power: float      # PE in kW (resistance × speed)
    brake_power: float          # PB in kW (with propulsive efficiency)
    power_per_ton: float        # kW/ton (efficiency metric)

    # === PERFORMANCE SCORES (0-100) ===
    stability_score: float      # Based on GM and hull spacing
    speed_score: float          # Based on Fn and resistance
    efficiency_score: float     # Based on power/displacement ratio
    overall_score: float        # Weighted composite (35-35-30)

    # === FAILURE FLAGS ===
    is_valid: bool             # Overall validity
    failure_reasons: list      # List of failure reason strings

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        # Convert failure_reasons list to proper format
        if not isinstance(result['failure_reasons'], list):
            result['failure_reasons'] = []
        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== PHYSICS SIMULATION RESULTS ===",
            f"Valid Design: {'✓ YES' if self.is_valid else '✗ NO'}",
        ]

        if not self.is_valid:
            lines.append(f"Failure Reasons: {', '.join(self.failure_reasons)}")

        lines.extend([
            "",
            "=== HYDROSTATICS ===",
            f"Displacement: {self.displacement_mass:.1f} metric tons",
            f"Draft: {self.draft_actual:.2f}m",
            f"Wetted Surface: {self.wetted_surface_area:.1f}m²",
            "",
            "=== STABILITY ===",
            f"GM (Metacentric Height): {self.metacentric_height:.3f}m",
            f"Stability Assessment: {self._assess_stability()}",
            "",
            "=== RESISTANCE (at design speed) ===",
            f"Froude Number: {self.froude_number:.3f}",
            f"Reynolds Number: {self.reynolds_number:.2e}",
            f"Total Resistance: {self.total_resistance:.1f} N",
            f"  - Friction: {self.frictional_resistance:.1f} N ({self.frictional_resistance/self.total_resistance*100:.1f}%)",
            f"  - Residuary: {self.residuary_resistance:.1f} N ({self.residuary_resistance/self.total_resistance*100:.1f}%)",
            "",
            "=== POWER ===",
            f"Effective Power: {self.effective_power:.1f} kW",
            f"Brake Power (est): {self.brake_power:.1f} kW",
            f"Specific Power: {self.power_per_ton:.2f} kW/ton",
            "",
            "=== PERFORMANCE SCORES ===",
            f"Stability Score: {self.stability_score:.1f}/100",
            f"Speed Score: {self.speed_score:.1f}/100",
            f"Efficiency Score: {self.efficiency_score:.1f}/100",
            f"Overall Score: {self.overall_score:.1f}/100",
        ])

        return "\n".join(lines)

    def _assess_stability(self) -> str:
        """Assess stability based on GM value."""
        if self.metacentric_height < 0:
            return "UNSTABLE (negative GM)"
        elif self.metacentric_height < 0.35:
            return "Poor (GM < 0.35m)"
        elif self.metacentric_height < 1.0:
            return "Adequate (0.35m < GM < 1.0m)"
        elif self.metacentric_height < 2.0:
            return "Good (1.0m < GM < 2.0m)"
        else:
            return "Excellent (GM > 2.0m)"


class PhysicsEngine:
    """
    CPU-based physics engine for hull design evaluation.

    Implements naval architecture calculations using classical formulas
    and empirical correlations. Suitable for sequential evaluation of designs.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize physics engine.

        Args:
            verbose: If True, print detailed calculation steps
        """
        self.verbose = verbose

    def simulate(self, hull_params: HullParameters) -> PhysicsResults:
        """
        Run complete physics simulation for a hull design.

        Args:
            hull_params: HullParameters object defining the design

        Returns:
            PhysicsResults: Complete simulation results with scores

        Raises:
            ValueError: If hull_params validation fails
        """
        # Ensure parameters are valid
        hull_params.validate()

        # Track failures
        failures = []
        is_valid = True

        # Calculate draft if not provided
        draft = self._calculate_draft(hull_params)

        # === HYDROSTATIC CALCULATIONS ===
        displacement_volume = self._calculate_displacement_volume(hull_params, draft)
        displacement_mass = displacement_volume * WATER_DENSITY / 1000.0  # Convert to tons

        # Check if calculated displacement matches specified displacement
        displacement_error = abs(displacement_mass - hull_params.displacement) / hull_params.displacement
        if displacement_error > 0.3:  # 30% tolerance
            failures.append(f"Displacement mismatch: calculated {displacement_mass:.1f}t vs specified {hull_params.displacement:.1f}t")
            is_valid = False

        wetted_surface = self._calculate_wetted_surface(hull_params, draft)

        # === STABILITY CALCULATIONS ===
        kb = self._calculate_kb(hull_params, draft)  # Center of buoyancy height
        it = self._calculate_transverse_inertia(hull_params, draft)  # Second moment of waterline area
        gm = self._calculate_metacentric_height(hull_params, draft, displacement_volume, it, kb)

        if gm < 0:
            failures.append(f"Negative GM ({gm:.3f}m) - unstable")
            is_valid = False
        elif gm < 0.15:
            failures.append(f"Critically low GM ({gm:.3f}m)")

        # === RESISTANCE CALCULATIONS ===
        speed_ms = hull_params.design_speed * KNOTS_TO_MS

        # Froude number
        fn = self._calculate_froude_number(speed_ms, hull_params.length_overall)

        # Reynolds number
        rn = self._calculate_reynolds_number(speed_ms, hull_params.length_overall)

        # Resistance components
        rf = self._calculate_frictional_resistance(speed_ms, wetted_surface, hull_params.length_overall)
        rr = self._calculate_residuary_resistance(speed_ms, hull_params, displacement_volume, fn)
        ra = self._calculate_appendage_resistance(rf)  # Assume 10% of friction
        r_air = self._calculate_air_resistance(speed_ms, hull_params)
        r_total = rf + rr + ra + r_air

        # === POWER CALCULATIONS ===
        pe = r_total * speed_ms / 1000.0  # kW
        pb = pe / 0.65  # Assume 65% propulsive efficiency (propeller + hull)
        power_per_ton = pb / displacement_mass

        # === PERFORMANCE SCORING ===
        stability_score = self._calculate_stability_score(gm, hull_params)
        speed_score = self._calculate_speed_score(fn, hull_params)
        efficiency_score = self._calculate_efficiency_score(power_per_ton, displacement_mass)

        # Weighted overall score: 35% stability, 35% speed, 30% efficiency
        overall_score = (0.35 * stability_score + 0.35 * speed_score + 0.30 * efficiency_score)

        return PhysicsResults(
            hull_params=hull_params.to_dict(),
            displacement_volume=displacement_volume,
            displacement_mass=displacement_mass,
            wetted_surface_area=wetted_surface,
            draft_actual=draft,
            metacentric_height=gm,
            transverse_inertia=it,
            volumetric_centroid=kb,
            froude_number=fn,
            reynolds_number=rn,
            frictional_resistance=rf,
            residuary_resistance=rr,
            appendage_resistance=ra,
            air_resistance=r_air,
            total_resistance=r_total,
            effective_power=pe,
            brake_power=pb,
            power_per_ton=power_per_ton,
            stability_score=stability_score,
            speed_score=speed_score,
            efficiency_score=efficiency_score,
            overall_score=overall_score,
            is_valid=is_valid,
            failure_reasons=failures,
        )

    # ========================================================================
    # HYDROSTATIC CALCULATIONS
    # ========================================================================

    def _calculate_draft(self, hull_params: HullParameters) -> float:
        """
        Calculate or verify draft.

        For twin-hull vessels, uses simplified volume approximation.
        If draft is specified in hull_params, use it; otherwise estimate.

        Args:
            hull_params: Hull parameters

        Returns:
            Draft in meters
        """
        if hull_params.draft is not None:
            return hull_params.draft

        # Estimate draft from displacement and hull dimensions
        # Simplified: assume each hull is a box-like shape
        # Volume = 2 × L × B × T × Cb
        # T = Displacement / (2 × ρ × L × B × Cb)

        estimated_draft = (hull_params.displacement * 1000.0) / (
            2.0 * WATER_DENSITY * hull_params.length_overall *
            hull_params.beam * hull_params.block_coefficient
        )

        return min(estimated_draft, hull_params.hull_depth * 0.9)  # Can't exceed 90% of depth

    def _calculate_displacement_volume(self, hull_params: HullParameters, draft: float) -> float:
        """
        Calculate displaced volume using block coefficient approximation.

        For twin-hull:
        Volume = 2 × L × B × T × Cb

        Where Cb is the block coefficient (ratio of actual volume to box volume).

        Args:
            hull_params: Hull parameters
            draft: Draft in meters

        Returns:
            Displacement volume in m³
        """
        volume_per_hull = (
            hull_params.length_overall *
            hull_params.beam *
            draft *
            hull_params.block_coefficient
        )

        total_volume = 2.0 * volume_per_hull

        if self.verbose:
            print(f"Displacement volume: {total_volume:.2f} m³ (2 hulls × {volume_per_hull:.2f} m³)")

        return total_volume

    def _calculate_wetted_surface(self, hull_params: HullParameters, draft: float) -> float:
        """
        Calculate total wetted surface area for both hulls.

        Uses simplified formula:
        S = 2 × [L × (2T + B) + 2 × B × T]  (for two rectangular hulls approximation)

        More accurate would use:
        S = C × sqrt(L × B × T × displacement_volume)
        Where C ≈ 2.5 for catamarans

        Args:
            hull_params: Hull parameters
            draft: Draft in meters

        Returns:
            Wetted surface area in m²
        """
        # Use Denny-Mumford formula adapted for catamarans
        # S = C × (displacement_volume)^0.5 × (L)^0.5
        # For catamarans: C ≈ 3.5-4.5

        displacement_vol = self._calculate_displacement_volume(hull_params, draft)

        # Use relationship: S ≈ 3.8 × sqrt(∇ × L)
        wetted_surface = 3.8 * math.sqrt(displacement_vol * hull_params.length_overall)

        if self.verbose:
            print(f"Wetted surface area: {wetted_surface:.2f} m²")

        return wetted_surface

    # ========================================================================
    # STABILITY CALCULATIONS
    # ========================================================================

    def _calculate_kb(self, hull_params: HullParameters, draft: float) -> float:
        """
        Calculate KB (vertical center of buoyancy above keel).

        For simplified box-like hulls:
        KB ≈ T / 2  (centroid at half-draft)

        More accurate formula accounts for hull shape via prismatic coefficient.

        Args:
            hull_params: Hull parameters
            draft: Draft in meters

        Returns:
            KB in meters
        """
        # For simplified calculation: KB at approximately half draft
        # Adjusted by block coefficient (fuller hulls have lower KB)
        kb = draft * (0.45 + 0.05 * hull_params.block_coefficient)

        return kb

    def _calculate_transverse_inertia(self, hull_params: HullParameters, draft: float) -> float:
        """
        Calculate I_T (transverse second moment of waterplane area).

        For twin hulls with rectangular waterplane approximation:
        For each hull: I_hull = (B_wl³ × L) / 12
        Parallel axis theorem for spacing: I_total = 2 × [I_hull + A × d²]

        Where:
        - B_wl = waterline beam
        - L = length overall
        - A = waterline area per hull
        - d = distance from centerline = hull_spacing / 2

        Args:
            hull_params: Hull parameters
            draft: Draft in meters

        Returns:
            I_T in m⁴
        """
        # Waterline area per hull (use prismatic coefficient for more accuracy)
        a_wl_per_hull = hull_params.length_overall * hull_params.waterline_beam * hull_params.prismatic_coefficient

        # Second moment about own centerline (per hull)
        i_own = (hull_params.waterline_beam ** 3 * hull_params.length_overall * hull_params.prismatic_coefficient) / 12.0

        # Distance from vessel centerline to hull centerline
        d = hull_params.hull_spacing / 2.0

        # Parallel axis theorem: I_total = 2 × (I_own + A × d²)
        i_t = 2.0 * (i_own + a_wl_per_hull * d**2)

        if self.verbose:
            print(f"Transverse inertia I_T: {i_t:.2f} m⁴")

        return i_t

    def _calculate_metacentric_height(
        self,
        hull_params: HullParameters,
        draft: float,
        displacement_volume: float,
        i_t: float,
        kb: float
    ) -> float:
        """
        Calculate GM (metacentric height) for transverse stability.

        Formula:
        GM = KB + BM - KG

        Where:
        - KB = height of center of buoyancy
        - BM = metacentric radius = I_T / ∇
        - KG = height of center of gravity (estimated)

        For catamarans, wide hull spacing significantly increases BM.

        Args:
            hull_params: Hull parameters
            draft: Draft in meters
            displacement_volume: Displaced volume in m³
            i_t: Transverse second moment in m⁴
            kb: Center of buoyancy height in m

        Returns:
            GM in meters
        """
        # Calculate BM (metacentric radius)
        bm = i_t / displacement_volume

        # Estimate KG (center of gravity height)
        # Assume KG at approximately 60% of hull depth (conservative)
        kg = hull_params.hull_depth * 0.60

        # GM = KB + BM - KG
        gm = kb + bm - kg

        if self.verbose:
            print(f"Stability calculation:")
            print(f"  KB (center of buoyancy): {kb:.3f}m")
            print(f"  BM (metacentric radius): {bm:.3f}m")
            print(f"  KG (center of gravity, est): {kg:.3f}m")
            print(f"  GM (metacentric height): {gm:.3f}m")

        return gm

    # ========================================================================
    # RESISTANCE CALCULATIONS
    # ========================================================================

    def _calculate_froude_number(self, speed_ms: float, length: float) -> float:
        """
        Calculate Froude number.

        Fn = V / sqrt(g × L)

        Critical regions:
        - Fn < 0.4: Displacement mode
        - 0.4 < Fn < 1.0: Transition/semi-planing
        - Fn > 1.0: Planing mode

        Args:
            speed_ms: Speed in m/s
            length: Length in meters

        Returns:
            Froude number (dimensionless)
        """
        fn = speed_ms / math.sqrt(GRAVITY * length)
        return fn

    def _calculate_reynolds_number(self, speed_ms: float, length: float) -> float:
        """
        Calculate Reynolds number.

        Rn = V × L / ν

        Where ν is kinematic viscosity.

        Args:
            speed_ms: Speed in m/s
            length: Length in meters

        Returns:
            Reynolds number (dimensionless)
        """
        rn = speed_ms * length / KINEMATIC_VISCOSITY
        return rn

    def _calculate_frictional_resistance(
        self,
        speed_ms: float,
        wetted_surface: float,
        length: float
    ) -> float:
        """
        Calculate frictional resistance using ITTC-1957 correlation line.

        Formula:
        Rf = 0.5 × ρ × V² × S × Cf

        Where:
        Cf = 0.075 / (log₁₀(Rn) - 2)²  (ITTC-1957)

        Args:
            speed_ms: Speed in m/s
            wetted_surface: Wetted surface area in m²
            length: Length in meters

        Returns:
            Frictional resistance in Newtons
        """
        rn = self._calculate_reynolds_number(speed_ms, length)

        # ITTC-1957 friction coefficient
        cf = 0.075 / (math.log10(rn) - 2.0) ** 2

        # Form factor (accounts for 3D flow effects) - assume 1.15 for catamarans
        form_factor = 1.15

        # Frictional resistance
        rf = 0.5 * WATER_DENSITY * speed_ms**2 * wetted_surface * cf * form_factor

        if self.verbose:
            print(f"Frictional resistance: {rf:.1f} N (Cf={cf:.5f}, Rn={rn:.2e})")

        return rf

    def _calculate_residuary_resistance(
        self,
        speed_ms: float,
        hull_params: HullParameters,
        displacement_volume: float,
        fn: float
    ) -> float:
        """
        Calculate residuary resistance (wave-making + form drag).

        Uses empirical correlation for catamarans:
        Rr = ρ × g × ∇ × Cr(Fn)

        Where Cr is residuary resistance coefficient (function of Froude number).

        For catamarans, wave resistance increases rapidly above Fn ≈ 0.4
        due to wave interference between hulls.

        Args:
            speed_ms: Speed in m/s
            hull_params: Hull parameters
            displacement_volume: Displaced volume in m³
            fn: Froude number

        Returns:
            Residuary resistance in Newtons
        """
        # Residuary resistance coefficient (empirical for catamarans)
        # Based on Insel-Molland resistance prediction method

        # Base coefficient increases with Froude number
        if fn < 0.35:
            cr_base = 0.0001 * fn**3
        elif fn < 0.50:
            cr_base = 0.002 * (fn - 0.35)**2 + 0.0001 * 0.35**3
        else:
            cr_base = 0.005 * (fn - 0.50)**3 + 0.002 * 0.15**2 + 0.0001 * 0.35**3

        # Hull spacing factor: closely spaced hulls have higher wave interference
        spacing_ratio = hull_params.hull_spacing / hull_params.length_overall
        if spacing_ratio < 0.25:
            spacing_factor = 1.4  # High interference
        elif spacing_ratio < 0.35:
            spacing_factor = 1.2
        else:
            spacing_factor = 1.0  # Minimal interference

        # Prismatic coefficient factor (fuller hulls have higher wave resistance)
        cp_factor = 1.0 + (hull_params.prismatic_coefficient - 0.60) * 0.5

        # Combined coefficient
        cr = cr_base * spacing_factor * cp_factor

        # Residuary resistance
        rr = WATER_DENSITY * GRAVITY * displacement_volume * cr

        if self.verbose:
            print(f"Residuary resistance: {rr:.1f} N (Cr={cr:.6f}, Fn={fn:.3f})")

        return rr

    def _calculate_appendage_resistance(self, frictional_resistance: float) -> float:
        """
        Estimate appendage resistance (rudders, shafts, etc.).

        Typically 5-15% of frictional resistance.

        Args:
            frictional_resistance: Frictional resistance in N

        Returns:
            Appendage resistance in Newtons
        """
        # Conservative estimate: 10% of friction
        return 0.10 * frictional_resistance

    def _calculate_air_resistance(self, speed_ms: float, hull_params: HullParameters) -> float:
        """
        Calculate aerodynamic drag.

        Formula:
        R_air = 0.5 × ρ_air × V² × A_frontal × Cd

        Args:
            speed_ms: Speed in m/s
            hull_params: Hull parameters

        Returns:
            Air resistance in Newtons
        """
        # Estimate frontal area (very rough approximation)
        # Assume rectangular superstructure: width ≈ hull_spacing, height ≈ freeboard
        a_frontal = hull_params.hull_spacing * hull_params.freeboard * 0.7

        # Drag coefficient for boxy superstructure
        cd_air = 0.8

        r_air = 0.5 * AIR_DENSITY * speed_ms**2 * a_frontal * cd_air

        return r_air

    # ========================================================================
    # PERFORMANCE SCORING
    # ========================================================================

    def _calculate_stability_score(self, gm: float, hull_params: HullParameters) -> float:
        """
        Score stability performance (0-100).

        Considers:
        - GM value (higher is better, but too high causes harsh motion)
        - Hull spacing (wider is more stable)

        Optimal GM: 1.0-2.5m for passenger comfort

        Args:
            gm: Metacentric height in meters
            hull_params: Hull parameters

        Returns:
            Stability score (0-100)
        """
        # GM scoring curve
        if gm < 0:
            gm_score = 0.0  # Unstable
        elif gm < 0.35:
            gm_score = 30.0 * (gm / 0.35)  # Poor stability
        elif gm < 1.0:
            gm_score = 30.0 + 40.0 * ((gm - 0.35) / 0.65)  # Improving
        elif gm < 2.5:
            gm_score = 70.0 + 30.0 * ((gm - 1.0) / 1.5)  # Excellent
        elif gm < 5.0:
            # Slight penalty for very high GM (can cause stiff motion)
            gm_score = 100.0 - 10.0 * ((gm - 2.5) / 2.5)
        else:
            # Moderate penalty for excessive GM (very stiff, but still stable)
            gm_score = max(70.0, 90.0 - 5.0 * (gm - 5.0))

        # Hull spacing bonus
        spacing_ratio = hull_params.hull_spacing / hull_params.length_overall
        if spacing_ratio > 0.35:
            spacing_bonus = 10.0
        elif spacing_ratio > 0.25:
            spacing_bonus = 5.0
        else:
            spacing_bonus = 0.0

        total_score = min(100.0, max(0.0, gm_score + spacing_bonus))

        return total_score

    def _calculate_speed_score(self, fn: float, hull_params: HullParameters) -> float:
        """
        Score speed performance (0-100).

        Higher Froude numbers indicate higher speed-to-length ratios.
        Catamarans can operate efficiently at higher Fn than monohulls.

        Froude number regimes:
        - Fn < 0.4: Displacement mode (efficient)
        - 0.4 < Fn < 1.0: Transition/semi-planing
        - Fn > 1.0: Planing mode (high power)

        Args:
            fn: Froude number
            hull_params: Hull parameters

        Returns:
            Speed score (0-100)
        """
        # Score based on achievable Froude number
        # Catamarans can operate at higher Fn than monohulls

        if fn < 0.25:
            # Very low speed - displacement mode
            score = 50.0 + 50.0 * (fn / 0.25)
        elif fn < 0.50:
            # Efficient displacement mode
            score = 100.0
        elif fn < 0.80:
            # Transition region - acceptable but less efficient
            score = 100.0 - 20.0 * ((fn - 0.50) / 0.30)
        elif fn < 1.2:
            # Semi-planing - high power but achievable
            score = 80.0 - 40.0 * ((fn - 0.80) / 0.40)
        else:
            # Planing/very high speed - extreme power requirements
            score = 40.0 - 30.0 * min((fn - 1.2) / 0.5, 1.0)

        return max(0.0, score)

    def _calculate_efficiency_score(self, power_per_ton: float, displacement: float) -> float:
        """
        Score power efficiency (0-100).

        Lower power-per-ton is better (more efficient).

        Typical values:
        - Displacement mode: 3-8 kW/ton
        - Semi-planing: 8-15 kW/ton
        - Planing: 15-30 kW/ton

        Args:
            power_per_ton: Brake power per displacement ton
            displacement: Displacement in metric tons

        Returns:
            Efficiency score (0-100)
        """
        # Efficiency curve (lower power/ton is better)
        if power_per_ton < 5.0:
            score = 100.0  # Excellent efficiency
        elif power_per_ton < 10.0:
            score = 100.0 - 20.0 * ((power_per_ton - 5.0) / 5.0)
        elif power_per_ton < 20.0:
            score = 80.0 - 40.0 * ((power_per_ton - 10.0) / 10.0)
        else:
            score = 40.0 - 40.0 * min((power_per_ton - 20.0) / 20.0, 1.0)

        # Bonus for larger vessels (economies of scale)
        if displacement > 50.0:
            score += 5.0
        elif displacement > 100.0:
            score += 10.0

        return min(100.0, max(0.0, score))


# === CONVENIENCE FUNCTION ===

def simulate_design(hull_params: HullParameters, verbose: bool = False) -> PhysicsResults:
    """
    Convenience function to simulate a single design.

    Args:
        hull_params: Hull parameters to simulate
        verbose: Enable verbose output

    Returns:
        PhysicsResults: Simulation results
    """
    engine = PhysicsEngine(verbose=verbose)
    return engine.simulate(hull_params)


if __name__ == "__main__":
    # Demonstrate physics engine with baseline catamaran
    from hull_parameters import get_baseline_catamaran, get_high_speed_catamaran

    print("=" * 70)
    print("PHYSICS ENGINE DEMONSTRATION")
    print("=" * 70)
    print()

    # Test baseline catamaran
    baseline = get_baseline_catamaran()
    print("Simulating BASELINE CATAMARAN...")
    print()
    results_baseline = simulate_design(baseline, verbose=True)
    print()
    print(results_baseline.summary())
    print()
    print("=" * 70)
    print()

    # Test high-speed catamaran
    high_speed = get_high_speed_catamaran()
    print("Simulating HIGH-SPEED CATAMARAN...")
    print()
    results_high_speed = simulate_design(high_speed, verbose=False)
    print(results_high_speed.summary())
