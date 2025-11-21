"""
Comprehensive test suite for naval physics engine.

Tests all physics calculations against known values and validates
that results are physically reasonable.
"""

import pytest
import sys
import os

# Add naval_domain to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../naval_domain'))

from hull_parameters import (
    HullParameters,
    get_baseline_catamaran,
    get_high_speed_catamaran,
    get_stability_optimized_catamaran
)
from physics_engine import PhysicsEngine, simulate_design
from hull_generator import generate_hull_metadata


class TestPhysicsEngine:
    """Test suite for PhysicsEngine class."""

    def test_baseline_catamaran_simulation(self):
        """Test that baseline catamaran produces valid results."""
        baseline = get_baseline_catamaran()
        results = simulate_design(baseline)

        # Check that simulation completed
        assert results is not None
        assert results.is_valid == True

        # Check hydrostatic properties
        assert results.displacement_mass > 0
        assert results.draft_actual > 0
        assert results.wetted_surface_area > 0

        # Check that draft is reasonable
        assert 0.3 <= results.draft_actual <= 2.0

        # Check stability
        assert results.metacentric_height > 0  # GM should be positive
        assert results.transverse_inertia > 0

        # Check resistance components
        assert results.frictional_resistance > 0
        assert results.total_resistance >= results.frictional_resistance

        # Check power
        assert results.effective_power > 0
        assert results.brake_power > results.effective_power  # Should include losses

        # Check scores
        assert 0 <= results.stability_score <= 100
        assert 0 <= results.speed_score <= 100
        assert 0 <= results.efficiency_score <= 100
        assert 0 <= results.overall_score <= 100

    def test_high_speed_catamaran_simulation(self):
        """Test high-speed catamaran variant."""
        high_speed = get_high_speed_catamaran()
        results = simulate_design(high_speed)

        assert results is not None

        # Should have higher Froude number than baseline
        assert results.froude_number > 0.8

        # Power should be higher for high speed
        assert results.power_per_ton > 20.0  # kW/ton

    def test_stability_optimized_catamaran(self):
        """Test stability-optimized variant."""
        stability_cat = get_stability_optimized_catamaran()
        results = simulate_design(stability_cat)

        assert results is not None
        assert results.is_valid == True

        # Should have good GM
        assert results.metacentric_height > 0.5

        # Stability score should be reasonable
        assert results.stability_score > 50

    def test_displacement_calculation(self):
        """Test displacement volume calculation."""
        engine = PhysicsEngine()
        baseline = get_baseline_catamaran()

        # Calculate displacement
        draft = baseline.draft if baseline.draft else 1.0
        volume = engine._calculate_displacement_volume(baseline, draft)

        # Volume should be positive
        assert volume > 0

        # Volume should be reasonable for an 18m catamaran
        assert 10.0 <= volume <= 100.0  # m³

    def test_wetted_surface_calculation(self):
        """Test wetted surface area calculation."""
        engine = PhysicsEngine()
        baseline = get_baseline_catamaran()

        draft = baseline.draft if baseline.draft else 1.0
        wetted_surface = engine._calculate_wetted_surface(baseline, draft)

        # Should be positive
        assert wetted_surface > 0

        # Should be reasonable for 18m catamaran (typically 50-150 m²)
        assert 30.0 <= wetted_surface <= 200.0

    def test_froude_number_calculation(self):
        """Test Froude number calculation."""
        engine = PhysicsEngine()

        # Test known values
        # Example: 10 knots (5.144 m/s) on 100m ship
        # Fn = V / sqrt(g*L) = 5.144 / sqrt(9.81 * 100) = 0.164
        fn = engine._calculate_froude_number(5.144, 100.0)
        assert abs(fn - 0.164) < 0.01

        # Test 25 knots on 18m ship
        speed_ms = 25.0 * 0.51444  # Convert knots to m/s
        fn = engine._calculate_froude_number(speed_ms, 18.0)

        # Should be in semi-planing regime
        assert 0.5 <= fn <= 1.5

    def test_reynolds_number_calculation(self):
        """Test Reynolds number calculation."""
        engine = PhysicsEngine()

        # Test that Reynolds number increases with speed and length
        rn1 = engine._calculate_reynolds_number(5.0, 10.0)
        rn2 = engine._calculate_reynolds_number(10.0, 10.0)
        rn3 = engine._calculate_reynolds_number(5.0, 20.0)

        assert rn2 > rn1  # Double speed, double Rn
        assert rn3 > rn1  # Double length, double Rn

        # Typical range for ships: 1e7 to 1e9
        assert 1e6 <= rn1 <= 1e10

    def test_metacentric_height_calculation(self):
        """Test GM (metacentric height) calculation."""
        engine = PhysicsEngine()
        baseline = get_baseline_catamaran()

        draft = baseline.draft if baseline.draft else 1.0
        volume = engine._calculate_displacement_volume(baseline, draft)
        it = engine._calculate_transverse_inertia(baseline, draft)
        kb = engine._calculate_kb(baseline, draft)

        gm = engine._calculate_metacentric_height(baseline, draft, volume, it, kb)

        # GM should be positive for stable vessel
        assert gm > 0

        # For catamarans, GM is typically large due to wide spacing
        # Should be in range 0.5-20m
        assert 0.1 <= gm <= 30.0

    def test_frictional_resistance_increases_with_speed(self):
        """Test that frictional resistance increases with speed squared."""
        engine = PhysicsEngine()
        baseline = get_baseline_catamaran()

        draft = baseline.draft if baseline.draft else 1.0
        wetted_surface = engine._calculate_wetted_surface(baseline, draft)

        # Calculate friction at different speeds
        rf_10 = engine._calculate_frictional_resistance(5.0, wetted_surface, baseline.length_overall)
        rf_20 = engine._calculate_frictional_resistance(10.0, wetted_surface, baseline.length_overall)

        # Friction should increase (approximately quadratically with speed)
        assert rf_20 > rf_10
        # Rough check: doubling speed should roughly quadruple resistance
        assert rf_20 > 3.0 * rf_10

    def test_stability_score_positive_gm(self):
        """Test stability scoring for positive GM."""
        engine = PhysicsEngine()
        baseline = get_baseline_catamaran()

        # Test various GM values
        score_low = engine._calculate_stability_score(0.3, baseline)
        score_good = engine._calculate_stability_score(1.5, baseline)
        score_high = engine._calculate_stability_score(10.0, baseline)

        # All should be positive
        assert score_low >= 0
        assert score_good >= 0
        assert score_high >= 0

        # Good GM should score higher than low
        assert score_good > score_low

    def test_stability_score_negative_gm(self):
        """Test stability scoring for unstable vessel (negative GM)."""
        engine = PhysicsEngine()
        baseline = get_baseline_catamaran()

        # Negative GM = unstable
        score = engine._calculate_stability_score(-0.5, baseline)

        # Should be zero or very low (may get spacing bonus, so allow up to 10)
        assert score <= 10.0

    def test_speed_score_varies_with_froude_number(self):
        """Test speed scoring across Froude number ranges."""
        engine = PhysicsEngine()
        baseline = get_baseline_catamaran()

        # Test different speed regimes
        score_low = engine._calculate_speed_score(0.20, baseline)
        score_optimal = engine._calculate_speed_score(0.45, baseline)
        score_high = engine._calculate_speed_score(1.5, baseline)

        # All should be in valid range
        assert 0 <= score_low <= 100
        assert 0 <= score_optimal <= 100
        assert 0 <= score_high <= 100

        # Optimal should score highest
        assert score_optimal >= score_low
        assert score_optimal >= score_high

    def test_efficiency_score_varies_with_power(self):
        """Test efficiency scoring."""
        engine = PhysicsEngine()

        # Test different power levels
        score_efficient = engine._calculate_efficiency_score(5.0, 50.0)
        score_moderate = engine._calculate_efficiency_score(15.0, 50.0)
        score_inefficient = engine._calculate_efficiency_score(30.0, 50.0)

        # All should be valid
        assert 0 <= score_efficient <= 100
        assert 0 <= score_moderate <= 100
        assert 0 <= score_inefficient <= 100

        # Lower power/ton should score higher
        assert score_efficient > score_moderate > score_inefficient

    def test_parameter_validation_catches_invalid_designs(self):
        """Test that invalid parameters are caught."""
        with pytest.raises(ValueError):
            # Negative length should fail
            HullParameters(
                length_overall=-10.0,
                beam=2.0,
                hull_depth=2.0,
                hull_spacing=5.0,
                deadrise_angle=12.0,
                freeboard=1.0,
                lcb_position=50.0,
                prismatic_coefficient=0.60,
                waterline_beam=1.8,
                block_coefficient=0.42,
                design_speed=25.0,
                displacement=35.0,
            )

    def test_multiple_designs_consistency(self):
        """Test that same design produces consistent results."""
        baseline = get_baseline_catamaran()

        # Run simulation twice
        results1 = simulate_design(baseline)
        results2 = simulate_design(baseline)

        # Results should be identical
        assert results1.overall_score == results2.overall_score
        assert results1.displacement_mass == results2.displacement_mass
        assert results1.total_resistance == results2.total_resistance

    def test_results_to_dict(self):
        """Test that results can be serialized to dictionary."""
        baseline = get_baseline_catamaran()
        results = simulate_design(baseline)

        # Convert to dict
        results_dict = results.to_dict()

        # Should have all required keys
        assert 'displacement_mass' in results_dict
        assert 'total_resistance' in results_dict
        assert 'overall_score' in results_dict
        assert 'is_valid' in results_dict
        assert 'failure_reasons' in results_dict

        # Should be JSON-serializable
        import json
        json_str = json.dumps(results_dict)
        assert len(json_str) > 0

    def test_simulation_handles_draft_none(self):
        """Test that simulation works when draft is not specified."""
        # Create parameters without draft
        params = HullParameters(
            length_overall=20.0,
            beam=2.0,
            hull_depth=2.5,
            hull_spacing=6.0,
            deadrise_angle=10.0,
            freeboard=1.3,
            lcb_position=50.0,
            prismatic_coefficient=0.65,
            waterline_beam=1.8,
            block_coefficient=0.40,
            design_speed=30.0,
            displacement=40.0,
            draft=None,  # Not specified
        )

        # Should still simulate successfully
        results = simulate_design(params)

        assert results is not None
        assert results.draft_actual > 0

    def test_extreme_designs(self):
        """Test physics engine with extreme but valid designs."""

        # Very long, slender hull (within valid ratios)
        slender = HullParameters(
            length_overall=35.0,
            beam=2.5,
            hull_depth=2.0,
            hull_spacing=9.0,
            deadrise_angle=5.0,
            freeboard=1.2,
            lcb_position=50.0,
            prismatic_coefficient=0.70,
            waterline_beam=2.2,
            block_coefficient=0.38,
            design_speed=38.0,
            displacement=70.0,
            draft=0.8,
        )

        results_slender = simulate_design(slender)
        assert results_slender is not None

        # Short, beamy hull
        beamy = HullParameters(
            length_overall=12.0,
            beam=1.8,
            hull_depth=1.5,
            hull_spacing=3.5,
            deadrise_angle=20.0,
            freeboard=0.8,
            lcb_position=48.0,
            prismatic_coefficient=0.55,
            waterline_beam=1.6,
            block_coefficient=0.48,
            design_speed=15.0,
            displacement=18.0,
            draft=0.7,
        )

        results_beamy = simulate_design(beamy)
        assert results_beamy is not None

        # Slender should have higher L/B ratio
        assert slender.length_overall / slender.beam > beamy.length_overall / beamy.beam


class TestPhysicsResults:
    """Test suite for PhysicsResults dataclass."""

    def test_results_summary_formatting(self):
        """Test that summary produces readable output."""
        baseline = get_baseline_catamaran()
        results = simulate_design(baseline)

        summary = results.summary()

        # Should be a multi-line string
        assert len(summary) > 100
        assert '\n' in summary

        # Should contain key information
        assert 'PHYSICS SIMULATION RESULTS' in summary
        assert 'Displacement' in summary
        assert 'GM' in summary
        assert 'PERFORMANCE SCORES' in summary


class TestIntegration:
    """Integration tests combining physics engine with hull generator."""

    def test_hull_generator_physics_integration(self):
        """Test that hull generator output matches physics calculations."""
        baseline = get_baseline_catamaran()

        # Generate hull metadata
        hull_data = generate_hull_metadata(baseline)

        # Run physics simulation
        physics_results = simulate_design(baseline)

        # Volume should match (within tolerance)
        hull_volume = hull_data['volume_properties']['total_volume']
        physics_volume = physics_results.displacement_volume

        assert abs(hull_volume - physics_volume) < 1.0  # Within 1 m³

        # Displacement mass should match
        hull_mass = hull_data['volume_properties']['displacement_mass']
        physics_mass = physics_results.displacement_mass

        assert abs(hull_mass - physics_mass) < 2.0  # Within 2 tons

    def test_all_baseline_configurations(self):
        """Test all predefined baseline configurations."""
        configs = [
            get_baseline_catamaran(),
            get_high_speed_catamaran(),
            get_stability_optimized_catamaran(),
        ]

        for config in configs:
            results = simulate_design(config)

            # All should produce results
            assert results is not None

            # All should have positive scores
            assert results.overall_score > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
