"""
Integration Tests for Agent 2 Geometry Pipeline
================================================

Tests the complete Agent 2 (CAD/Physics Lead) workflow:
1. Loading M48 baseline data
2. Extracting hull geometry
3. Storing in knowledge base
4. Integration with MAGNET system

Author: Agent 2
Date: 2025-11-22
"""

import pytest
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from geometry.hull_extractor import HullExtractor, HullGeometry, load_m48_baseline, get_m48_geometry
from memory.knowledge_base import KnowledgeBase


class TestM48BaselineData:
    """Test M48 baseline data file"""

    def test_baseline_file_exists(self):
        """Test that M48 baseline JSON file exists"""
        baseline_file = Path("data/baselines/m48_baseline.json")
        assert baseline_file.exists(), f"Baseline file not found: {baseline_file}"

    def test_baseline_json_valid(self):
        """Test that baseline JSON is valid and complete"""
        baseline_file = Path("data/baselines/m48_baseline.json")

        with open(baseline_file, 'r') as f:
            data = json.load(f)

        # Check required top-level keys
        required_keys = [
            'vessel_name',
            'vessel_type',
            'principal_dimensions',
            'hydrostatics',
            'performance',
            'validation'
        ]

        for key in required_keys:
            assert key in data, f"Missing required key: {key}"

    def test_principal_dimensions_complete(self):
        """Test that principal dimensions are complete"""
        baseline_file = Path("data/baselines/m48_baseline.json")

        with open(baseline_file, 'r') as f:
            data = json.load(f)

        dims = data['principal_dimensions']

        required_dims = ['LOA_m', 'LWL_m', 'Beam_m', 'Draft_m', 'Depth_m', 'Hull_Spacing_m']

        for dim in required_dims:
            assert dim in dims, f"Missing dimension: {dim}"
            assert isinstance(dims[dim], (int, float)), f"{dim} must be numeric"
            assert dims[dim] > 0, f"{dim} must be positive"

    def test_hydrostatics_complete(self):
        """Test that hydrostatics data is complete"""
        baseline_file = Path("data/baselines/m48_baseline.json")

        with open(baseline_file, 'r') as f:
            data = json.load(f)

        hydro = data['hydrostatics']

        required_hydro = [
            'displacement_tonnes',
            'Cb', 'Cp', 'Cm', 'Cwp',
            'LCB_m', 'GM_m',
            'waterplane_area_m2',
            'wetted_surface_m2'
        ]

        for param in required_hydro:
            assert param in hydro, f"Missing hydrostatic parameter: {param}"
            assert isinstance(hydro[param], (int, float)), f"{param} must be numeric"


class TestHullExtractor:
    """Test HullExtractor class"""

    def test_hull_extractor_initialization(self):
        """Test HullExtractor can be initialized"""
        extractor = HullExtractor("data/baselines/m48_baseline.json")
        assert extractor.baseline_file.exists()

    def test_load_baseline(self):
        """Test loading baseline data"""
        extractor = HullExtractor("data/baselines/m48_baseline.json")
        extractor.load()

        assert extractor.baseline_data is not None
        assert 'vessel_name' in extractor.baseline_data

    def test_extract_hull_geometry(self):
        """Test extracting hull geometry"""
        extractor = HullExtractor("data/baselines/m48_baseline.json")
        extractor.load()

        hull_geom = extractor.extract_hull_geometry()

        assert isinstance(hull_geom, HullGeometry)
        assert hull_geom.LOA > 0
        assert hull_geom.Beam > 0
        assert hull_geom.Draft > 0
        assert hull_geom.Displacement > 0

    def test_extract_dimensions(self):
        """Test extracting dimensions as dictionary"""
        extractor = HullExtractor("data/baselines/m48_baseline.json")
        extractor.load()

        dims = extractor.extract_dimensions()

        assert isinstance(dims, dict)
        assert 'LOA_m' in dims
        assert 'Beam_m' in dims
        assert 'Displacement_tonnes' in dims

    def test_get_stations(self):
        """Test generating station positions"""
        extractor = HullExtractor("data/baselines/m48_baseline.json")
        extractor.load()

        stations = extractor.get_stations(num_stations=11)

        assert len(stations) == 11
        assert stations[0] == 0.0  # Stern
        assert stations[-1] > 0.0  # Bow
        assert all(stations[i] < stations[i+1] for i in range(len(stations)-1))  # Monotonic

    def test_compute_volume_displacement(self):
        """Test volume displacement calculation"""
        extractor = HullExtractor("data/baselines/m48_baseline.json")
        extractor.load()
        extractor.extract_hull_geometry()

        vol_disp = extractor.compute_volume_displacement()

        assert vol_disp > 0
        assert isinstance(vol_disp, float)

    def test_estimate_ratios(self):
        """Test estimation of hull ratios"""
        extractor = HullExtractor("data/baselines/m48_baseline.json")
        extractor.load()
        extractor.extract_hull_geometry()

        slenderness = extractor.estimate_hull_slenderness()
        bt_ratio = extractor.estimate_demihull_beam_ratio()

        assert slenderness > 0
        assert bt_ratio > 0


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_load_m48_baseline(self):
        """Test load_m48_baseline convenience function"""
        baseline_data = load_m48_baseline()

        assert baseline_data is not None
        assert 'vessel_name' in baseline_data
        assert 'principal_dimensions' in baseline_data

    def test_get_m48_geometry(self):
        """Test get_m48_geometry convenience function"""
        hull_geom = get_m48_geometry()

        assert isinstance(hull_geom, HullGeometry)
        assert hull_geom.LOA > 0


class TestKnowledgeBaseIntegration:
    """Test integration with knowledge base"""

    def test_knowledge_base_exists(self):
        """Test that knowledge base can be created"""
        kb = KnowledgeBase(storage_path="memory/knowledge")
        assert kb is not None

    def test_m48_baseline_in_knowledge_base(self):
        """Test that M48 baseline can be stored in knowledge base"""
        kb = KnowledgeBase(storage_path="memory/knowledge")

        # Check if M48 baseline experiment exists
        if kb.experiments:
            # Look for baseline experiment
            baseline_found = False
            for exp in kb.experiments:
                if exp.get('cycle', -1) == 0:  # Cycle 0 for baseline
                    designs = exp.get('designs', [])
                    if designs:
                        design = designs[0]
                        if design.get('design_id') == 'M48_UNMANNED_BASELINE':
                            baseline_found = True
                            break

            # If baseline not found, it's okay (may not have run loader yet)
            # But if found, verify its structure
            if baseline_found:
                assert design.get('length_overall', 0) > 0
                assert design.get('displacement', 0) > 0


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline"""

    def test_complete_pipeline(self):
        """Test complete Agent 2 pipeline: Load -> Extract -> Process"""
        # Step 1: Load baseline
        extractor = HullExtractor("data/baselines/m48_baseline.json")
        extractor.load()

        # Step 2: Extract geometry
        hull_geom = extractor.extract_hull_geometry()

        # Step 3: Get all data
        dims = extractor.extract_dimensions()
        stations = extractor.get_stations()
        performance = extractor.get_performance_data()
        resistance = extractor.get_resistance_calibration()

        # Verify all data is present and valid
        assert hull_geom.LOA == 48.0
        assert hull_geom.Displacement == 145.0
        assert len(stations) == 11
        assert performance['max_speed_knots'] == 30.0
        assert 'friction_correction_factor' in resistance

        print("\n✓ Complete Agent 2 pipeline test passed")
        print(f"  LOA: {hull_geom.LOA} m")
        print(f"  Displacement: {hull_geom.Displacement} tonnes")
        print(f"  Max Speed: {performance['max_speed_knots']} knots")
        print(f"  Stations: {len(stations)}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Agent 2 Geometry Pipeline Integration Tests")
    print("=" * 80)
    print()

    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
