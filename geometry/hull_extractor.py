"""
Hull Extractor for MAGNET Naval Design System
==============================================

Extracts hull parameters from M48 baseline data and provides geometry
utilities for the MAGNET autonomous design system.

Author: Agent 2 (CAD/Physics Lead)
Date: 2025-11-22
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HullGeometry:
    """Hull geometry parameters extracted from baseline data"""
    LOA: float  # Length overall (m)
    LWL: float  # Length waterline (m)
    Beam: float  # Beam per hull (m)
    Draft: float  # Draft (m)
    Depth: float  # Hull depth (m)
    Hull_Spacing: float  # Center-to-center hull spacing (m)
    Displacement: float  # Displacement (tonnes)

    # Form coefficients
    Cb: float  # Block coefficient
    Cp: float  # Prismatic coefficient
    Cm: float  # Midship coefficient
    Cwp: float  # Waterplane coefficient

    # Hydrostatics
    LCB: float  # Longitudinal center of buoyancy (m from aft)
    LCF: float  # Longitudinal center of flotation (m from aft)
    KB: float  # Vertical center of buoyancy (m above keel)
    BM: float  # Metacentric radius (m)
    GM: float  # Metacentric height (m)

    # Areas
    waterplane_area: float  # Waterplane area (m²)
    wetted_surface: float  # Wetted surface area (m²)


class HullExtractor:
    """
    Extract and process hull parameters from M48 baseline data.

    Provides utilities for:
    - Loading baseline design data
    - Extracting principal dimensions
    - Calculating hydrostatic properties
    - Generating station positions
    - Computing form coefficients
    """

    def __init__(self, baseline_file: Optional[str] = None):
        """
        Initialize hull extractor.

        Args:
            baseline_file: Path to M48 baseline JSON file.
                          If None, uses default location.
        """
        if baseline_file is None:
            baseline_file = "data/baselines/m48_baseline.json"

        self.baseline_file = Path(baseline_file)
        self.baseline_data: Optional[Dict[str, Any]] = None
        self.hull_geometry: Optional[HullGeometry] = None

    def load(self) -> 'HullExtractor':
        """
        Load M48 baseline data from JSON file.

        Returns:
            self (for method chaining)

        Raises:
            FileNotFoundError: If baseline file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        if not self.baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {self.baseline_file}")

        with open(self.baseline_file, 'r') as f:
            self.baseline_data = json.load(f)

        print(f"✓ Loaded baseline: {self.baseline_data.get('vessel_name', 'Unknown')}")
        print(f"  Data source: {self.baseline_data.get('data_source', 'Unknown')}")
        print(f"  Confidence: {self.baseline_data.get('confidence', 0.0):.2f}")

        return self

    def extract_hull_geometry(self) -> HullGeometry:
        """
        Extract hull geometry from baseline data.

        Returns:
            HullGeometry object with all parameters

        Raises:
            ValueError: If baseline data not loaded
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not loaded. Call load() first.")

        dims = self.baseline_data['principal_dimensions']
        hydro = self.baseline_data['hydrostatics']

        self.hull_geometry = HullGeometry(
            LOA=dims['LOA_m'],
            LWL=dims['LWL_m'],
            Beam=dims['Beam_m'],
            Draft=dims['Draft_m'],
            Depth=dims['Depth_m'],
            Hull_Spacing=dims['Hull_Spacing_m'],
            Displacement=hydro['displacement_tonnes'],
            Cb=hydro['Cb'],
            Cp=hydro['Cp'],
            Cm=hydro['Cm'],
            Cwp=hydro['Cwp'],
            LCB=hydro['LCB_m'],
            LCF=hydro['LCF_m'],
            KB=hydro['KB_m'],
            BM=hydro['BM_m'],
            GM=hydro['GM_m'],
            waterplane_area=hydro['waterplane_area_m2'],
            wetted_surface=hydro['wetted_surface_m2']
        )

        return self.hull_geometry

    def extract_dimensions(self) -> Dict[str, float]:
        """
        Extract principal dimensions as dictionary.

        Returns:
            Dictionary of principal dimensions
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not loaded. Call load() first.")

        dims = self.baseline_data['principal_dimensions']
        hydro = self.baseline_data['hydrostatics']

        return {
            'LOA_m': dims['LOA_m'],
            'LWL_m': dims['LWL_m'],
            'Beam_m': dims['Beam_m'],
            'Draft_m': dims['Draft_m'],
            'Depth_m': dims['Depth_m'],
            'Hull_Spacing_m': dims['Hull_Spacing_m'],
            'Displacement_tonnes': hydro['displacement_tonnes'],
            'Cb': hydro['Cb'],
            'Cp': hydro['Cp'],
            'GM_m': hydro['GM_m'],
            'wetted_surface_m2': hydro['wetted_surface_m2']
        }

    def get_stations(self, num_stations: int = 11) -> List[float]:
        """
        Generate station positions along hull length.

        Standard naval architecture uses 10 stations (0-10) along LWL,
        giving 11 positions including bow and stern.

        Args:
            num_stations: Number of station positions (default: 11)

        Returns:
            List of station x-positions from aft (m)
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not loaded. Call load() first.")

        lwl = self.baseline_data['principal_dimensions']['LWL_m']

        # Generate evenly spaced stations from stern (0) to bow (LWL)
        stations = np.linspace(0, lwl, num_stations)

        return stations.tolist()

    def get_performance_data(self) -> Dict[str, Any]:
        """
        Extract performance characteristics.

        Returns:
            Dictionary of performance parameters
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not loaded. Call load() first.")

        perf = self.baseline_data['performance']

        return {
            'max_speed_knots': perf['max_speed_knots'],
            'cruise_speed_knots': perf['cruise_speed_knots'],
            'design_speed_knots': perf['design_speed_knots'],
            'range_nm': perf['range_nm'],
            'fuel_capacity_liters': perf['fuel_capacity_liters']
        }

    def get_resistance_calibration(self) -> Dict[str, float]:
        """
        Extract resistance calibration factors from sea trial data.

        Returns:
            Dictionary of calibration factors
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not loaded. Call load() first.")

        return self.baseline_data['resistance_calibration']

    def compute_volume_displacement(self) -> float:
        """
        Calculate volumetric displacement (m³).

        Returns:
            Volumetric displacement in cubic meters
        """
        if self.hull_geometry is None:
            self.extract_hull_geometry()

        # Displacement (tonnes) / seawater density (1.025 t/m³)
        seawater_density = 1.025  # tonnes/m³

        return self.hull_geometry.Displacement / seawater_density

    def estimate_demihull_beam_ratio(self) -> float:
        """
        Estimate demihull beam/draft ratio (important for resistance).

        Returns:
            Beam/draft ratio
        """
        if self.hull_geometry is None:
            self.extract_hull_geometry()

        return self.hull_geometry.Beam / self.hull_geometry.Draft

    def estimate_hull_slenderness(self) -> float:
        """
        Estimate hull slenderness ratio (LWL / Beam).

        For catamarans, use demihull beam, not overall beam.

        Returns:
            Slenderness ratio
        """
        if self.hull_geometry is None:
            self.extract_hull_geometry()

        return self.hull_geometry.LWL / self.hull_geometry.Beam

    def get_full_baseline_dict(self) -> Dict[str, Any]:
        """
        Get complete baseline data dictionary.

        Returns:
            Full baseline data
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not loaded. Call load() first.")

        return self.baseline_data

    def print_summary(self):
        """Print formatted summary of hull parameters."""
        if self.hull_geometry is None:
            self.extract_hull_geometry()

        geom = self.hull_geometry

        print("\n" + "=" * 80)
        print(f"M48 HULL GEOMETRY SUMMARY")
        print("=" * 80)
        print(f"\nVessel: {self.baseline_data.get('vessel_name', 'Unknown')}")
        print(f"Type: {self.baseline_data['principal_dimensions'].get('Hull_Type', 'Unknown')}")
        print(f"\nPRINCIPAL DIMENSIONS:")
        print(f"  LOA:          {geom.LOA:.2f} m")
        print(f"  LWL:          {geom.LWL:.2f} m")
        print(f"  Beam (each):  {geom.Beam:.2f} m")
        print(f"  Hull Spacing: {geom.Hull_Spacing:.2f} m")
        print(f"  Draft:        {geom.Draft:.2f} m")
        print(f"  Depth:        {geom.Depth:.2f} m")
        print(f"  Displacement: {geom.Displacement:.1f} tonnes")
        print(f"\nFORM COEFFICIENTS:")
        print(f"  Cb (Block):     {geom.Cb:.3f}")
        print(f"  Cp (Prismatic): {geom.Cp:.3f}")
        print(f"  Cm (Midship):   {geom.Cm:.3f}")
        print(f"  Cwp (WP):       {geom.Cwp:.3f}")
        print(f"\nHYDROSTATICS:")
        print(f"  LCB: {geom.LCB:.2f} m from aft")
        print(f"  LCF: {geom.LCF:.2f} m from aft")
        print(f"  KB:  {geom.KB:.2f} m above keel")
        print(f"  GM:  {geom.GM:.2f} m (metacentric height)")
        print(f"\nAREAS:")
        print(f"  Waterplane: {geom.waterplane_area:.1f} m²")
        print(f"  Wetted:     {geom.wetted_surface:.1f} m²")
        print(f"\nRATIOS:")
        print(f"  Slenderness (L/B): {self.estimate_hull_slenderness():.2f}")
        print(f"  B/T ratio:         {self.estimate_demihull_beam_ratio():.2f}")
        print(f"  Volume disp.:      {self.compute_volume_displacement():.2f} m³")
        print("=" * 80 + "\n")


def load_m48_baseline() -> Dict[str, Any]:
    """
    Convenience function to load M48 baseline data.

    Returns:
        M48 baseline data dictionary
    """
    extractor = HullExtractor()
    extractor.load()
    return extractor.get_full_baseline_dict()


def get_m48_geometry() -> HullGeometry:
    """
    Convenience function to get M48 hull geometry.

    Returns:
        HullGeometry object
    """
    extractor = HullExtractor()
    extractor.load()
    return extractor.extract_hull_geometry()


# ============================================================================
# Test/Demo Code
# ============================================================================

if __name__ == "__main__":
    print("M48 Hull Extractor - Test Run")
    print("=" * 80)
    print()

    # Create extractor and load data
    extractor = HullExtractor()

    try:
        extractor.load()
        print("✓ Successfully loaded M48 baseline data\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Please ensure data/baselines/m48_baseline.json exists")
        exit(1)

    # Extract geometry
    hull_geom = extractor.extract_hull_geometry()
    print("✓ Extracted hull geometry\n")

    # Print detailed summary
    extractor.print_summary()

    # Get stations
    stations = extractor.get_stations(num_stations=11)
    print(f"Station positions (11 stations from stern to bow):")
    for i, x in enumerate(stations):
        print(f"  Station {i}: {x:.2f} m")
    print()

    # Get performance data
    perf = extractor.get_performance_data()
    print("Performance Data:")
    for key, value in perf.items():
        print(f"  {key}: {value}")
    print()

    # Get resistance calibration
    res_cal = extractor.get_resistance_calibration()
    print("Resistance Calibration Factors:")
    for key, value in res_cal.items():
        print(f"  {key}: {value}")
    print()

    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
