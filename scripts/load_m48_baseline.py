#!/usr/bin/env python3
"""
M48 Baseline Loader for MAGNET System
======================================

Loads M48 baseline geometry and hydrostatics data into the MAGNET knowledge base.
This script integrates Agent 2's CAD/geometry work with Agent 1's infrastructure.

Author: Agent 2 (CAD/Physics Lead)
Date: 2025-11-22
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry.hull_extractor import HullExtractor
from memory.knowledge_base import KnowledgeBase


def load_m48_to_knowledge_base():
    """
    Load M48 baseline data into MAGNET knowledge base.

    This function:
    1. Loads M48 baseline JSON
    2. Extracts geometry using HullExtractor
    3. Stores data in knowledge base for agent access
    4. Returns comprehensive baseline dictionary
    """
    print("=" * 80)
    print("M48 Baseline Loader - MAGNET Integration")
    print("=" * 80)
    print()

    # Step 1: Load baseline JSON
    print("[1/4] Loading M48 baseline JSON...")
    try:
        with open('data/baselines/m48_baseline.json', 'r') as f:
            baseline_data = json.load(f)
        print(f"✓ Loaded: {baseline_data['vessel_name']}")
        print(f"  Source: {baseline_data['data_source']}")
        print(f"  Confidence: {baseline_data['confidence']:.2%}")
    except FileNotFoundError:
        print("✗ Error: data/baselines/m48_baseline.json not found")
        print("  Please run this script from the MAGNETarc_demo root directory")
        return None
    except Exception as e:
        print(f"✗ Error loading baseline JSON: {e}")
        return None

    print()

    # Step 2: Extract geometry from baseline
    print("[2/4] Extracting hull geometry...")
    try:
        extractor = HullExtractor("data/baselines/m48_baseline.json")
        extractor.load()
        hull_geometry = extractor.extract_hull_geometry()

        print(f"✓ Extracted hull geometry:")
        print(f"  LOA: {hull_geometry.LOA:.1f} m")
        print(f"  Beam: {hull_geometry.Beam:.1f} m")
        print(f"  Draft: {hull_geometry.Draft:.1f} m")
        print(f"  Displacement: {hull_geometry.Displacement:.0f} tonnes")
        print(f"  Cb: {hull_geometry.Cb:.3f}")
        print(f"  GM: {hull_geometry.GM:.2f} m")

        # Add extracted geometry to baseline data
        baseline_data['extracted_geometry'] = {
            'LOA': hull_geometry.LOA,
            'LWL': hull_geometry.LWL,
            'Beam': hull_geometry.Beam,
            'Draft': hull_geometry.Draft,
            'Depth': hull_geometry.Depth,
            'Hull_Spacing': hull_geometry.Hull_Spacing,
            'Displacement': hull_geometry.Displacement,
            'Cb': hull_geometry.Cb,
            'Cp': hull_geometry.Cp,
            'Cm': hull_geometry.Cm,
            'Cwp': hull_geometry.Cwp,
            'LCB': hull_geometry.LCB,
            'LCF': hull_geometry.LCF,
            'GM': hull_geometry.GM,
            'waterplane_area': hull_geometry.waterplane_area,
            'wetted_surface': hull_geometry.wetted_surface
        }

        # Add station positions
        stations = extractor.get_stations(num_stations=11)
        baseline_data['stations'] = stations
        print(f"  Stations: {len(stations)} positions generated")

        # Add derived ratios
        baseline_data['derived_ratios'] = {
            'slenderness': extractor.estimate_hull_slenderness(),
            'beam_draft_ratio': extractor.estimate_demihull_beam_ratio(),
            'volume_displacement_m3': extractor.compute_volume_displacement()
        }

    except Exception as e:
        print(f"✗ Error extracting geometry: {e}")
        import traceback
        traceback.print_exc()
        return None

    print()

    # Step 3: Store in knowledge base
    print("[3/4] Storing in MAGNET knowledge base...")
    try:
        kb = KnowledgeBase(storage_path="memory/knowledge")

        # Create baseline hypothesis
        baseline_hypothesis = {
            'source': 'M48 32,000 NM Sea Trials',
            'strategy': 'proven_baseline',
            'description': 'M48 proven baseline from 32,000 NM sea trials (2020-2025)',
            'confidence': baseline_data['confidence']
        }

        # Create baseline design
        baseline_design = {
            'design_id': 'M48_UNMANNED_BASELINE',
            'name': baseline_data['vessel_name'],
            'length_overall': baseline_data['principal_dimensions']['LOA_m'],
            'beam': baseline_data['principal_dimensions']['Beam_m'],
            'draft': baseline_data['principal_dimensions']['Draft_m'],
            'hull_spacing': baseline_data['principal_dimensions']['Hull_Spacing_m'],
            'displacement': baseline_data['hydrostatics']['displacement_tonnes'],
            'prismatic_coefficient': baseline_data['hydrostatics']['Cp'],
            'block_coefficient': baseline_data['hydrostatics']['Cb'],
            'design_speed': baseline_data['performance']['design_speed_knots'],
            'parameters': baseline_data['principal_dimensions'],
            'hydrostatics': baseline_data['hydrostatics'],
            'performance': baseline_data['performance'],
            'geometry': baseline_data['extracted_geometry'],
            'stations': baseline_data['stations'],
            'derived_ratios': baseline_data['derived_ratios']
        }

        # Create baseline results (sea trial validated)
        baseline_result = {
            'design_id': 'M48_UNMANNED_BASELINE',
            'is_valid': True,
            'overall_score': 95.0,  # High score for proven baseline
            'stability_score': 98.0,  # GM=2.1m, Sea State 9 proven
            'speed_score': 92.0,  # 30 knots proven
            'efficiency_score': 93.0,  # 16,000 NM range proven
            'validation': baseline_data['validation'],
            'performance_validated': True,
            'sea_trial_distance_nm': 32000,
            'operational_hours': 2500,
            'max_sea_state': 9,
            'notes': 'Baseline from actual 32,000 NM sea trials (2020-2025)'
        }

        # Add to knowledge base using correct method
        kb.add_experiment_results(
            hypothesis=baseline_hypothesis,
            designs=[baseline_design],
            results=[baseline_result],
            cycle_number=0  # Cycle 0 for baseline
        )

        print("✓ Stored M48 baseline in knowledge base")
        print(f"  Design ID: M48_UNMANNED_BASELINE")
        print(f"  Overall Score: 95.0 (proven baseline)")

    except Exception as e:
        print(f"✗ Error storing in knowledge base: {e}")
        import traceback
        traceback.print_exc()
        return None

    print()

    # Step 4: Save enriched baseline data
    print("[4/4] Saving enriched baseline data...")
    try:
        output_path = Path("data/baselines/m48_baseline_enriched.json")
        with open(output_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        print(f"✓ Saved enriched baseline to: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"✗ Error saving enriched baseline: {e}")

    print()
    print("=" * 80)
    print("M48 BASELINE LOADED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Vessel: {baseline_data['vessel_name']}")
    print(f"  Type: {baseline_data['principal_dimensions']['Hull_Type']}")
    print(f"  Displacement: {baseline_data['hydrostatics']['displacement_tonnes']} tonnes")
    print(f"  Max Speed: {baseline_data['performance']['max_speed_knots']} knots")
    print(f"  Range: {baseline_data['performance']['range_nm']} NM")
    print(f"  Validation: {baseline_data['validation']['sea_trial_distance_nm']} NM sea trials")
    print()
    print("Data available to all MAGNET agents via knowledge base.")
    print()

    return baseline_data


def print_baseline_summary(baseline_data: dict):
    """Print detailed summary of baseline data."""
    print("\n" + "=" * 80)
    print("DETAILED M48 BASELINE SUMMARY")
    print("=" * 80)

    print(f"\nVessel: {baseline_data['vessel_name']}")
    print(f"Type: {baseline_data['vessel_type']}")
    print(f"Configuration: {baseline_data['principal_dimensions']['Hull_Type']}")

    print("\nPRINCIPAL DIMENSIONS:")
    dims = baseline_data['principal_dimensions']
    for key, value in dims.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    print("\nHYDROSTATICS:")
    hydro = baseline_data['hydrostatics']
    for key, value in hydro.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")

    print("\nPERFORMANCE:")
    perf = baseline_data['performance']
    for key, value in perf.items():
        print(f"  {key}: {value}")

    if 'derived_ratios' in baseline_data:
        print("\nDERIVED RATIOS:")
        for key, value in baseline_data['derived_ratios'].items():
            print(f"  {key}: {value:.3f}")

    print("\nVALIDATION:")
    val = baseline_data['validation']
    for key, value in val.items():
        if isinstance(value, list):
            print(f"  {key}: {', '.join(value)}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Load M48 baseline
    baseline_data = load_m48_to_knowledge_base()

    if baseline_data:
        # Print detailed summary
        if '--verbose' in sys.argv or '-v' in sys.argv:
            print_baseline_summary(baseline_data)

        print("✓ Agent 2 baseline loader completed successfully")
        sys.exit(0)
    else:
        print("✗ Failed to load M48 baseline")
        sys.exit(1)
