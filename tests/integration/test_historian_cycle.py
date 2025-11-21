#!/usr/bin/env python3
"""
Integration Tests for Historian Naval Agent
===========================================

Tests the Historian agent's ability to compress history and extract patterns.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Direct imports to avoid ARC conflicts
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
base_naval = load_module("base_naval_agent", os.path.join(base_path, "agents/base_naval_agent.py"))
historian = load_module("historian_naval_agent", os.path.join(base_path, "agents/historian_naval_agent.py"))
local_client = load_module("local_client", os.path.join(base_path, "llm/local_client.py"))


def create_mock_experiments(num_experiments=50, cycle_range=(1, 10)):
    """Create mock experiments for testing"""
    import random
    experiments = []

    for i in range(num_experiments):
        cycle = random.randint(*cycle_range)
        is_valid = random.random() > 0.2  # 80% success rate

        exp = {
            "design_id": f"exp_{i:03d}",
            "cycle_number": cycle,
            "parameters": {
                "length_overall": 14.0 + random.random() * 8.0,
                "beam": 4.0 + random.random() * 4.0,
                "hull_spacing": 3.5 + random.random() * 2.5,
                "hull_depth": 2.0 + random.random() * 1.5,
                "deadrise_angle": 8.0 + random.random() * 10.0,
                "freeboard": 1.0 + random.random() * 1.0
            },
            "results": {
                "is_valid": is_valid
            }
        }

        if is_valid:
            exp["results"].update({
                "overall_score": 60.0 + random.random() * 30.0,
                "stability_score": 65.0 + random.random() * 25.0,
                "speed_score": 60.0 + random.random() * 30.0,
                "efficiency_score": 55.0 + random.random() * 35.0
            })
        else:
            exp["results"]["failure_reasons"] = ["stability_failure"]

        experiments.append(exp)

    return experiments


def test_historian_compresses_history():
    """Test Historian compresses large histories"""
    print("\n" + "="*70)
    print("TEST 1: Historian Compresses History")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="historian_001",
        role="historian",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    historian_agent = historian.HistorianNavalAgent(config, client)

    # Create 100 experiments to compress
    all_experiments = create_mock_experiments(100, cycle_range=(1, 20))

    # Compress
    compressed = historian_agent.compress_history(all_experiments, cycle_number=20)

    print(f"✓ Original experiments: {compressed['total_experiments']}")
    print(f"✓ Kept experiments: {compressed['kept_experiments']}")
    print(f"✓ Compression ratio: {compressed['compression_ratio']:.2f}")
    print(f"✓ Summary stats: {list(compressed['summary_statistics'].keys())}")

    assert compressed["kept_experiments"] < compressed["total_experiments"]
    assert compressed["compression_ratio"] < 1.0
    assert "summary_statistics" in compressed

    return True


def test_historian_identifies_patterns():
    """Test Historian identifies parameter patterns"""
    print("\n" + "="*70)
    print("TEST 2: Historian Identifies Patterns")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="historian_001",
        role="historian",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    historian_agent = historian.HistorianNavalAgent(config, client)

    # Create experiments with clear pattern: longer hull = better score
    experiments = []
    for i in range(20):
        loa = 15.0 + i * 0.3  # Increasing length
        score = 60.0 + i * 1.5  # Increasing score (strong correlation)

        experiments.append({
            "design_id": f"pattern_exp_{i}",
            "cycle_number": i // 5,
            "parameters": {
                "length_overall": loa,
                "beam": 6.0,
                "hull_spacing": 4.5,
                "hull_depth": 2.5,
                "deadrise_angle": 12.0,
                "freeboard": 1.5,
                "lcb_position": 0.50,
                "prismatic_coefficient": 0.62
            },
            "results": {
                "is_valid": True,
                "overall_score": score,
                "stability_score": score - 5.0,
                "speed_score": score + 5.0,
                "efficiency_score": score
            }
        })

    # Identify patterns
    patterns = historian_agent.identify_patterns(experiments)

    print(f"✓ Patterns identified: {len(patterns)}")
    for i, pattern in enumerate(patterns[:5], 1):
        print(f"  {i}. {pattern}")

    assert len(patterns) > 0, "Should identify at least one pattern"
    # Should find that length_overall correlates with performance
    length_patterns = [p for p in patterns if "length_overall" in p]
    assert len(length_patterns) > 0, "Should identify length_overall pattern"

    return True


def test_historian_infers_constraints():
    """Test Historian infers constraints from failures"""
    print("\n" + "="*70)
    print("TEST 3: Historian Infers Constraints")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="historian_001",
        role="historian",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    historian_agent = historian.HistorianNavalAgent(config, client)

    # Create experiments with failure pattern: small beam causes failures
    experiments = []

    # Valid experiments with normal beam
    for i in range(10):
        experiments.append({
            "design_id": f"valid_{i}",
            "parameters": {
                "length_overall": 18.0,
                "beam": 6.0 + i * 0.1,  # Normal beam
                "hull_spacing": 4.5,
                "hull_depth": 2.5
            },
            "results": {
                "is_valid": True,
                "overall_score": 70.0
            }
        })

    # Failed experiments with small beam
    for i in range(10):
        experiments.append({
            "design_id": f"failed_{i}",
            "parameters": {
                "length_overall": 18.0,
                "beam": 3.5 + i * 0.05,  # Small beam = failures
                "hull_spacing": 4.5,
                "hull_depth": 2.5
            },
            "results": {
                "is_valid": False,
                "failure_reasons": ["stability_failure"]
            }
        })

    # Infer constraints
    constraints = historian_agent.infer_constraints(experiments)

    print(f"✓ Constraints inferred: {len(constraints)}")
    for i, constraint in enumerate(constraints, 1):
        print(f"  {i}. {constraint}")

    assert len(constraints) > 0, "Should infer at least one constraint"

    return True


def test_historian_tracks_trends():
    """Test Historian tracks performance trends"""
    print("\n" + "="*70)
    print("TEST 4: Historian Tracks Performance Trends")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="historian_001",
        role="historian",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    historian_agent = historian.HistorianNavalAgent(config, client)

    # Create experiments showing improvement over time
    experiments = []
    for cycle in range(10):
        for i in range(5):
            score = 60.0 + cycle * 2.0  # Improving 2 points per cycle
            experiments.append({
                "design_id": f"cycle{cycle}_exp{i}",
                "cycle_number": cycle,
                "parameters": {"length_overall": 18.0, "beam": 6.0},
                "results": {
                    "is_valid": True,
                    "overall_score": score + i * 0.5
                }
            })

    # Track trends
    trends = historian_agent.track_performance_trends(experiments, {})

    print(f"✓ Improvement rate: {trends['improvement_rate']:.2f} points/cycle")
    print(f"✓ Best score: {trends['best_so_far']['score']:.1f}")
    print(f"✓ Stagnation detected: {trends['stagnation_detected']}")
    print(f"✓ Total valid experiments: {trends['total_valid_experiments']}")

    assert trends["improvement_rate"] > 0, "Should detect positive improvement"
    assert not trends["stagnation_detected"], "Should not detect stagnation with steady improvement"

    return True


def test_historian_detects_stagnation():
    """Test Historian detects stagnation"""
    print("\n" + "="*70)
    print("TEST 5: Historian Detects Stagnation")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="historian_001",
        role="historian",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    historian_agent = historian.HistorianNavalAgent(config, client)

    # Create experiments with stagnation (no improvement for last 5 cycles)
    experiments = []

    # Early improvement
    for cycle in range(5):
        for i in range(3):
            score = 60.0 + cycle * 3.0
            experiments.append({
                "design_id": f"early_cycle{cycle}_exp{i}",
                "cycle_number": cycle,
                "parameters": {"length_overall": 18.0},
                "results": {
                    "is_valid": True,
                    "overall_score": score
                }
            })

    # Stagnation (5+ cycles with no improvement)
    for cycle in range(5, 11):
        for i in range(3):
            score = 75.0  # Flat score
            experiments.append({
                "design_id": f"stagnant_cycle{cycle}_exp{i}",
                "cycle_number": cycle,
                "parameters": {"length_overall": 18.0},
                "results": {
                    "is_valid": True,
                    "overall_score": score
                }
            })

    # Track trends
    trends = historian_agent.track_performance_trends(experiments, {})

    print(f"✓ Improvement rate: {trends['improvement_rate']:.2f} points/cycle")
    print(f"✓ Stagnation detected: {trends['stagnation_detected']}")
    print(f"✓ Cycles without improvement: {trends['cycles_without_improvement']}")

    assert trends["stagnation_detected"], "Should detect stagnation"
    assert trends["cycles_without_improvement"] > 0

    return True


def test_historian_full_cycle():
    """Test full Historian analysis cycle"""
    print("\n" + "="*70)
    print("TEST 6: Full Historian Cycle")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="historian_001",
        role="historian",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    historian_agent = historian.HistorianNavalAgent(config, client)

    # Create context
    new_results = create_mock_experiments(10, cycle_range=(10, 10))
    current_history = {"experiments": create_mock_experiments(40, cycle_range=(1, 9))}

    context = {
        "new_results": new_results,
        "current_history": current_history,
        "knowledge_base": {},
        "cycle_number": 10
    }

    # Run cycle
    response = historian_agent.autonomous_cycle(context)

    assert response.agent_id == "historian_001"
    assert response.action == "update_history"
    assert "compressed_history" in response.data
    assert "new_patterns" in response.data
    assert "inferred_constraints" in response.data
    assert "performance_trends" in response.data

    analysis = response.data
    print(f"✓ Summary: {analysis.get('summary', 'N/A')[:80]}...")
    print(f"✓ Patterns: {len(analysis['new_patterns'])}")
    print(f"✓ Constraints: {len(analysis['inferred_constraints'])}")
    print(f"✓ Compressed: {analysis['compressed_history']['kept_experiments']} experiments")

    return True


def main():
    """Run all Historian tests"""
    tests = [
        ("Historian Compresses History", test_historian_compresses_history),
        ("Historian Identifies Patterns", test_historian_identifies_patterns),
        ("Historian Infers Constraints", test_historian_infers_constraints),
        ("Historian Tracks Trends", test_historian_tracks_trends),
        ("Historian Detects Stagnation", test_historian_detects_stagnation),
        ("Full Historian Cycle", test_historian_full_cycle)
    ]

    print("\n" + "#"*70)
    print("# Historian Naval Agent Integration Tests")
    print("#"*70)

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n✅ PASSED: {name}\n")
            else:
                failed += 1
                print(f"\n❌ FAILED: {name}\n")
        except Exception as e:
            failed += 1
            print(f"\n❌ ERROR in {name}: {e}\n")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total:  {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("="*70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
