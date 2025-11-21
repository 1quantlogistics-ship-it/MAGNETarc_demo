#!/usr/bin/env python3
"""
Integration Tests for Critic Naval Agent
==========================================

Tests the Critic agent's ability to review designs and critique results.
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
critic = load_module("critic_naval_agent", os.path.join(base_path, "agents/critic_naval_agent.py"))
architect = load_module("architect_agent", os.path.join(base_path, "agents/experimental_architect_agent.py"))
local_client = load_module("local_client", os.path.join(base_path, "llm/local_client.py"))


def test_critic_review_safe_designs():
    """Test Critic reviews safe designs and approves"""
    print("\n" + "="*70)
    print("TEST 1: Critic Reviews Safe Designs")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="critic_001",
        role="critic",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    critic_agent = critic.CriticNavalAgent(config, client)

    # Create safe designs
    designs = [
        {
            "design_id": "test_001",
            "parameters": {
                "length_overall": 18.0,
                "beam": 6.0,
                "hull_spacing": 4.5,
                "hull_depth": 2.5,
                "freeboard": 1.5,
                "displacement": 45000.0
            }
        },
        {
            "design_id": "test_002",
            "parameters": {
                "length_overall": 19.0,
                "beam": 6.5,
                "hull_spacing": 5.0,
                "hull_depth": 2.7,
                "freeboard": 1.6,
                "displacement": 48000.0
            }
        }
    ]

    hypothesis = {
        "id": "hyp_001",
        "statement": "Test designs",
        "test_protocol": {
            "parameters_to_vary": ["hull_spacing"],
            "ranges": [[4.0, 6.0]]
        }
    }

    context = {
        "designs": designs,
        "hypothesis": hypothesis,
        "experiment_history": []
    }

    # Run review
    response = critic_agent.autonomous_cycle(context)

    assert response.agent_id == "critic_001"
    assert response.action == "submit_pre_review"
    assert "verdict" in response.data

    critique = response.data
    print(f"✓ Verdict: {critique['verdict']}")
    print(f"✓ Designs reviewed: {critique['designs_reviewed']}")
    print(f"✓ Safety flags: {len(critique['safety_flags'])}")
    print(f"✓ Concerns: {len(critique['concerns'])}")

    assert critique["verdict"] == "approve", "Safe designs should be approved"
    assert len(critique["safety_flags"]) == 0, "No safety flags expected for safe designs"

    return True


def test_critic_detects_unsafe_designs():
    """Test Critic detects and rejects unsafe designs"""
    print("\n" + "="*70)
    print("TEST 2: Critic Detects Unsafe Designs")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="critic_001",
        role="critic",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    critic_agent = critic.CriticNavalAgent(config, client)

    # Create unsafe designs
    designs = [
        {
            "design_id": "unsafe_001",
            "parameters": {
                "length_overall": 18.0,
                "beam": 6.0,
                "hull_spacing": 6.5,  # UNSAFE: spacing > beam!
                "hull_depth": 2.5,
                "freeboard": 1.5,
                "displacement": 45000.0
            }
        },
        {
            "design_id": "unsafe_002",
            "parameters": {
                "length_overall": 18.0,
                "beam": 6.0,
                "hull_spacing": 4.5,
                "hull_depth": 2.5,
                "freeboard": 2.6,  # UNSAFE: freeboard > depth!
                "displacement": 45000.0
            }
        }
    ]

    hypothesis = {
        "id": "hyp_002",
        "statement": "Test unsafe designs",
        "test_protocol": {
            "parameters_to_vary": ["hull_spacing", "freeboard"],
            "ranges": [[4.0, 7.0], [1.0, 3.0]]
        }
    }

    context = {
        "designs": designs,
        "hypothesis": hypothesis,
        "experiment_history": []
    }

    response = critic_agent.autonomous_cycle(context)
    critique = response.data

    print(f"✓ Verdict: {critique['verdict']}")
    print(f"✓ Safety flags: {len(critique['safety_flags'])}")
    for flag in critique["safety_flags"]:
        print(f"  - {flag['design_id']}: {len(flag['issues'])} issues")
        for issue in flag["issues"][:2]:
            print(f"    * {issue}")

    assert critique["verdict"] == "reject", "Unsafe designs should be rejected"
    assert len(critique["safety_flags"]) > 0, "Safety flags should be raised"

    return True


def test_critic_detects_redundancy():
    """Test Critic detects redundant designs"""
    print("\n" + "="*70)
    print("TEST 3: Critic Detects Redundancy")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="critic_001",
        role="critic",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    critic_agent = critic.CriticNavalAgent(config, client)

    # Create near-duplicate designs
    base_params = {
        "length_overall": 18.0,
        "beam": 6.0,
        "hull_spacing": 4.5,
        "hull_depth": 2.5,
        "freeboard": 1.5,
        "deadrise_angle": 12.0,
        "lcb_position": 0.50,
        "prismatic_coefficient": 0.62,
        "displacement": 45000.0
    }

    designs = [
        {"design_id": "dup_001", "parameters": base_params.copy()},
        {"design_id": "dup_002", "parameters": {**base_params, "hull_spacing": 4.51}},  # Nearly identical
        {"design_id": "unique_001", "parameters": {**base_params, "hull_spacing": 5.5}}  # Different
    ]

    hypothesis = {"id": "hyp_003", "test_protocol": {"parameters_to_vary": ["hull_spacing"], "ranges": [[4.0, 6.0]]}}

    context = {
        "designs": designs,
        "hypothesis": hypothesis,
        "experiment_history": []
    }

    response = critic_agent.autonomous_cycle(context)
    critique = response.data

    print(f"✓ Verdict: {critique['verdict']}")
    print(f"✓ Redundancies detected: {len(critique['redundancies'])}")
    for red in critique["redundancies"]:
        print(f"  - {red['design_id']}: {red['similarity']} similar")

    assert len(critique["redundancies"]) > 0, "Should detect near-duplicate designs"

    return True


def test_critic_analyzes_results():
    """Test Critic analyzes simulation results"""
    print("\n" + "="*70)
    print("TEST 4: Critic Analyzes Simulation Results")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="critic_001",
        role="critic",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    critic_agent = critic.CriticNavalAgent(config, client)

    # Mock simulation results
    experiment_results = [
        {
            "design_id": "exp_001",
            "parameters": {"length_overall": 18.0, "beam": 6.0, "hull_spacing": 4.5},
            "results": {
                "stability_score": 78.5,
                "speed_score": 72.3,
                "efficiency_score": 65.8,
                "overall_score": 72.2,
                "is_valid": True,
                "failure_reasons": []
            }
        },
        {
            "design_id": "exp_002",
            "parameters": {"length_overall": 19.0, "beam": 6.5, "hull_spacing": 5.0},
            "results": {
                "stability_score": 82.1,
                "speed_score": 75.4,
                "efficiency_score": 68.9,
                "overall_score": 75.5,
                "is_valid": True,
                "failure_reasons": []
            }
        },
        {
            "design_id": "exp_003",
            "parameters": {"length_overall": 20.0, "beam": 7.0, "hull_spacing": 5.5},
            "results": {
                "is_valid": False,
                "failure_reasons": ["stability_failure"]
            }
        }
    ]

    hypothesis = {
        "id": "hyp_004",
        "statement": "Larger hulls improve performance",
        "success_criteria": "overall_score > 70"
    }

    context = {
        "experiment_results": experiment_results,
        "hypothesis": hypothesis,
        "experiment_history": []
    }

    response = critic_agent.autonomous_cycle(context)
    critique = response.data

    print(f"✓ Verdict: {critique['verdict']}")
    print(f"✓ Successful: {critique['successful_designs']}/{critique['results_analyzed']}")
    print(f"✓ Hypothesis met: {critique['hypothesis_met']}")
    print(f"✓ Insights: {len(critique['insights'])}")
    for insight in critique["insights"][:3]:
        print(f"  - {insight}")

    assert critique["successful_designs"] == 2
    assert critique["failed_designs"] == 1
    assert len(critique["insights"]) > 0

    return True


def test_critic_identifies_breakthroughs():
    """Test Critic identifies breakthrough designs"""
    print("\n" + "="*70)
    print("TEST 5: Critic Identifies Breakthroughs")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="critic_001",
        role="critic",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    critic_agent = critic.CriticNavalAgent(config, client)

    # Historical results with lower scores
    experiment_history = [
        {
            "design_id": "old_001",
            "results": {"overall_score": 65.0, "is_valid": True}
        },
        {
            "design_id": "old_002",
            "results": {"overall_score": 68.0, "is_valid": True}
        }
    ]

    # New results with breakthrough
    experiment_results = [
        {
            "design_id": "new_001",
            "parameters": {"length_overall": 18.5, "beam": 6.2},
            "results": {
                "overall_score": 80.5,  # Breakthrough! (+12.5 over historical best)
                "is_valid": True
            }
        }
    ]

    hypothesis = {"id": "hyp_005", "statement": "Test breakthrough"}

    context = {
        "experiment_results": experiment_results,
        "hypothesis": hypothesis,
        "experiment_history": experiment_history
    }

    response = critic_agent.autonomous_cycle(context)
    critique = response.data

    print(f"✓ Breakthroughs identified: {len(critique['breakthroughs'])}")
    for bt in critique["breakthroughs"]:
        print(f"  - {bt['design_id']}: {bt['reason']}")

    assert len(critique["breakthroughs"]) > 0, "Should identify breakthrough design"

    return True


def main():
    """Run all Critic tests"""
    tests = [
        ("Critic Reviews Safe Designs", test_critic_review_safe_designs),
        ("Critic Detects Unsafe Designs", test_critic_detects_unsafe_designs),
        ("Critic Detects Redundancy", test_critic_detects_redundancy),
        ("Critic Analyzes Results", test_critic_analyzes_results),
        ("Critic Identifies Breakthroughs", test_critic_identifies_breakthroughs)
    ]

    print("\n" + "#"*70)
    print("# Critic Naval Agent Integration Tests")
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
