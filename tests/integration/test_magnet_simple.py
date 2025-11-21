#!/usr/bin/env python3
"""
Simple standalone test for MAGNET Agent 2
==========================================

Tests Agent 2 functionality without complex imports.
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Direct imports to avoid ARC conflicts
import importlib.util

# Load modules directly
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load our modules
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
base_naval = load_module("base_naval_agent", os.path.join(base_path, "agents/base_naval_agent.py"))
explorer = load_module("explorer_agent", os.path.join(base_path, "agents/explorer_agent.py"))
architect = load_module("architect_agent", os.path.join(base_path, "agents/experimental_architect_agent.py"))
local_client = load_module("local_client", os.path.join(base_path, "llm/local_client.py"))


def test_mock_llm():
    """Test mock LLM client"""
    print("\n" + "="*70)
    print("TEST 1: Mock LLM Client")
    print("="*70)

    client = local_client.MockLLMClient()

    # Test generation
    response = client.generate("Test prompt")
    assert len(response) > 0
    print(f"✓ Generated response: {len(response)} characters")

    # Test JSON
    json_resp = client.generate_json("Test")
    assert isinstance(json_resp, dict)
    print(f"✓ JSON generation works")

    # Test health
    assert client.health_check()
    print(f"✓ Health check passed")

    stats = client.get_stats()
    print(f"✓ Stats: {stats['total_calls']} calls")

    return True


def test_explorer():
    """Test Explorer agent"""
    print("\n" + "="*70)
    print("TEST 2: Explorer Agent")
    print("="*70)

    # Create client and agent
    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="explorer_001",
        role="explorer",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    agent = explorer.ExplorerAgent(config, client)

    # Run cycle
    context = {
        "knowledge_base": {},
        "experiment_history": [],
        "current_best": {
            "length_overall": 18.0,
            "beam": 6.0,
            "hull_spacing": 4.5
        },
        "cycle_number": 1
    }

    response = agent.autonomous_cycle(context)

    assert response.agent_id == "explorer_001"
    assert response.action == "submit_hypothesis"
    assert "hypothesis" in response.data

    hyp = response.data["hypothesis"]
    print(f"✓ Generated hypothesis: {hyp['statement'][:60]}...")
    print(f"✓ Type: {hyp['type']}")
    print(f"✓ Parameters to vary: {hyp['test_protocol']['parameters_to_vary']}")

    return True


def test_architect():
    """Test Architect agent"""
    print("\n" + "="*70)
    print("TEST 3: Architect Agent (Agent 2)")
    print("="*70)

    # Create client and agent
    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="architect_001",
        role="architect",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    agent = architect.ExperimentalArchitectAgent(config, client)

    # Create hypothesis
    hypothesis = {
        "id": "hyp_001",
        "statement": "Test hull spacing variation",
        "type": "exploration",
        "test_protocol": {
            "parameters_to_vary": ["hull_spacing"],
            "ranges": [[4.0, 6.0]],
            "num_samples": 8,
            "fixed_parameters": {}
        },
        "expected_outcome": "Find optimal spacing",
        "success_criteria": "stability > 75"
    }

    context = {
        "hypothesis": hypothesis,
        "current_best_design": architect.ExperimentalArchitectAgent.PARAMETER_DEFAULTS.copy()
    }

    # Run cycle
    response = agent.autonomous_cycle(context)

    assert response.agent_id == "architect_001"
    assert response.action == "submit_experiments"
    assert "designs" in response.data

    designs = response.data["designs"]
    print(f"✓ Designed {len(designs)} experiments")

    # Check first design
    first = designs[0]
    print(f"✓ Design ID: {first['design_id']}")
    print(f"✓ Has {len(first['parameters'])} parameters")

    # Check required parameters
    required = ["length_overall", "beam", "hull_spacing", "hull_depth"]
    for param in required:
        assert param in first['parameters']
    print(f"✓ All required parameters present")

    return True


def test_constraints():
    """Test constraint enforcement"""
    print("\n" + "="*70)
    print("TEST 4: Constraint Enforcement")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="architect_001",
        role="architect",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    agent = architect.ExperimentalArchitectAgent(config, client)

    hypothesis = {
        "id": "hyp_002",
        "statement": "Test extreme values",
        "type": "counter-intuitive",
        "test_protocol": {
            "parameters_to_vary": ["hull_spacing", "beam"],
            "ranges": [[5.0, 7.0], [5.0, 7.0]],
            "num_samples": 10,
            "fixed_parameters": {}
        },
        "expected_outcome": "Test",
        "success_criteria": "valid"
    }

    context = {
        "hypothesis": hypothesis,
        "current_best_design": architect.ExperimentalArchitectAgent.PARAMETER_DEFAULTS.copy()
    }

    response = agent.autonomous_cycle(context)
    designs = response.data["designs"]

    # Check constraints
    violations = 0
    for design in designs:
        params = design["parameters"]
        # Constraint: hull_spacing < beam
        if params["hull_spacing"] >= params["beam"]:
            violations += 1
            print(f"  Violation: spacing={params['hull_spacing']:.2f} >= beam={params['beam']:.2f}")

    print(f"✓ Checked {len(designs)} designs")
    print(f"✓ Constraint violations: {violations}")
    assert violations == 0, f"Found {violations} violations"

    return True


def test_sampling_strategies():
    """Test different sampling strategies"""
    print("\n" + "="*70)
    print("TEST 5: Sampling Strategies")
    print("="*70)

    client = local_client.MockLLMClient()
    config = base_naval.NavalAgentConfig(
        agent_id="architect_001",
        role="architect",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    agent = architect.ExperimentalArchitectAgent(config, client)

    strategies = ["exploration", "exploitation", "counter-intuitive"]

    for strategy in strategies:
        hypothesis = {
            "id": f"hyp_{strategy}",
            "statement": f"Test {strategy}",
            "type": strategy,
            "test_protocol": {
                "parameters_to_vary": ["hull_spacing"],
                "ranges": [[4.0, 6.0]],
                "num_samples": 6,
                "fixed_parameters": {}
            },
            "expected_outcome": "Test",
            "success_criteria": "valid"
        }

        context = {
            "hypothesis": hypothesis,
            "current_best_design": architect.ExperimentalArchitectAgent.PARAMETER_DEFAULTS.copy()
        }

        response = agent.autonomous_cycle(context)
        designs = response.data["designs"]

        print(f"✓ {strategy:20s}: {len(designs)} designs generated")

    return True


def main():
    """Run all tests"""
    tests = [
        ("Mock LLM Client", test_mock_llm),
        ("Explorer Agent", test_explorer),
        ("Architect Agent (Agent 2)", test_architect),
        ("Constraint Enforcement", test_constraints),
        ("Sampling Strategies", test_sampling_strategies)
    ]

    print("\n" + "#"*70)
    print("# MAGNET Agent 2 Integration Tests")
    print("# Simple standalone version")
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
