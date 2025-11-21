#!/usr/bin/env python3
"""
Standalone test runner for MAGNET Agent 2 tests
================================================

Runs tests without pytest to avoid conflicts with ARC conftest.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.base_naval_agent import NavalAgentConfig
from agents.explorer_agent import ExplorerAgent
from agents.experimental_architect_agent import ExperimentalArchitectAgent
from llm.local_client import MockLLMClient, LocalLLMClient


def test_explorer_generates_hypothesis():
    """Test that Explorer agent generates valid hypothesis"""
    print("\n" + "="*70)
    print("TEST: Explorer generates hypothesis")
    print("="*70)

    # Create mock LLM client
    mock_llm = MockLLMClient()

    # Create Explorer agent
    config = NavalAgentConfig(
        agent_id="explorer_001",
        role="explorer",
        model="mock-llm",
        memory_path="/tmp/magnet_test/memory"
    )
    explorer = ExplorerAgent(config, mock_llm)

    # Prepare context
    context = {
        "knowledge_base": {},
        "experiment_history": [],
        "current_best": {
            "length_overall": 18.0,
            "beam": 6.0,
            "hull_spacing": 4.5,
            "stability_score": 75.0,
            "speed_score": 70.0
        },
        "cycle_number": 1
    }

    # Generate hypothesis
    response = explorer.autonomous_cycle(context)

    # Validate
    assert response.agent_id == "explorer_001"
    assert response.action == "submit_hypothesis"
    assert "hypothesis" in response.data

    hypothesis = response.data["hypothesis"]
    print(f"✓ Generated hypothesis: {hypothesis['statement']}")
    print(f"✓ Type: {hypothesis['type']}")
    print(f"✓ Parameters to vary: {hypothesis['test_protocol']['parameters_to_vary']}")
    print(f"✓ Confidence: {hypothesis.get('confidence', 0.0)}")

    return True


def test_architect_designs_experiments():
    """Test that Architect agent designs valid experiments"""
    print("\n" + "="*70)
    print("TEST: Architect designs experiments")
    print("="*70)

    # Create mock LLM client
    mock_llm = MockLLMClient()

    # Create Architect agent
    config = NavalAgentConfig(
        agent_id="architect_001",
        role="architect",
        model="mock-llm",
        memory_path="/tmp/magnet_test/memory"
    )
    architect = ExperimentalArchitectAgent(config, mock_llm)

    # Create mock hypothesis
    hypothesis = {
        "id": "hyp_001_01",
        "statement": "Increasing hull spacing improves stability",
        "type": "exploration",
        "test_protocol": {
            "parameters_to_vary": ["hull_spacing"],
            "ranges": [[4.0, 6.0]],
            "num_samples": 8,
            "fixed_parameters": {}
        },
        "expected_outcome": "Stability score increases",
        "success_criteria": "stability_score > 78"
    }

    context = {
        "hypothesis": hypothesis,
        "current_best_design": {
            "length_overall": 18.0,
            "beam": 6.0,
            "hull_spacing": 4.5,
            "hull_depth": 2.5,
            "deadrise_angle": 12.0,
            "freeboard": 1.5,
            "lcb_position": 0.50,
            "prismatic_coefficient": 0.62
        }
    }

    # Design experiments
    response = architect.autonomous_cycle(context)

    # Validate
    assert response.agent_id == "architect_001"
    assert response.action == "submit_experiments"
    assert "designs" in response.data

    designs = response.data["designs"]
    print(f"✓ Designed {len(designs)} experiments")

    # Check first design
    first_design = designs[0]
    print(f"✓ Design ID: {first_design['design_id']}")
    print(f"✓ Parameters: {list(first_design['parameters'].keys())}")
    print(f"✓ Sample hull_spacing values: {[d['parameters']['hull_spacing'] for d in designs[:3]]}")

    return True


def test_constraint_enforcement():
    """Test that physical constraints are enforced"""
    print("\n" + "="*70)
    print("TEST: Constraint enforcement")
    print("="*70)

    mock_llm = MockLLMClient()
    config = NavalAgentConfig(
        agent_id="architect_001",
        role="architect",
        model="mock-llm",
        memory_path="/tmp/magnet_test/memory"
    )
    architect = ExperimentalArchitectAgent(config, mock_llm)

    # Create hypothesis that might violate constraints
    hypothesis = {
        "id": "hyp_002_01",
        "statement": "Test extreme hull spacing",
        "type": "counter-intuitive",
        "test_protocol": {
            "parameters_to_vary": ["hull_spacing", "beam"],
            "ranges": [[5.0, 7.0], [5.0, 7.0]],
            "num_samples": 10,
            "fixed_parameters": {}
        },
        "expected_outcome": "Identify boundaries",
        "success_criteria": "valid"
    }

    context = {
        "hypothesis": hypothesis,
        "current_best_design": ExperimentalArchitectAgent.PARAMETER_DEFAULTS.copy()
    }

    response = architect.autonomous_cycle(context)
    designs = response.data["designs"]

    # Check constraints
    violations = 0
    for design in designs:
        params = design["parameters"]

        # Constraint: hull_spacing < beam
        if params["hull_spacing"] >= params["beam"]:
            violations += 1

        # Constraint: freeboard < hull_depth
        if params["freeboard"] >= params["hull_depth"]:
            violations += 1

    print(f"✓ Checked {len(designs)} designs")
    print(f"✓ Constraint violations: {violations} (should be 0)")
    assert violations == 0, f"Found {violations} constraint violations"

    return True


def test_full_cycle():
    """Test full cycle: Explorer → Architect"""
    print("\n" + "="*70)
    print("TEST: Full cycle (Explorer → Architect)")
    print("="*70)

    mock_llm = MockLLMClient()

    # Create agents
    explorer_config = NavalAgentConfig(
        agent_id="explorer_001",
        role="explorer",
        model="mock-llm",
        memory_path="/tmp/magnet_test/memory"
    )
    explorer = ExplorerAgent(explorer_config, mock_llm)

    architect_config = NavalAgentConfig(
        agent_id="architect_001",
        role="architect",
        model="mock-llm",
        memory_path="/tmp/magnet_test/memory"
    )
    architect = ExperimentalArchitectAgent(architect_config, mock_llm)

    # Step 1: Explorer generates hypothesis
    explorer_context = {
        "knowledge_base": {},
        "experiment_history": [
            {
                "parameters": {"length_overall": 18.0, "beam": 6.0, "hull_spacing": 4.5},
                "results": {"stability_score": 75.0, "speed_score": 70.0}
            }
        ],
        "current_best": {
            "length_overall": 18.0,
            "beam": 6.0,
            "hull_spacing": 4.5,
            "stability_score": 75.0,
            "speed_score": 70.0
        },
        "cycle_number": 1
    }

    explorer_response = explorer.autonomous_cycle(explorer_context)
    hypothesis = explorer_response.data["hypothesis"]
    print(f"✓ Explorer: {hypothesis['statement']}")

    # Step 2: Architect designs experiments
    architect_context = {
        "hypothesis": hypothesis,
        "current_best_design": ExperimentalArchitectAgent.PARAMETER_DEFAULTS.copy()
    }

    architect_response = architect.autonomous_cycle(architect_context)
    designs = architect_response.data["designs"]
    print(f"✓ Architect: Designed {len(designs)} experiments")

    # Step 3: Validate for Physics Engine
    required_params = [
        "length_overall", "beam", "hull_spacing", "hull_depth",
        "deadrise_angle", "freeboard", "lcb_position",
        "prismatic_coefficient"
    ]

    for design in designs:
        params = design["parameters"]
        for param in required_params:
            assert param in params, f"Missing parameter: {param}"

    print(f"✓ All designs have required parameters for Physics Engine")
    print(f"✓ Ready for simulation!")

    return True


def test_llm_client():
    """Test LLM client functionality"""
    print("\n" + "="*70)
    print("TEST: LLM Client")
    print("="*70)

    client = MockLLMClient()

    # Test generation
    response = client.generate("Test prompt")
    print(f"✓ Generated response: {len(response)} chars")

    # Test JSON generation
    json_response = client.generate_json("Test JSON prompt")
    print(f"✓ Generated JSON: {list(json_response.keys())}")

    # Test health check
    health = client.health_check()
    print(f"✓ Health check: {health}")

    # Test stats
    stats = client.get_stats()
    print(f"✓ Stats: {stats['total_calls']} calls, {stats['success_rate']*100:.0f}% success")

    # Test JSON extraction
    test_cases = [
        ('{"key": "value"}', "pure JSON"),
        ('```json\n{"key": "value"}\n```', "code block"),
        ('<think>...</think>{"key": "value"}', "with <think> tags")
    ]

    for text, description in test_cases:
        result = LocalLLMClient.extract_json(text)
        assert result == {"key": "value"}
        print(f"✓ JSON extraction works for {description}")

    return True


def main():
    """Run all tests"""
    tests = [
        ("Explorer generates hypothesis", test_explorer_generates_hypothesis),
        ("Architect designs experiments", test_architect_designs_experiments),
        ("Constraint enforcement", test_constraint_enforcement),
        ("Full cycle", test_full_cycle),
        ("LLM client", test_llm_client)
    ]

    print("\n" + "#"*70)
    print("# MAGNET Agent 2 Integration Tests")
    print("#"*70)

    passed = 0
    failed = 0
    errors = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n✅ PASSED: {test_name}")
            else:
                failed += 1
                print(f"\n❌ FAILED: {test_name}")
                errors.append(test_name)
        except Exception as e:
            failed += 1
            print(f"\n❌ ERROR in {test_name}: {e}")
            errors.append(f"{test_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if errors:
        print("\nFailed tests:")
        for error in errors:
            print(f"  - {error}")

    print("="*70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
