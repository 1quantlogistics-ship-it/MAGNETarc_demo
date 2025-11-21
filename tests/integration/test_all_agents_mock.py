#!/usr/bin/env python3
"""
Full Integration Test for All MAGNET Agents (Mock Mode)
========================================================

Tests the complete agent cycle with mock physics simulation.
This version works without the physics engine (for Mac/CPU development).

Flow:
1. Explorer generates hypothesis
2. Architect designs experiments
3. Critic reviews designs (pre-simulation)
4. [Mock Physics Simulation]
5. Critic critiques results (post-simulation)
6. Historian compresses history and extracts patterns
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Direct imports
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load all modules
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
base_naval = load_module("base_naval_agent", os.path.join(base_path, "agents/base_naval_agent.py"))
explorer = load_module("explorer_agent", os.path.join(base_path, "agents/explorer_agent.py"))
architect = load_module("architect_agent", os.path.join(base_path, "agents/experimental_architect_agent.py"))
critic = load_module("critic_naval_agent", os.path.join(base_path, "agents/critic_naval_agent.py"))
historian = load_module("historian_naval_agent", os.path.join(base_path, "agents/historian_naval_agent.py"))
local_client = load_module("local_client", os.path.join(base_path, "llm/local_client.py"))


def mock_physics_simulation(designs):
    """
    Mock physics simulation for testing.

    Returns realistic-looking results without actual physics calculations.
    """
    import random

    results = []
    for design in designs:
        design_id = design.get("design_id")
        params = design.get("parameters", {})

        # Simple heuristic: score based on parameters
        loa = params.get("length_overall", 18.0)
        beam = params.get("beam", 6.0)
        spacing = params.get("hull_spacing", 4.5)
        depth = params.get("hull_depth", 2.5)

        # Check for safety violations
        is_valid = True
        failure_reasons = []

        if spacing >= beam:
            is_valid = False
            failure_reasons.append("hull_spacing >= beam")

        if params.get("freeboard", 1.5) >= depth:
            is_valid = False
            failure_reasons.append("freeboard >= depth")

        # Calculate mock scores
        if is_valid:
            # Simple scoring based on L/B ratio and spacing
            lb_ratio = loa / beam
            stability_score = 70.0 + (spacing / beam) * 10.0 + random.uniform(-5, 5)
            speed_score = 65.0 + (lb_ratio - 3.0) * 5.0 + random.uniform(-5, 5)
            efficiency_score = 60.0 + (loa / 20.0) * 15.0 + random.uniform(-5, 5)
            overall_score = (stability_score + speed_score + efficiency_score) / 3.0

            result = {
                "design_id": design_id,
                "parameters": params,
                "results": {
                    "is_valid": True,
                    "stability_score": max(0, min(100, stability_score)),
                    "speed_score": max(0, min(100, speed_score)),
                    "efficiency_score": max(0, min(100, efficiency_score)),
                    "overall_score": max(0, min(100, overall_score)),
                    "failure_reasons": []
                }
            }
        else:
            result = {
                "design_id": design_id,
                "parameters": params,
                "results": {
                    "is_valid": False,
                    "failure_reasons": failure_reasons
                }
            }

        results.append(result)

    return results


def test_full_agent_cycle():
    """Test complete agent cycle with all agents"""
    print("\n" + "="*70)
    print("FULL AGENT CYCLE TEST (Mock Physics)")
    print("="*70)

    # Initialize LLM client (mock mode)
    llm = local_client.MockLLMClient()

    # Create all agents
    explorer_config = base_naval.NavalAgentConfig(
        agent_id="explorer_001",
        role="explorer",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    explorer_agent = explorer.ExplorerAgent(explorer_config, llm)

    architect_config = base_naval.NavalAgentConfig(
        agent_id="architect_001",
        role="architect",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    architect_agent = architect.ExperimentalArchitectAgent(architect_config, llm)

    critic_config = base_naval.NavalAgentConfig(
        agent_id="critic_001",
        role="critic",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    critic_agent = critic.CriticNavalAgent(critic_config, llm)

    historian_config = base_naval.NavalAgentConfig(
        agent_id="historian_001",
        role="historian",
        model="mock",
        memory_path="/tmp/magnet_test"
    )
    historian_agent = historian.HistorianNavalAgent(historian_config, llm)

    print("\n[Step 1] Explorer generates hypothesis...")

    # Initial context for Explorer
    explorer_context = {
        "knowledge_base": {},
        "experiment_history": [],
        "current_best": {
            "length_overall": 18.0,
            "beam": 6.0,
            "hull_spacing": 4.5,
            "stability_score": 70.0,
            "speed_score": 68.0
        },
        "cycle_number": 1
    }

    explorer_response = explorer_agent.autonomous_cycle(explorer_context)
    hypothesis = explorer_response.data["hypothesis"]

    print(f"✓ Hypothesis: {hypothesis['statement'][:60]}...")
    print(f"✓ Type: {hypothesis['type']}")
    print(f"✓ Parameters to vary: {hypothesis['test_protocol']['parameters_to_vary']}")

    # -------------------------------------------------------------------------

    print("\n[Step 2] Architect designs experiments...")

    architect_context = {
        "hypothesis": hypothesis,
        "current_best_design": architect.ExperimentalArchitectAgent.PARAMETER_DEFAULTS.copy()
    }

    architect_response = architect_agent.autonomous_cycle(architect_context)
    designs = architect_response.data["designs"]

    print(f"✓ Designed {len(designs)} experiments")
    print(f"✓ First design ID: {designs[0]['design_id']}")

    # -------------------------------------------------------------------------

    print("\n[Step 3] Critic reviews designs (pre-simulation)...")

    critic_pre_context = {
        "designs": designs,
        "hypothesis": hypothesis,
        "experiment_history": []
    }

    critic_pre_response = critic_agent.autonomous_cycle(critic_pre_context)
    pre_review = critic_pre_response.data

    print(f"✓ Verdict: {pre_review['verdict']}")
    print(f"✓ Safety flags: {len(pre_review['safety_flags'])}")
    print(f"✓ Recommendations: {len(pre_review['recommendations'])}")

    # -------------------------------------------------------------------------

    print("\n[Step 4] Mock Physics simulation...")

    # Run mock physics
    experiment_results = mock_physics_simulation(designs)

    successful = [r for r in experiment_results if r["results"]["is_valid"]]
    failed = [r for r in experiment_results if not r["results"]["is_valid"]]

    print(f"✓ Simulated {len(experiment_results)} designs")
    print(f"✓ Successful: {len(successful)}")
    print(f"✓ Failed: {len(failed)}")

    if successful:
        best = max(successful, key=lambda r: r["results"]["overall_score"])
        print(f"✓ Best score: {best['results']['overall_score']:.1f}")

    # -------------------------------------------------------------------------

    print("\n[Step 5] Critic critiques results (post-simulation)...")

    critic_post_context = {
        "experiment_results": experiment_results,
        "hypothesis": hypothesis,
        "experiment_history": []
    }

    critic_post_response = critic_agent.autonomous_cycle(critic_post_context)
    critique = critic_post_response.data

    print(f"✓ Verdict: {critique['verdict']}")
    print(f"✓ Successful: {critique['successful_designs']}/{critique['results_analyzed']}")
    print(f"✓ Insights: {len(critique['insights'])}")
    if critique['insights']:
        print(f"  - {critique['insights'][0]}")

    # -------------------------------------------------------------------------

    print("\n[Step 6] Historian compresses history and extracts patterns...")

    historian_context = {
        "new_results": experiment_results,
        "current_history": {"experiments": []},
        "knowledge_base": {},
        "cycle_number": 1
    }

    historian_response = historian_agent.autonomous_cycle(historian_context)
    history_analysis = historian_response.data

    print(f"✓ Compressed to {history_analysis['compressed_history']['kept_experiments']} experiments")
    print(f"✓ Patterns identified: {len(history_analysis['new_patterns'])}")
    if history_analysis['new_patterns']:
        print(f"  - {history_analysis['new_patterns'][0]}")
    print(f"✓ Constraints inferred: {len(history_analysis['inferred_constraints'])}")

    # -------------------------------------------------------------------------

    print("\n" + "="*70)
    print("FULL CYCLE VALIDATION")
    print("="*70)

    # Validate all agents used correct interfaces
    checks = [
        ("Explorer response format", "hypothesis" in explorer_response.data),
        ("Architect response format", "designs" in architect_response.data),
        ("Critic pre-review format", "verdict" in pre_review),
        ("Critic post-critique format", "insights" in critique),
        ("Historian format", "compressed_history" in history_analysis),
        ("All agents use NavalAgentResponse", all([
            hasattr(explorer_response, 'agent_id') and hasattr(explorer_response, 'data'),
            hasattr(architect_response, 'agent_id') and hasattr(architect_response, 'data'),
            hasattr(critic_pre_response, 'agent_id') and hasattr(critic_pre_response, 'data'),
            hasattr(critic_post_response, 'agent_id') and hasattr(critic_post_response, 'data'),
            hasattr(historian_response, 'agent_id') and hasattr(historian_response, 'data')
        ])),
        ("Designs have correct format", all("parameters" in d for d in designs)),
        ("Results have correct format", all("results" in r for r in experiment_results)),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False

    print("="*70)

    return all_passed


def test_multi_cycle():
    """Test multiple research cycles"""
    print("\n" + "="*70)
    print("MULTI-CYCLE TEST (3 Cycles)")
    print("="*70)

    llm = local_client.MockLLMClient()

    # Create agents
    explorer_agent = explorer.ExplorerAgent(
        base_naval.NavalAgentConfig("explorer_001", "explorer", "mock", memory_path="/tmp/magnet_test"),
        llm
    )
    architect_agent = architect.ExperimentalArchitectAgent(
        base_naval.NavalAgentConfig("architect_001", "architect", "mock", memory_path="/tmp/magnet_test"),
        llm
    )
    historian_agent = historian.HistorianNavalAgent(
        base_naval.NavalAgentConfig("historian_001", "historian", "mock", memory_path="/tmp/magnet_test"),
        llm
    )

    # Run 3 cycles
    all_results = []
    current_history = {"experiments": []}

    for cycle in range(1, 4):
        print(f"\n--- Cycle {cycle} ---")

        # Explorer
        explorer_context = {
            "knowledge_base": {},
            "experiment_history": all_results,
            "current_best": {
                "length_overall": 18.0,
                "beam": 6.0,
                "hull_spacing": 4.5
            },
            "cycle_number": cycle
        }
        explorer_response = explorer_agent.autonomous_cycle(explorer_context)
        hypothesis = explorer_response.data["hypothesis"]

        # Architect
        architect_context = {
            "hypothesis": hypothesis,
            "current_best_design": architect.ExperimentalArchitectAgent.PARAMETER_DEFAULTS.copy()
        }
        architect_response = architect_agent.autonomous_cycle(architect_context)
        designs = architect_response.data["designs"]

        # Mock physics
        results = mock_physics_simulation(designs)

        # Add cycle number to results
        for r in results:
            r["cycle_number"] = cycle

        all_results.extend(results)

        # Historian
        historian_context = {
            "new_results": results,
            "current_history": current_history,
            "knowledge_base": {},
            "cycle_number": cycle
        }
        historian_response = historian_agent.autonomous_cycle(historian_context)
        current_history = historian_response.data["compressed_history"]

        print(f"  ✓ Hypothesis generated")
        print(f"  ✓ {len(designs)} designs created")
        print(f"  ✓ {len(results)} results simulated")
        print(f"  ✓ History compressed to {current_history['kept_experiments']} experiments")

    print(f"\n✓ Completed 3 cycles")
    print(f"✓ Total experiments: {len(all_results)}")
    print(f"✓ Compressed to: {current_history['kept_experiments']} experiments")
    print(f"✓ Compression ratio: {current_history['compression_ratio']:.2f}")

    return True


def main():
    """Run all integration tests"""
    tests = [
        ("Full Agent Cycle (6 steps)", test_full_agent_cycle),
        ("Multi-Cycle Test (3 cycles)", test_multi_cycle)
    ]

    print("\n" + "#"*70)
    print("# MAGNET Full Integration Tests (Mock Mode)")
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
