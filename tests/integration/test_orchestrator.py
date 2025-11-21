#!/usr/bin/env python3
"""
Integration Tests for Autonomous Orchestrator
==============================================

Tests the Autonomous Orchestrator's ability to coordinate all agents
through complete research cycles.
"""

import sys
import os
import asyncio

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

# Add base path to sys.path for imports
import sys
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# Now import normally
from orchestration.autonomous_orchestrator import create_orchestrator, AutonomousOrchestrator
from llm.local_client import MockLLMClient

local_client = load_module("local_client", os.path.join(base_path, "llm/local_client.py"))


def mock_physics_simulation(designs):
    """Mock physics simulation for testing"""
    import random

    results = []
    for design in designs:
        design_id = design.get("design_id")
        params = design.get("parameters", {})

        loa = params.get("length_overall", 18.0)
        beam = params.get("beam", 6.0)
        spacing = params.get("hull_spacing", 4.5)
        depth = params.get("hull_depth", 2.5)

        is_valid = True
        failure_reasons = []

        if spacing >= beam:
            is_valid = False
            failure_reasons.append("hull_spacing >= beam")

        if params.get("freeboard", 1.5) >= depth:
            is_valid = False
            failure_reasons.append("freeboard >= depth")

        if is_valid:
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


def test_orchestrator_single_cycle():
    """Test orchestrator runs a single cycle successfully"""
    print("\n" + "="*70)
    print("TEST 1: Orchestrator Single Cycle")
    print("="*70)

    # Create orchestrator
    llm = MockLLMClient()
    orchestrator = create_orchestrator(
        llm_client=llm,
        physics_simulator=mock_physics_simulation,
        memory_path="/tmp/magnet_test_orch",
        config={"cycle_delay": 0}
    )

    # Run single cycle
    async def run_test():
        cycle_results = await orchestrator.run_cycle()
        return cycle_results

    results = asyncio.run(run_test())

    print(f"✓ Cycle number: {results['cycle']}")
    print(f"✓ Success: {results.get('success', False)}")
    print(f"✓ Steps completed: {len(results['steps'])}")

    # Verify all steps ran
    expected_steps = ["explorer", "architect", "critic_pre", "physics", "critic_post", "historian", "supervisor"]
    for step in expected_steps:
        assert step in results["steps"], f"Step '{step}' missing"
        print(f"  ✓ {step}: {results['steps'][step].get('success', False)}")

    assert results.get("success", False), "Cycle should succeed"
    assert orchestrator.state.cycle_number == 1, "Should complete 1 cycle"

    return True


def test_orchestrator_multiple_cycles():
    """Test orchestrator runs multiple cycles"""
    print("\n" + "="*70)
    print("TEST 2: Orchestrator Multiple Cycles")
    print("="*70)

    # Create orchestrator
    llm = MockLLMClient()
    orchestrator = create_orchestrator(
        llm_client=llm,
        physics_simulator=mock_physics_simulation,
        memory_path="/tmp/magnet_test_orch2",
        config={"cycle_delay": 0}
    )

    # Run 3 cycles
    async def run_test():
        await orchestrator.run(max_cycles=3)

    asyncio.run(run_test())

    print(f"✓ Total cycles: {orchestrator.state.cycle_number}")
    print(f"✓ Total experiments: {orchestrator.state.total_experiments}")
    print(f"✓ Valid designs: {orchestrator.state.total_valid_designs}")
    print(f"✓ Best score: {orchestrator.state.best_overall_score:.1f}")
    print(f"✓ Errors: {orchestrator.state.error_count}")

    assert orchestrator.state.cycle_number == 3, "Should complete 3 cycles"
    assert orchestrator.state.total_experiments > 0, "Should run experiments"
    assert orchestrator.state.error_count == 0, "Should have no errors"

    return True


def test_orchestrator_state_persistence():
    """Test orchestrator saves and loads state"""
    print("\n" + "="*70)
    print("TEST 3: Orchestrator State Persistence")
    print("="*70)

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    state_file = os.path.join(temp_dir, "orch_state.json")

    try:
        # Create orchestrator and run 2 cycles
        llm = local_client.MockLLMClient()
        orch1 = orchestrator_module.create_orchestrator(
            llm_client=llm,
            physics_simulator=mock_physics_simulation,
            memory_path="/tmp/magnet_test_orch3",
            config={"cycle_delay": 0, "state_file": state_file}
        )

        async def run_test1():
            await orch1.run(max_cycles=2)

        asyncio.run(run_test1())

        cycle_count_1 = orch1.state.cycle_number
        best_score_1 = orch1.state.best_overall_score

        print(f"✓ First run: {cycle_count_1} cycles, best={best_score_1:.1f}")

        # Create new orchestrator from same state file
        orch2 = orchestrator_module.create_orchestrator(
            llm_client=llm,
            physics_simulator=mock_physics_simulation,
            memory_path="/tmp/magnet_test_orch3",
            config={"cycle_delay": 0, "state_file": state_file}
        )

        cycle_count_2 = orch2.state.cycle_number
        best_score_2 = orch2.state.best_overall_score

        print(f"✓ Loaded state: {cycle_count_2} cycles, best={best_score_2:.1f}")

        assert cycle_count_2 == cycle_count_1, "Should load same cycle count"
        assert abs(best_score_2 - best_score_1) < 0.1, "Should load same best score"

        # Run 1 more cycle
        async def run_test2():
            await orch2.run(max_cycles=3)

        asyncio.run(run_test2())

        print(f"✓ After resume: {orch2.state.cycle_number} cycles")

        assert orch2.state.cycle_number == 3, "Should continue from saved state"

    finally:
        shutil.rmtree(temp_dir)

    return True


def test_orchestrator_knowledge_base_integration():
    """Test orchestrator integrates with knowledge base"""
    print("\n" + "="*70)
    print("TEST 4: Orchestrator Knowledge Base Integration")
    print("="*70)

    # Create orchestrator
    llm = MockLLMClient()
    orchestrator = create_orchestrator(
        llm_client=llm,
        physics_simulator=mock_physics_simulation,
        memory_path="/tmp/magnet_test_orch4",
        config={"cycle_delay": 0}
    )

    # Run 2 cycles
    async def run_test():
        await orchestrator.run(max_cycles=2)

    asyncio.run(run_test())

    # Check knowledge base
    kb = orchestrator.knowledge_base
    stats = kb.get_statistics()

    print(f"✓ KB experiments: {stats['total_experiments']}")
    print(f"✓ KB designs evaluated: {stats['total_designs_evaluated']}")
    print(f"✓ KB cycles: {stats['total_cycles']}")

    assert len(kb.experiments) == 2, "Should store 2 experiments in KB"
    assert stats['total_cycles'] == 2, "KB should track 2 cycles"

    # Check best designs
    best_designs = kb.get_best_designs(n=3)
    print(f"✓ Best designs stored: {len(best_designs)}")

    assert len(best_designs) > 0, "Should have best designs"

    return True


def test_orchestrator_strategy_evolution():
    """Test orchestrator adapts strategy over cycles"""
    print("\n" + "="*70)
    print("TEST 5: Orchestrator Strategy Evolution")
    print("="*70)

    # Create orchestrator
    llm = MockLLMClient()
    orchestrator = create_orchestrator(
        llm_client=llm,
        physics_simulator=mock_physics_simulation,
        memory_path="/tmp/magnet_test_orch5",
        config={"cycle_delay": 0}
    )

    # Track strategy changes
    strategies = []

    async def run_test():
        for cycle in range(5):
            await orchestrator.run_cycle()
            strategy = orchestrator.state.exploration_strategy.copy()
            strategies.append(strategy)

    asyncio.run(run_test())

    print(f"✓ Completed {len(strategies)} cycles")
    for i, strategy in enumerate(strategies, 1):
        mode = strategy.get("mode", "unknown")
        temp = strategy.get("exploration_temperature", 0.0)
        print(f"  Cycle {i}: {mode} (temp={temp:.2f})")

    # Early cycles should use exploration
    assert strategies[0].get("mode") == "exploration", "Cycle 1 should explore"
    assert len(strategies) == 5, "Should track 5 strategy changes"

    return True


def test_orchestrator_error_handling():
    """Test orchestrator handles errors gracefully"""
    print("\n" + "="*70)
    print("TEST 6: Orchestrator Error Handling")
    print("="*70)

    def failing_physics(designs):
        """Physics simulator that fails"""
        raise RuntimeError("Mock physics failure")

    # Create orchestrator with failing physics
    llm = MockLLMClient()
    orchestrator = create_orchestrator(
        llm_client=llm,
        physics_simulator=failing_physics,
        memory_path="/tmp/magnet_test_orch6",
        config={"cycle_delay": 0}
    )

    # Run cycle (should handle error)
    async def run_test():
        cycle_results = await orchestrator.run_cycle()
        return cycle_results

    results = asyncio.run(run_test())

    print(f"✓ Cycle completed: {results.get('success', True)}")
    print(f"✓ Error recorded: {orchestrator.state.error_count}")
    print(f"✓ Last error: {orchestrator.state.last_error}")

    assert not results.get("success", True), "Cycle should fail"
    assert orchestrator.state.error_count > 0, "Should track error"
    assert orchestrator.state.last_error is not None, "Should record error message"

    return True


def main():
    """Run all Orchestrator tests"""
    tests = [
        ("Orchestrator Single Cycle", test_orchestrator_single_cycle),
        ("Orchestrator Multiple Cycles", test_orchestrator_multiple_cycles),
        ("Orchestrator State Persistence", test_orchestrator_state_persistence),
        ("Orchestrator Knowledge Base Integration", test_orchestrator_knowledge_base_integration),
        ("Orchestrator Strategy Evolution", test_orchestrator_strategy_evolution),
        ("Orchestrator Error Handling", test_orchestrator_error_handling),
    ]

    print("\n" + "#"*70)
    print("# Autonomous Orchestrator Integration Tests")
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
