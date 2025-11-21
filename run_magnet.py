#!/usr/bin/env python3
"""
MAGNET Autonomous Research System - CLI Entry Point
====================================================

Command-line interface for running MAGNET autonomous naval design research.

Usage:
    python3 run_magnet.py --cycles 10 --mock
    python3 run_magnet.py --cycles 5 --gpu
    python3 run_magnet.py --resume

Author: Agent 2
"""

import argparse
import asyncio
import logging
from pathlib import Path

# Ensure project is in path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from orchestration.autonomous_orchestrator import create_orchestrator
from llm.local_client import LocalLLMClient, MockLLMClient


def mock_physics_simulation(designs):
    """Mock physics simulation for CPU-only testing"""
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


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MAGNET Autonomous Naval Design Research System"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Number of research cycles to run (default: infinite)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM and physics (CPU-only mode)"
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="memory/knowledge",
        help="Path to knowledge base storage"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("MAGNET")

    logger.info("="*70)
    logger.info("🚀 MAGNET AUTONOMOUS NAVAL DESIGN RESEARCH SYSTEM")
    logger.info("="*70)
    logger.info(f"Mode: {'MOCK (CPU-only)' if args.mock else 'GPU'}")
    logger.info(f"Cycles: {args.cycles if args.cycles else 'Infinite'}")
    logger.info(f"Memory: {args.memory}")
    logger.info("="*70)

    # Create LLM client
    if args.mock:
        llm_client = MockLLMClient()
        logger.info("✓ Using MockLLMClient")
    else:
        llm_client = LocalLLMClient()
        logger.info("✓ Using LocalLLMClient (vLLM)")

    # Create physics simulator
    if args.mock:
        physics_sim = mock_physics_simulation
        logger.info("✓ Using mock physics simulation")
    else:
        try:
            from naval_domain.parallel_physics_engine import ParallelPhysicsEngine
            physics_engine = ParallelPhysicsEngine()
            physics_sim = physics_engine.simulate_designs
            logger.info("✓ Using ParallelPhysicsEngine (GPU)")
        except ImportError:
            logger.warning("⚠ GPU physics not available, falling back to mock")
            physics_sim = mock_physics_simulation

    # Create orchestrator
    orchestrator = create_orchestrator(
        llm_client=llm_client,
        physics_simulator=physics_sim,
        memory_path=args.memory,
        config={"cycle_delay": 5}
    )

    logger.info("✓ Orchestrator created")
    logger.info("\nStarting autonomous research loop...\n")

    # Run
    try:
        await orchestrator.run(max_cycles=args.cycles)
    except KeyboardInterrupt:
        logger.info("\n⏸ Interrupted by user")
        orchestrator.stop()

    logger.info("\n✅ MAGNET session complete")


if __name__ == "__main__":
    asyncio.run(main())
