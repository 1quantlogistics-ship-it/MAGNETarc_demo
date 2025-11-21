#!/usr/bin/env python3
"""
MAGNET Autonomous Research System - CLI Entry Point
====================================================

Command-line interface for running MAGNET autonomous naval design research.

Usage:
    python3 run_magnet.py --cycles 10 --mock
    python3 run_magnet.py --cycles 5 --gpu
    python3 run_magnet.py --resume
    python3 run_magnet.py --cycles 5 --mock --demo-mode --live-render

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


async def _generate_visualizations(orchestrator, args, logger):
    """Generate visualizations and optionally open in browser"""
    from datetime import datetime
    import webbrowser

    logger.info("\n📊 Generating visualizations...")

    kb = orchestrator.knowledge_base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine HTML path
    if args.export_html:
        html_path = args.export_html
    else:
        html_path = f"results/dashboard_{timestamp}.html"

    # Create results directory
    Path("results").mkdir(exist_ok=True)

    try:
        # Generate HTML dashboard
        kb.export_html_report(html_path)
        logger.info(f"  ✅ Dashboard: {html_path}")

        # Generate plots
        kb.plot_improvement_over_time(f"results/improvement_{timestamp}.png")
        logger.info(f"  ✅ Improvement plot: results/improvement_{timestamp}.png")

        kb.visualize_design_space_2d(f"results/design_space_{timestamp}.png")
        logger.info(f"  ✅ Design space plot: results/design_space_{timestamp}.png")

        kb.visualize_pareto_frontier(f"results/pareto_{timestamp}.png")
        logger.info(f"  ✅ Pareto frontier: results/pareto_{timestamp}.png")

        # Auto-open dashboard if requested
        if args.auto_open:
            logger.info(f"\n🌐 Opening dashboard in browser...")
            webbrowser.open(f"file://{Path(html_path).absolute()}")

    except Exception as e:
        logger.error(f"❌ Visualization generation failed: {e}")


def _print_metrics_report(orchestrator, logger):
    """Print comprehensive metrics report"""
    state = orchestrator.state
    kb = orchestrator.knowledge_base
    stats = kb.get_statistics()

    logger.info("\n" + "="*70)
    logger.info(" 📈 MAGNET METRICS REPORT")
    logger.info("="*70)

    # Basic stats
    logger.info(f"  Cycles completed:      {state.cycle_number}")
    logger.info(f"  Total experiments:     {state.total_experiments}")
    logger.info(f"  Valid designs:         {state.total_valid_designs} ({state.total_valid_designs/max(state.total_experiments,1)*100:.1f}%)")
    logger.info(f"  Best score:            {state.best_overall_score:.2f}")

    # Knowledge base stats
    logger.info(f"\n  KB Statistics:")
    logger.info(f"    Total experiments:   {stats['total_experiments']}")
    logger.info(f"    Designs evaluated:   {stats['total_designs_evaluated']}")
    logger.info(f"    Cycles tracked:      {stats['total_cycles']}")
    logger.info(f"    Avg overall score:   {stats['avg_overall_score']:.2f}")
    logger.info(f"    Hypotheses confirmed:{stats['successful_hypotheses']}")
    logger.info(f"    Hypotheses refuted:  {stats['failed_hypotheses']}")

    # Best designs
    best_designs = kb.get_best_designs(n=3)
    if best_designs:
        logger.info(f"\n  Top 3 Designs:")
        for i, design in enumerate(best_designs[:3], 1):
            params = design.get("parameters", {})
            results = design.get("results", {})
            logger.info(f"    {i}. Score: {results.get('overall_score', 0):.2f}")
            logger.info(f"       Length: {params.get('length_overall', 0):.1f}m, "
                       f"Beam: {params.get('beam', 0):.1f}m, "
                       f"Spacing: {params.get('hull_spacing', 0):.1f}m")

    # Errors
    if state.error_count > 0:
        logger.info(f"\n  ⚠️  Errors encountered: {state.error_count}")
        logger.info(f"      Last error: {state.last_error}")

    logger.info("="*70)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MAGNET Autonomous Naval Design Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 cycles with visualizations
  python run_magnet.py --cycles 10 --mock --visualize --auto-open

  # Demo mode with high-quality 3D rendering
  python run_magnet.py --cycles 5 --mock --demo-mode --live-render

  # Continuous monitoring with metrics
  python run_magnet.py --watch --metrics-report

  # Resume previous run
  python run_magnet.py --resume memory/orchestrator_state.json --cycles 5

  # Generate HTML dashboard
  python run_magnet.py --cycles 5 --mock --export-html results/my_dashboard.html
        """
    )

    # Core arguments
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

    # NEW: Visualization arguments
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations after run completes"
    )
    parser.add_argument(
        "--export-html",
        type=str,
        metavar="PATH",
        help="Export HTML dashboard to specified path"
    )
    parser.add_argument(
        "--auto-open",
        action="store_true",
        help="Auto-open dashboard in browser (requires --visualize or --export-html)"
    )

    # NEW: Reporting arguments
    parser.add_argument(
        "--metrics-report",
        action="store_true",
        help="Print detailed metrics summary at end"
    )

    # M48 mode argument
    parser.add_argument(
        "--m48-mode",
        action="store_true",
        help="Enable M48 mission mode (NAVSEA HC-MASC optimization with sea trial calibration)"
    )

    # NEW: State management arguments
    parser.add_argument(
        "--resume",
        type=str,
        metavar="STATE_FILE",
        help="Resume from previous state file"
    )
    parser.add_argument(
        "--save-state-every",
        type=int,
        default=1,
        metavar="N",
        help="Save state every N cycles (default: 1)"
    )

    # NEW: Monitoring arguments
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously, watching for new experiments"
    )

    # NEW: 3D Rendering and Demo arguments
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Enable enhanced 3D mesh rendering for demos (higher resolution)"
    )
    parser.add_argument(
        "--live-render",
        action="store_true",
        help="Enable real-time 3D rendering during execution"
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
    logger.info(f"M48 Mission Mode: {'ENABLED' if args.m48_mode else 'DISABLED'}")
    logger.info(f"Cycles: {args.cycles if args.cycles else 'Infinite'}")
    logger.info(f"Memory: {args.memory}")
    logger.info("="*70)

    # Print M48 configuration if enabled
    if args.m48_mode:
        logger.info("\n🎯 M48 MISSION CONFIGURATION LOADED")
        logger.info("  Platform: Magnet Defense M48 (48m catamaran)")
        logger.info("  Program: NAVSEA HC-MASC (N00024-25-R-6314)")
        logger.info("  Sea Trials: 32,000 NM validation data")
        logger.info("  Objectives: Stability (40%), Efficiency (35%), Speed (25%)")
        logger.info("  Research: Pareto frontier optimization for Navy proposal")
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
    config = {
        "cycle_delay": 5,
        "save_state_every": args.save_state_every,
        "m48_mode": args.m48_mode  # Pass M48 mode to orchestrator
    }

    # Handle resume
    if args.resume:
        if not Path(args.resume).exists():
            logger.error(f"❌ State file not found: {args.resume}")
            return
        config["state_file"] = args.resume
        logger.info(f"✓ Resuming from {args.resume}")

    orchestrator = create_orchestrator(
        llm_client=llm_client,
        physics_simulator=physics_sim,
        memory_path=args.memory,
        config=config
    )

    logger.info("✓ Orchestrator created")

    if args.resume:
        logger.info(f"   Starting at cycle {orchestrator.state.cycle_number + 1}")

    logger.info("\nStarting autonomous research loop...\n")

    # Run
    try:
        if args.watch:
            # Watch mode: run continuously
            logger.info("👀 Watch mode: Running continuously...")
            logger.info("   Press Ctrl+C to stop\n")
            import time
            while True:
                await orchestrator.run_cycle()
                if args.metrics_report:
                    _print_metrics(orchestrator)
                time.sleep(5)  # Brief pause between cycles
        else:
            # Normal mode: run specified cycles
            await orchestrator.run(max_cycles=args.cycles)

    except KeyboardInterrupt:
        logger.info("\n⏸ Interrupted by user")
        orchestrator.stop()

    # Post-run operations
    logger.info("\n✅ MAGNET session complete")

    # Generate visualizations if requested
    if args.visualize or args.export_html:
        await _generate_visualizations(orchestrator, args, logger)

    # Print metrics report if requested
    if args.metrics_report:
        _print_metrics_report(orchestrator, logger)


if __name__ == "__main__":
    asyncio.run(main())
