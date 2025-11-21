#!/usr/bin/env python3
"""
Dynamic 3D Rendering Demo for MAGNET
=====================================

Interactive demo that showcases the complete MAGNET pipeline:
- Real-time design generation
- Parametric 3D mesh creation
- Physics simulation with performance metrics
- Live visualization streaming via WebSocket

Author: Agent 2
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

# MAGNET imports
from naval_domain.hull_parameters import HullParameters, get_baseline_catamaran, get_high_speed_catamaran
from naval_domain.physics_engine import simulate_design


class DynamicRenderingDemo:
    """
    Orchestrates real-time design generation and visualization for demos.

    Features:
    - Generates designs with parametric variations
    - Creates 3D meshes in real-time
    - Streams to viewer via existing WebSocket infrastructure
    - Displays live performance metrics
    - Records design evolution history
    """

    def __init__(
        self,
        base_design: Optional[HullParameters] = None,
        output_dir: str = "outputs/demo",
        demo_mode: bool = True
    ):
        """
        Initialize demo orchestrator.

        Args:
            base_design: Starting design (uses baseline if None)
            output_dir: Directory for demo outputs
            demo_mode: Enable high-quality rendering
        """
        self.base_design = base_design or self._get_baseline_design()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.demo_mode = demo_mode

        self.design_history: List[Dict[str, Any]] = []
        self.current_design_id: Optional[str] = None

    def _get_baseline_design(self) -> HullParameters:
        """Get baseline catamaran design."""
        return get_baseline_catamaran()

    def generate_design_variation(
        self,
        variation_type: str = "length",
        variation_amount: float = 0.1
    ) -> HullParameters:
        """
        Generate design variation from current base.

        Args:
            variation_type: Parameter to vary (length, beam, spacing, etc.)
            variation_amount: Fractional variation (0.1 = 10% change)

        Returns:
            New design with variation applied
        """
        from dataclasses import replace

        # Apply variation based on type
        if variation_type == "length":
            return replace(
                self.base_design,
                length_overall=self.base_design.length_overall * (1.0 + variation_amount)
            )
        elif variation_type == "beam":
            return replace(
                self.base_design,
                beam=self.base_design.beam * (1.0 + variation_amount),
                waterline_beam=self.base_design.waterline_beam * (1.0 + variation_amount)
            )
        elif variation_type == "spacing":
            return replace(
                self.base_design,
                hull_spacing=self.base_design.hull_spacing * (1.0 + variation_amount)
            )
        elif variation_type == "depth":
            return replace(
                self.base_design,
                hull_depth=self.base_design.hull_depth * (1.0 + variation_amount)
            )
        else:
            return self.base_design

    def simulate_and_render(
        self,
        design: HullParameters,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Simulate design and generate 3D mesh.

        Args:
            design: Design to simulate
            verbose: Print progress messages

        Returns:
            Dictionary with design_id, results, and mesh_path
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Simulating: {design.name}")
            print(f"Design ID: {design.design_id}")
            print(f"{'='*70}")

        # Run physics simulation with mesh generation
        start_time = time.time()
        results = simulate_design(
            design,
            verbose=verbose,
            generate_mesh=True,  # Always generate mesh for demo
            demo_mode=True       # Use higher resolution for better visuals
        )
        sim_time = time.time() - start_time

        if verbose:
            print(f"\n✓ Simulation complete in {sim_time:.2f}s")
            if results.is_valid:
                print(f"  Overall Score: {results.overall_score:.1f}")
                print(f"  Stability: {results.stability_score:.1f}")
                print(f"  Speed: {results.speed_score:.1f}")
                print(f"  Efficiency: {results.efficiency_score:.1f}")
            else:
                print(f"  ❌ Invalid design: {results.failure_reasons}")

        # Store in history
        design_record = {
            "design_id": design.design_id,
            "name": design.name,
            "timestamp": datetime.now().isoformat(),
            "parameters": design.to_dict(),
            "results": results.to_dict(),
            "simulation_time": sim_time,
            "mesh_path": results.mesh_path if hasattr(results, 'mesh_path') else None
        }

        self.design_history.append(design_record)
        self.current_design_id = design.design_id

        return design_record

    def run_parametric_sweep(
        self,
        parameter: str = "length",
        min_variation: float = -0.2,
        max_variation: float = 0.2,
        num_steps: int = 5,
        delay_seconds: float = 2.0
    ):
        """
        Run parametric sweep and generate designs in real-time.

        Args:
            parameter: Parameter to sweep
            min_variation: Minimum variation (e.g., -0.2 = -20%)
            max_variation: Maximum variation (e.g., 0.2 = +20%)
            num_steps: Number of designs to generate
            delay_seconds: Delay between designs (for visualization)
        """
        print(f"\n{'='*70}")
        print(f"PARAMETRIC SWEEP DEMO: {parameter.upper()}")
        print(f"{'='*70}")
        print(f"Generating {num_steps} designs...")
        print(f"Variation range: {min_variation*100:.0f}% to {max_variation*100:.0f}%")
        print(f"")

        variations = [
            min_variation + (max_variation - min_variation) * (i / (num_steps - 1))
            for i in range(num_steps)
        ]

        for i, variation in enumerate(variations):
            print(f"\n[{i+1}/{num_steps}] Variation: {variation*100:+.1f}%")

            # Generate design
            design = self.generate_design_variation(parameter, variation)

            # Simulate and render
            self.simulate_and_render(design, verbose=True)

            # Delay for visualization (except last iteration)
            if i < num_steps - 1 and delay_seconds > 0:
                print(f"\nWaiting {delay_seconds}s for visualization...")
                time.sleep(delay_seconds)

        print(f"\n{'='*70}")
        print(f"SWEEP COMPLETE: {num_steps} designs generated")
        print(f"{'='*70}")

    def run_comparison_demo(
        self,
        design_a: HullParameters,
        design_b: HullParameters,
        delay_seconds: float = 3.0
    ):
        """
        Generate two designs for side-by-side comparison.

        Args:
            design_a: First design
            design_b: Second design
            delay_seconds: Delay between generations
        """
        print(f"\n{'='*70}")
        print(f"COMPARISON DEMO")
        print(f"{'='*70}")

        print(f"\nDesign A: {design_a.name}")
        record_a = self.simulate_and_render(design_a, verbose=True)

        if delay_seconds > 0:
            print(f"\nWaiting {delay_seconds}s...")
            time.sleep(delay_seconds)

        print(f"\nDesign B: {design_b.name}")
        record_b = self.simulate_and_render(design_b, verbose=True)

        # Print comparison
        print(f"\n{'='*70}")
        print(f"COMPARISON RESULTS")
        print(f"{'='*70}")

        if record_a["results"]["is_valid"] and record_b["results"]["is_valid"]:
            score_a = record_a["results"]["overall_score"]
            score_b = record_b["results"]["overall_score"]
            delta = score_b - score_a

            print(f"\nDesign A Score: {score_a:.1f}")
            print(f"Design B Score: {score_b:.1f}")
            print(f"Delta: {delta:+.1f} ({delta/score_a*100:+.1f}%)")

            if delta > 0:
                print(f"\n✓ Design B is superior by {delta:.1f} points")
            elif delta < 0:
                print(f"\n✓ Design A is superior by {abs(delta):.1f} points")
            else:
                print(f"\n✓ Designs have equal performance")

        return record_a, record_b

    def export_summary(self, output_path: Optional[str] = None):
        """
        Export demo summary with all designs and results.

        Args:
            output_path: Path for summary JSON (auto-generated if None)
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"demo_summary_{timestamp}.json"

        summary = {
            "demo_info": {
                "timestamp": datetime.now().isoformat(),
                "total_designs": len(self.design_history),
                "demo_mode": self.demo_mode
            },
            "base_design": self.base_design.to_dict(),
            "design_history": self.design_history
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Demo summary exported to: {output_path}")

        return output_path

    def print_summary(self):
        """Print demo execution summary."""
        print(f"\n{'='*70}")
        print(f"DEMO SUMMARY")
        print(f"{'='*70}")
        print(f"Total designs generated: {len(self.design_history)}")

        if self.design_history:
            valid_designs = [d for d in self.design_history if d["results"]["is_valid"]]
            print(f"Valid designs: {len(valid_designs)}/{len(self.design_history)}")

            if valid_designs:
                scores = [d["results"]["overall_score"] for d in valid_designs]
                print(f"Best score: {max(scores):.1f}")
                print(f"Average score: {sum(scores)/len(scores):.1f}")

                avg_sim_time = sum(d["simulation_time"] for d in self.design_history) / len(self.design_history)
                print(f"Average simulation time: {avg_sim_time:.2f}s")

        print(f"{'='*70}")


# Example demo scenarios

def demo_length_sweep():
    """Demo: Sweep vessel length."""
    demo = DynamicRenderingDemo()
    demo.run_parametric_sweep(
        parameter="length",
        min_variation=-0.15,  # -15%
        max_variation=0.15,   # +15%
        num_steps=5,
        delay_seconds=2.0
    )
    demo.print_summary()
    demo.export_summary()


def demo_beam_sweep():
    """Demo: Sweep vessel beam."""
    demo = DynamicRenderingDemo()
    demo.run_parametric_sweep(
        parameter="beam",
        min_variation=-0.1,  # -10%
        max_variation=0.1,   # +10%
        num_steps=4,
        delay_seconds=2.0
    )
    demo.print_summary()
    demo.export_summary()


def demo_spacing_sweep():
    """Demo: Sweep hull spacing."""
    demo = DynamicRenderingDemo()
    demo.run_parametric_sweep(
        parameter="spacing",
        min_variation=-0.2,  # -20%
        max_variation=0.2,   # +20%
        num_steps=6,
        delay_seconds=2.0
    )
    demo.print_summary()
    demo.export_summary()


def demo_comparison():
    """Demo: Compare two designs."""
    demo = DynamicRenderingDemo()

    # Design A: Baseline
    design_a = demo._get_baseline_design()
    design_a.name = "Baseline Catamaran"

    # Design B: Longer, narrower variant
    design_b = demo.generate_design_variation("length", 0.15)
    design_b.beam *= 0.95  # Slightly narrower
    design_b.name = "High-Speed Variant"

    demo.run_comparison_demo(design_a, design_b, delay_seconds=3.0)
    demo.print_summary()
    demo.export_summary()


def main():
    """Main entry point - run interactive demo."""
    print(f"\n{'='*70}")
    print(f"MAGNET DYNAMIC 3D RENDERING DEMO")
    print(f"{'='*70}")
    print(f"\nAvailable demos:")
    print(f"  1. Length sweep (-15% to +15%)")
    print(f"  2. Beam sweep (-10% to +10%)")
    print(f"  3. Hull spacing sweep (-20% to +20%)")
    print(f"  4. Design comparison (baseline vs high-speed)")
    print(f"  5. Quick test (3 designs)")
    print(f"")

    try:
        choice = input("Select demo (1-5, or 'all' to run all): ").strip().lower()

        if choice == '1':
            demo_length_sweep()
        elif choice == '2':
            demo_beam_sweep()
        elif choice == '3':
            demo_spacing_sweep()
        elif choice == '4':
            demo_comparison()
        elif choice == '5':
            # Quick test
            demo = DynamicRenderingDemo()
            demo.run_parametric_sweep(
                parameter="length",
                min_variation=-0.1,
                max_variation=0.1,
                num_steps=3,
                delay_seconds=1.5
            )
            demo.print_summary()
        elif choice == 'all':
            print("\nRunning all demos...")
            demo_length_sweep()
            demo_beam_sweep()
            demo_spacing_sweep()
            demo_comparison()
        else:
            print("Invalid choice. Running quick test...")
            demo = DynamicRenderingDemo()
            demo.run_parametric_sweep(num_steps=3)
            demo.print_summary()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
