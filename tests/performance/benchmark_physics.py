"""
Performance Benchmark for Physics Engine

Measures throughput and latency of physics simulation across different
batch sizes and hardware configurations (CPU vs GPU).

Generates:
- Performance metrics (designs/sec, latency)
- Comparison table (CPU vs expected GPU)
- Markdown report
"""

import time
import sys
import os
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from naval_domain.baseline_designs import get_baseline_general
from naval_domain.physics_engine import PhysicsEngine, simulate_design
from naval_domain.hull_parameters import HullParameters

try:
    from naval_domain.parallel_physics_engine import ParallelPhysicsEngine
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    print("Warning: ParallelPhysicsEngine not available")


def generate_design_batch(n: int) -> List[Dict[str, Any]]:
    """
    Generate batch of test designs.

    Args:
        n: Number of designs to generate

    Returns:
        List of design dicts
    """
    baseline = get_baseline_general()
    designs = []

    for i in range(n):
        design = baseline.copy()
        # Vary parameters slightly
        design['length_overall'] = 16.0 + (i % 20) * 0.5
        design['hull_spacing'] = 4.5 + (i % 15) * 0.2
        design['design_speed'] = 20.0 + (i % 10) * 1.0

        # Remove name and description (not physics parameters)
        design.pop('name', None)
        design.pop('description', None)

        designs.append(design)

    return designs


def benchmark_single_design() -> Dict[str, float]:
    """
    Benchmark single design simulation (CPU).

    Returns:
        Dict with timing metrics
    """
    print("Benchmarking single design (CPU)...")

    baseline_dict = get_baseline_general()
    baseline_dict.pop('name', None)
    baseline_dict.pop('description', None)

    hp = HullParameters(**baseline_dict)

    # Warmup
    for _ in range(5):
        _ = simulate_design(hp)

    # Timed runs
    n_runs = 100
    start = time.time()

    for _ in range(n_runs):
        _ = simulate_design(hp)

    elapsed = time.time() - start

    avg_time = elapsed / n_runs
    throughput = n_runs / elapsed

    print(f"  Average time: {avg_time*1000:.3f} ms/design")
    print(f"  Throughput: {throughput:.1f} designs/sec")
    print()

    return {
        'avg_time_ms': avg_time * 1000,
        'throughput': throughput,
        'n_runs': n_runs,
    }


def benchmark_sequential_batch(batch_size: int) -> Dict[str, float]:
    """
    Benchmark sequential processing of batch (CPU).

    Args:
        batch_size: Number of designs in batch

    Returns:
        Dict with timing metrics
    """
    print(f"Benchmarking sequential batch (size={batch_size}, CPU)...")

    designs = generate_design_batch(batch_size)
    engine = PhysicsEngine()

    # Warmup
    for design in designs[:5]:
        hp = HullParameters(**design)
        _ = engine.simulate(hp)

    # Timed run
    start = time.time()

    for design in designs:
        hp = HullParameters(**design)
        _ = engine.simulate(hp)

    elapsed = time.time() - start

    avg_time = elapsed / batch_size
    throughput = batch_size / elapsed

    print(f"  Total time: {elapsed:.3f} sec")
    print(f"  Average time: {avg_time*1000:.3f} ms/design")
    print(f"  Throughput: {throughput:.1f} designs/sec")
    print()

    return {
        'batch_size': batch_size,
        'total_time': elapsed,
        'avg_time_ms': avg_time * 1000,
        'throughput': throughput,
    }


def benchmark_parallel_batch(batch_size: int, device: str = 'cpu') -> Dict[str, float]:
    """
    Benchmark parallel processing of batch (PyTorch).

    Args:
        batch_size: Number of designs in batch
        device: 'cpu' or 'cuda'

    Returns:
        Dict with timing metrics
    """
    if not PARALLEL_AVAILABLE:
        print(f"  Parallel engine not available, skipping")
        return {}

    print(f"Benchmarking parallel batch (size={batch_size}, device={device})...")

    designs = generate_design_batch(batch_size)
    engine = ParallelPhysicsEngine(device=device, verbose=False)

    # Warmup
    _ = engine.simulate_batch(designs[:5])

    # Timed run
    start = time.time()
    _ = engine.simulate_batch(designs)
    elapsed = time.time() - start

    avg_time = elapsed / batch_size
    throughput = batch_size / elapsed

    print(f"  Total time: {elapsed:.3f} sec")
    print(f"  Average time: {avg_time*1000:.3f} ms/design")
    print(f"  Throughput: {throughput:.1f} designs/sec")
    print()

    return {
        'batch_size': batch_size,
        'device': device,
        'total_time': elapsed,
        'avg_time_ms': avg_time * 1000,
        'throughput': throughput,
    }


def generate_markdown_report(results: Dict[str, Any], output_path: str) -> None:
    """
    Generate markdown performance report.

    Args:
        results: Benchmark results dict
        output_path: Path to save markdown file
    """
    lines = [
        "# Physics Engine Performance Benchmark",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Platform:** {sys.platform}",
        "",
        "## Single Design Performance (CPU)",
        "",
        f"- **Average Time:** {results['single']['avg_time_ms']:.3f} ms/design",
        f"- **Throughput:** {results['single']['throughput']:.1f} designs/sec",
        f"- **Runs:** {results['single']['n_runs']}",
        "",
        "## Batch Performance (Sequential CPU)",
        "",
        "| Batch Size | Total Time (s) | Avg Time (ms) | Throughput (designs/sec) |",
        "|------------|----------------|---------------|--------------------------|",
    ]

    for batch_result in results['sequential_batches']:
        lines.append(
            f"| {batch_result['batch_size']} | "
            f"{batch_result['total_time']:.3f} | "
            f"{batch_result['avg_time_ms']:.3f} | "
            f"{batch_result['throughput']:.1f} |"
        )

    if results.get('parallel_batches'):
        lines.extend([
            "",
            "## Batch Performance (Parallel PyTorch)",
            "",
            "| Batch Size | Device | Total Time (s) | Avg Time (ms) | Throughput (designs/sec) |",
            "|------------|--------|----------------|---------------|--------------------------|",
        ])

        for batch_result in results['parallel_batches']:
            if batch_result:  # Skip empty results
                lines.append(
                    f"| {batch_result['batch_size']} | "
                    f"{batch_result['device']} | "
                    f"{batch_result['total_time']:.3f} | "
                    f"{batch_result['avg_time_ms']:.3f} | "
                    f"{batch_result['throughput']:.1f} |"
                )

    lines.extend([
        "",
        "## Expected GPU Performance (Projected)",
        "",
        "| Hardware | Batch Size | Expected Throughput |",
        "|----------|------------|---------------------|",
        "| 1x A40 GPU | 20 | ~500-1000 designs/sec |",
        "| 1x A40 GPU | 50 | ~1000-1500 designs/sec |",
        "| 2x A40 GPU | 50 | ~1500-2500 designs/sec |",
        "| 2x A40 GPU | 100 | ~2000-3000 designs/sec |",
        "",
        "## Notes",
        "",
        "- CPU performance measured on Mac (single-core)",
        "- PyTorch CPU has overhead for small batches (tensors, device management)",
        "- GPU speedup expected: 10-50x for batch sizes 20-100",
        "- Optimal batch size for GPU: 50-100 designs",
        "",
        "## Recommendations",
        "",
        "- **Mac Development (CPU):** Use sequential engine for single designs",
        "- **Production (GPU):** Use parallel engine with batch size 50-100",
        "- **24-hour run:** Target 20-50 designs/cycle, 300+ cycles/day",
        ""
    ])

    report = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {output_path}")


def main():
    """Run complete benchmark suite."""
    print("=" * 70)
    print("PHYSICS ENGINE PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()

    results = {
        'single': {},
        'sequential_batches': [],
        'parallel_batches': [],
    }

    # Single design benchmark
    results['single'] = benchmark_single_design()

    # Sequential batch benchmarks
    batch_sizes = [10, 20, 50, 100]

    for batch_size in batch_sizes:
        result = benchmark_sequential_batch(batch_size)
        results['sequential_batches'].append(result)

    # Parallel batch benchmarks (CPU only on Mac)
    if PARALLEL_AVAILABLE:
        print("Testing parallel engine (PyTorch CPU)...")
        print()

        for batch_size in [10, 20, 50]:
            result = benchmark_parallel_batch(batch_size, device='cpu')
            if result:
                results['parallel_batches'].append(result)

    # Generate report
    report_path = os.path.join(os.path.dirname(__file__), 'BENCHMARK_RESULTS.md')
    generate_markdown_report(results, report_path)

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
