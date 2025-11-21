"""
System Validation Test Suite for MAGNET v0.1.0

End-to-end system validation tests covering:
- 5-cycle autonomous completion
- Knowledge base population and persistence
- Visualization generation
- Metrics tracking
- Memory stability
- State persistence and recovery

These tests validate the entire MAGNET system is ready for v0.1.0 release.
"""

import pytest
import sys
import os
import tempfile
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from memory.knowledge_base import KnowledgeBase
from memory.metrics_tracker import MetricsTracker
from naval_domain.baseline_designs import get_baseline_general
from naval_domain.physics_engine import PhysicsEngine
from naval_domain.hull_parameters import HullParameters


class TestSystemValidation:
    """End-to-end system validation tests for v0.1.0."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary directory for this test
        self.temp_dir = tempfile.mkdtemp()
        self.kb_path = os.path.join(self.temp_dir, 'knowledge')
        self.metrics_path = os.path.join(self.temp_dir, 'metrics')
        self.results_path = os.path.join(self.temp_dir, 'results')

        # Create results directory
        os.makedirs(self.results_path, exist_ok=True)

        # Initialize components
        self.kb = KnowledgeBase(storage_path=self.kb_path)
        self.metrics = MetricsTracker(storage_path=self.metrics_path)
        self.physics_engine = PhysicsEngine(verbose=False)
        self.baseline = get_baseline_general()

    def teardown_method(self):
        """Cleanup test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _simulate_research_cycle(self, cycle_number: int, n_designs: int = 10) -> Dict[str, Any]:
        """
        Simulate a single autonomous research cycle.

        Args:
            cycle_number: Current cycle number
            n_designs: Number of designs to evaluate

        Returns:
            Cycle results dictionary
        """
        # Start cycle timing
        start_time = self.metrics.start_cycle(cycle_number)

        # Mock hypothesis generation
        hypothesis = {
            'cycle': cycle_number,
            'description': f'Test hypothesis {cycle_number}',
            'parameter_range': {'hull_spacing': (5.0, 7.0)}
        }

        # Mock experiment design (vary hull_spacing)
        designs = []
        for i in range(n_designs):
            design = self.baseline.copy()
            design.pop('name', None)
            design.pop('description', None)
            design['hull_spacing'] = 5.0 + (i / (n_designs - 1)) * 2.0 if n_designs > 1 else 5.5
            designs.append(design)

        # Physics simulation
        physics_start = time.time()
        results = []
        for design in designs:
            hp = HullParameters(**design)
            result = self.physics_engine.simulate(hp)
            results.append(result.to_dict())
        physics_time = time.time() - physics_start

        # Record metrics
        self.metrics.record_agent_time('explorer', 0.1)
        self.metrics.record_agent_time('architect', 0.2)
        self.metrics.record_agent_time('critic', 0.1)
        self.metrics.record_physics_time(n_designs, physics_time, device='cpu')

        # Update knowledge base
        self.kb.add_experiment_results(hypothesis, designs, results, cycle_number)

        # End cycle timing
        self.metrics.end_cycle(start_time, n_designs)

        return {
            'cycle': cycle_number,
            'hypothesis': hypothesis,
            'n_designs': n_designs,
            'results': results,
            'physics_time': physics_time
        }

    def test_five_cycle_completion(self):
        """
        Test that 5 autonomous cycles complete successfully.

        Validates:
        - All cycles execute without errors
        - Each cycle produces results
        - No crashes or exceptions
        - Reasonable execution time
        """
        n_cycles = 5
        cycle_results = []

        start_time = time.time()

        for cycle in range(1, n_cycles + 1):
            result = self._simulate_research_cycle(cycle, n_designs=10)
            cycle_results.append(result)

            # Validate each cycle
            assert result is not None
            assert result['cycle'] == cycle
            assert result['n_designs'] == 10
            assert len(result['results']) == 10
            assert all(r['is_valid'] for r in result['results'])

        total_time = time.time() - start_time

        # Validate overall execution
        assert len(cycle_results) == n_cycles
        assert total_time < 60.0  # Should complete in under 60 seconds

        # Verify all agents executed (mock timing)
        summary = self.metrics.get_performance_summary()
        assert summary['total_cycles'] == n_cycles
        assert summary['total_designs_simulated'] == n_cycles * 10

    def test_knowledge_base_population(self):
        """
        Verify knowledge base accumulates data correctly across cycles.

        Validates:
        - Experiments stored correctly
        - Principles extracted from patterns
        - Best designs tracked
        - Statistics updated properly
        """
        n_cycles = 5

        # Run cycles
        for cycle in range(1, n_cycles + 1):
            self._simulate_research_cycle(cycle, n_designs=10)

        # Verify experiments stored
        assert len(self.kb.experiments) == n_cycles
        assert self.kb.statistics['total_experiments'] == n_cycles
        assert self.kb.statistics['total_designs_evaluated'] == n_cycles * 10

        # Verify best designs tracked
        assert len(self.kb.best_designs) > 0
        assert len(self.kb.best_designs) <= 100  # Max 100 designs

        # Verify scores are reasonable
        assert self.kb.statistics['best_overall_score'] > 0.0
        assert self.kb.statistics['avg_overall_score'] > 0.0

        # Verify principles may be extracted
        # (Not guaranteed with only 5 cycles of random data)
        assert isinstance(self.kb.principles, list)

        # Verify context generation works
        context = self.kb.get_context_for_explorer(max_entries=10)
        assert len(context['recent_experiments']) == n_cycles
        assert context['total_experiments'] == n_cycles
        assert 'statistics' in context
        assert 'best_designs' in context

    def test_visualization_generation(self):
        """
        Test that all visualizations generate without errors.

        Validates:
        - HTML dashboard generation
        - Improvement over time plot
        - Design space 2D plot
        - Pareto frontier plot
        - Files created with reasonable sizes
        """
        # Populate knowledge base with data
        for cycle in range(1, 6):
            self._simulate_research_cycle(cycle, n_designs=10)

        # Test HTML dashboard
        html_path = os.path.join(self.results_path, 'dashboard.html')
        result_path = self.kb.export_html_report(html_path)

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 1000  # At least 1KB

        # Verify HTML content
        with open(result_path, 'r') as f:
            html_content = f.read()
            assert 'MAGNET' in html_content
            assert 'Research Statistics' in html_content
            assert '<html>' in html_content

        # Test improvement over time plot
        improvement_path = os.path.join(self.results_path, 'improvement.png')
        result_path = self.kb.plot_improvement_over_time(improvement_path)

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 5000  # At least 5KB for PNG

        # Test design space 2D plot
        design_space_path = os.path.join(self.results_path, 'design_space.png')
        result_path = self.kb.visualize_design_space_2d(
            design_space_path,
            param_x='length_overall',
            param_y='hull_spacing'
        )

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 5000

        # Test Pareto frontier plot
        pareto_path = os.path.join(self.results_path, 'pareto.png')
        result_path = self.kb.visualize_pareto_frontier(pareto_path)

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 5000

    def test_metrics_tracking(self):
        """
        Validate metrics tracker captures all performance data.

        Validates:
        - Cycle timing tracked
        - Agent latency recorded
        - Physics throughput measured
        - Success rates calculated
        - Summary generation works
        """
        # Run cycles with metrics
        for cycle in range(1, 6):
            self._simulate_research_cycle(cycle, n_designs=10)

        # Get performance summary
        summary = self.metrics.get_performance_summary()

        # Verify cycle metrics
        assert summary['total_cycles'] == 5
        assert summary['total_designs_simulated'] == 50
        assert 'avg_cycle_time' in summary
        assert summary['avg_cycle_time'] > 0.0

        # Verify agent latency tracked
        assert 'agent_latency' in summary
        for agent in ['explorer', 'architect', 'critic']:
            assert agent in summary['agent_latency']
            assert summary['agent_latency'][agent]['calls'] == 5
            assert summary['agent_latency'][agent]['avg'] > 0.0

        # Verify physics throughput
        assert 'physics_throughput' in summary
        assert summary['physics_throughput']['avg'] > 0.0
        assert summary['physics_throughput']['designs_per_second'] > 0.0

        # Test metrics report generation
        report_path = os.path.join(self.results_path, 'metrics_report.md')
        result = self.metrics.export_metrics_report(report_path)

        assert os.path.exists(result)

        with open(result, 'r') as f:
            content = f.read()
            assert 'Performance Metrics Report' in content
            assert 'Cycle Performance' in content
            assert 'Agent Latency' in content

    def test_memory_stability(self):
        """
        Ensure no memory leaks during extended runs.

        Validates:
        - Memory usage doesn't grow unbounded
        - Resources properly released
        - No accumulation of temp files
        """
        import psutil
        import gc

        process = psutil.Process()

        # Get initial memory
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run 10 cycles
        for cycle in range(1, 11):
            self._simulate_research_cycle(cycle, n_designs=10)

            # Force garbage collection periodically
            if cycle % 3 == 0:
                gc.collect()

        # Get final memory
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_growth = final_memory - initial_memory

        # Memory should not grow excessively (allow up to 50MB growth)
        # This is generous to account for normal Python memory behavior
        assert memory_growth < 50.0, f"Memory grew by {memory_growth:.1f}MB (too much)"

        # Verify no temp files accumulated
        temp_files = list(Path(self.temp_dir).rglob('*.tmp'))
        assert len(temp_files) == 0, "Temp files not cleaned up"

    def test_state_persistence(self):
        """
        Test that system state persists correctly across sessions.

        Validates:
        - Knowledge base persists to disk
        - Metrics persist to disk
        - State can be restored
        - Data integrity maintained
        """
        # Run 3 cycles
        for cycle in range(1, 4):
            self._simulate_research_cycle(cycle, n_designs=5)

        # Save state explicitly
        self.kb.save()
        self.metrics.save()

        # Record current state
        original_exp_count = len(self.kb.experiments)
        original_designs_count = self.kb.statistics['total_designs_evaluated']
        original_cycles = self.metrics.get_performance_summary()['total_cycles']

        # Verify files exist
        assert os.path.exists(os.path.join(self.kb_path, 'experiments.json'))
        assert os.path.exists(os.path.join(self.kb_path, 'statistics.json'))
        assert os.path.exists(os.path.join(self.metrics_path, 'metrics.json'))

        # Simulate system restart by creating new instances
        kb2 = KnowledgeBase(storage_path=self.kb_path)
        metrics2 = MetricsTracker(storage_path=self.metrics_path)

        # Verify data loaded correctly
        assert len(kb2.experiments) == original_exp_count
        assert kb2.statistics['total_designs_evaluated'] == original_designs_count

        summary2 = metrics2.get_performance_summary()
        assert summary2['total_cycles'] == original_cycles

        # Continue with more cycles
        physics_engine2 = PhysicsEngine(verbose=False)
        baseline2 = get_baseline_general()

        for cycle in range(4, 6):
            # Mock cycle with new instances
            hypothesis = {
                'cycle': cycle,
                'description': f'Test hypothesis {cycle}',
                'parameter_range': {'hull_spacing': (5.0, 7.0)}
            }

            designs = []
            for i in range(5):
                design = baseline2.copy()
                design.pop('name', None)
                design.pop('description', None)
                design['hull_spacing'] = 5.0 + i * 0.4
                designs.append(design)

            results = []
            for design in designs:
                hp = HullParameters(**design)
                result = physics_engine2.simulate(hp)
                results.append(result.to_dict())

            kb2.add_experiment_results(hypothesis, designs, results, cycle)

        # Verify cumulative state
        assert len(kb2.experiments) == 5  # 3 original + 2 new
        assert kb2.statistics['total_experiments'] == 5

    def test_error_recovery(self):
        """
        Test system handles errors gracefully.

        Validates:
        - Invalid designs handled
        - Empty results handled
        - Partial failures don't crash system
        """
        # Test with invalid design (will fail HullParameters validation)
        invalid_design = self.baseline.copy()
        invalid_design.pop('name', None)
        invalid_design.pop('description', None)
        invalid_design['length_overall'] = 5.0  # Too small

        with pytest.raises(ValueError):
            hp = HullParameters(**invalid_design)

        # Test with empty experiment
        hypothesis = {'description': 'Empty test', 'parameter_range': {}}
        self.kb.add_experiment_results(hypothesis, [], [], cycle_number=1)

        # Should not crash
        stats = self.kb.get_statistics()
        assert stats['total_experiments'] == 1
        assert stats['total_designs_evaluated'] == 0

        # Test visualization with no data should raise proper error
        kb_empty = KnowledgeBase(storage_path=os.path.join(self.temp_dir, 'empty_kb'))

        with pytest.raises(ValueError, match="No experiments available"):
            kb_empty.plot_improvement_over_time("test.png")

        with pytest.raises(ValueError, match="No designs available"):
            kb_empty.visualize_pareto_frontier("test.png")


class TestIntegrationReadiness:
    """Additional integration readiness checks for v0.1.0."""

    def test_all_imports_working(self):
        """Verify all critical imports work without errors."""
        try:
            from memory import KnowledgeBase, MetricsTracker
            from naval_domain.physics_engine import PhysicsEngine
            from naval_domain.hull_parameters import HullParameters
            from naval_domain.baseline_designs import get_baseline_general

            # Verify classes instantiate
            kb = KnowledgeBase()
            mt = MetricsTracker()
            pe = PhysicsEngine()
            baseline = get_baseline_general()

            assert kb is not None
            assert mt is not None
            assert pe is not None
            assert baseline is not None

        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_baseline_designs_valid(self):
        """Verify baseline designs are valid and simulate correctly."""
        from naval_domain.baseline_designs import get_all_baselines

        physics = PhysicsEngine(verbose=False)
        baselines = get_all_baselines()

        assert len(baselines) >= 1

        for baseline in baselines:
            # Remove metadata fields
            design = baseline.copy()
            design.pop('name', None)
            design.pop('description', None)

            # Should create valid HullParameters
            hp = HullParameters(**design)

            # Should simulate successfully
            result = physics.simulate(hp)

            assert result.is_valid
            assert result.overall_score > 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
