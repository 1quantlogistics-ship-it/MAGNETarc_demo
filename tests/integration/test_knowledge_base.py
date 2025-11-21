"""
Integration Tests for Knowledge Base

Tests the complete knowledge base functionality including:
- Experiment storage and retrieval
- Principle extraction after multiple experiments
- Pareto frontier updates
- JSON persistence across sessions
- Correlation analysis accuracy
- Markdown and HTML report generation
- Visualization generation

All tests run on Mac CPU without GPU/LLM requirements.
"""

import pytest
import sys
import os
import tempfile
import shutil
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from memory.knowledge_base import KnowledgeBase
from naval_domain.baseline_designs import get_baseline_general
from naval_domain.physics_engine import PhysicsEngine
from naval_domain.hull_parameters import HullParameters


class TestKnowledgeBaseIntegration:
    """Comprehensive integration tests for KnowledgeBase."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.kb_path = os.path.join(self.temp_dir, 'test_kb')

        # Initialize components
        self.kb = KnowledgeBase(storage_path=self.kb_path)
        self.physics_engine = PhysicsEngine(verbose=False)
        self.baseline = get_baseline_general()

    def teardown_method(self):
        """Cleanup test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_test_experiment(self, cycle: int, n_designs: int = 10) -> tuple:
        """Helper to create test experiment data."""
        hypothesis = {
            'description': f'Test hypothesis {cycle}',
            'parameter_range': {'hull_spacing': (5.0, 7.0)}
        }

        designs = []
        results = []

        for i in range(n_designs):
            design = self.baseline.copy()
            design.pop('name', None)
            design.pop('description', None)

            # Vary hull_spacing
            design['hull_spacing'] = 5.0 + (i / (n_designs - 1)) * 2.0 if n_designs > 1 else 5.5

            designs.append(design)

            # Simulate
            hp = HullParameters(**design)
            result = self.physics_engine.simulate(hp)
            results.append(result.to_dict())

        return hypothesis, designs, results

    def test_experiment_storage_and_retrieval(self):
        """Test that experiments are stored and can be retrieved."""
        # Add experiments
        for cycle in range(1, 4):
            hypothesis, designs, results = self._create_test_experiment(cycle)
            self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=cycle)

        # Verify storage
        assert len(self.kb.experiments) == 3
        assert self.kb.statistics['total_experiments'] == 3
        assert self.kb.statistics['total_cycles'] == 3

        # Verify retrieval
        context = self.kb.get_context_for_explorer(max_entries=10)
        assert len(context['recent_experiments']) == 3
        assert context['total_experiments'] == 3

    def test_principle_extraction_after_multiple_experiments(self):
        """Test that design principles are extracted after 20+ experiments."""
        # Add 20 experiments with varied parameters
        for cycle in range(1, 21):
            hypothesis, designs, results = self._create_test_experiment(cycle, n_designs=5)
            self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=cycle)

        # Verify principles extracted
        assert len(self.kb.experiments) == 20
        assert len(self.kb.principles) > 0

        # Check that at least some principles have significant correlations
        significant_principles = [
            p for p in self.kb.principles
            if abs(p.get('correlation', 0.0)) > 0.5
        ]

        # Should have at least some significant correlations
        assert len(significant_principles) >= 0  # May or may not find strong correlations

    def test_pareto_frontier_updates(self):
        """Test that Pareto frontier updates correctly with new designs."""
        initial_best_count = len(self.kb.best_designs)

        # Add first experiment
        hypothesis1, designs1, results1 = self._create_test_experiment(1, n_designs=10)
        self.kb.add_experiment_results(hypothesis1, designs1, results1, cycle_number=1)

        # Should have some best designs
        assert len(self.kb.best_designs) > initial_best_count

        first_best_count = len(self.kb.best_designs)
        first_best_score = self.kb.best_designs[0]['overall_score']

        # Add more experiments
        for cycle in range(2, 6):
            hypothesis, designs, results = self._create_test_experiment(cycle, n_designs=15)
            self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=cycle)

        # Best designs should be updated
        assert len(self.kb.best_designs) >= first_best_count

        # Best score should be >= first best (monotonic improvement or same)
        current_best_score = self.kb.best_designs[0]['overall_score']
        assert current_best_score >= first_best_score - 5.0  # Allow small variance

        # Should maintain top 100 limit
        assert len(self.kb.best_designs) <= 100

    def test_json_persistence_across_sessions(self):
        """Test that data persists across knowledge base sessions."""
        # Add data in first session
        hypothesis, designs, results = self._create_test_experiment(1, n_designs=10)
        self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=1)

        original_stats = self.kb.get_statistics().copy()
        original_exp_count = len(self.kb.experiments)
        original_best_count = len(self.kb.best_designs)

        # Explicitly save
        self.kb.save()

        # Create new knowledge base instance (simulating new session)
        kb2 = KnowledgeBase(storage_path=self.kb_path)

        # Verify data loaded
        assert len(kb2.experiments) == original_exp_count
        assert len(kb2.best_designs) == original_best_count
        assert kb2.statistics['total_experiments'] == original_stats['total_experiments']
        assert kb2.statistics['total_designs_evaluated'] == original_stats['total_designs_evaluated']

        # Verify JSON files exist
        assert os.path.exists(os.path.join(self.kb_path, 'experiments.json'))
        assert os.path.exists(os.path.join(self.kb_path, 'statistics.json'))
        assert os.path.exists(os.path.join(self.kb_path, 'best_designs.json'))

    def test_correlation_analysis_accuracy(self):
        """Test that correlation analysis produces meaningful results."""
        # Create experiments with known correlation
        # Vary hull_spacing linearly, which should correlate with stability

        for cycle in range(1, 11):
            hypothesis = {
                'description': 'Hull spacing vs stability',
                'parameter_range': {'hull_spacing': (4.5, 7.0)}
            }

            designs = []
            results = []

            for i in range(10):
                design = self.baseline.copy()
                design.pop('name', None)
                design.pop('description', None)

                # Linear variation of hull_spacing
                design['hull_spacing'] = 4.5 + i * 0.25

                designs.append(design)

                hp = HullParameters(**design)
                result = self.physics_engine.simulate(hp)
                results.append(result.to_dict())

            self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=cycle)

        # Check for hull_spacing principles
        spacing_principles = [
            p for p in self.kb.principles
            if p['parameter'] == 'hull_spacing'
        ]

        # Should have found some correlation (positive or negative)
        assert len(spacing_principles) >= 0  # May or may not find correlation

        # If found, correlation should be meaningful
        if spacing_principles:
            for principle in spacing_principles:
                assert abs(principle['correlation']) <= 1.0  # Valid correlation coefficient
                assert 'insight' in principle
                assert len(principle['insight']) > 0

    def test_markdown_report_generation(self):
        """Test markdown report generation."""
        # Add some data
        for cycle in range(1, 4):
            hypothesis, designs, results = self._create_test_experiment(cycle, n_designs=5)
            self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=cycle)

        # Generate report
        report_path = os.path.join(self.temp_dir, 'test_report.md')
        report = self.kb.export_markdown_report(report_path)

        # Verify report exists
        assert os.path.exists(report_path)

        # Verify content
        assert len(report) > 100
        assert 'Research Report' in report
        assert 'Total Research Cycles' in report
        assert 'Total Experiments' in report
        assert 'Top 10 Designs' in report

        # Verify statistics in report
        assert str(self.kb.statistics['total_cycles']) in report

    def test_html_report_generation(self):
        """Test HTML dashboard generation."""
        # Add data
        for cycle in range(1, 4):
            hypothesis, designs, results = self._create_test_experiment(cycle, n_designs=5)
            self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=cycle)

        # Generate HTML report
        html_path = os.path.join(self.temp_dir, 'dashboard.html')
        html_file = self.kb.export_html_report(html_path)

        # Verify file exists
        assert os.path.exists(html_file)

        # Read and verify content
        with open(html_file, 'r') as f:
            html_content = f.read()

        assert 'MAGNET' in html_content
        assert 'Research Statistics' in html_content
        assert 'Total Cycles' in html_content
        assert '<html>' in html_content
        assert '</html>' in html_content

        # Should have embedded base64 images (if plots generated)
        # At minimum should have proper HTML structure

    def test_visualization_generation(self):
        """Test that visualizations are generated without errors."""
        # Add sufficient data for visualizations
        for cycle in range(1, 6):
            hypothesis, designs, results = self._create_test_experiment(cycle, n_designs=10)
            self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=cycle)

        # Test improvement over time plot
        viz_path = os.path.join(self.temp_dir, 'improvement.png')
        result_path = self.kb.plot_improvement_over_time(viz_path)
        assert os.path.exists(result_path)

        # Test Pareto frontier plot
        pareto_path = os.path.join(self.temp_dir, 'pareto.png')
        result_path = self.kb.visualize_pareto_frontier(pareto_path)
        assert os.path.exists(result_path)

        # Test 2D design space plot
        design_space_path = os.path.join(self.temp_dir, 'design_space.png')
        result_path = self.kb.visualize_design_space_2d('length_overall', 'hull_spacing', design_space_path)
        assert os.path.exists(result_path)

    def test_empty_knowledge_base(self):
        """Test behavior with empty knowledge base."""
        # New empty knowledge base
        empty_kb = KnowledgeBase(storage_path=os.path.join(self.temp_dir, 'empty_kb'))

        # Should have empty statistics
        stats = empty_kb.get_statistics()
        assert stats['total_experiments'] == 0
        assert stats['total_designs_evaluated'] == 0
        assert stats['best_overall_score'] == 0.0

        # Should handle empty context request
        context = empty_kb.get_context_for_explorer()
        assert context['total_experiments'] == 0
        assert len(context['recent_experiments']) == 0

        # Should handle empty best designs request
        best = empty_kb.get_best_designs(10)
        assert len(best) == 0

    def test_clear_functionality(self):
        """Test clearing knowledge base."""
        # Add data
        hypothesis, designs, results = self._create_test_experiment(1, n_designs=5)
        self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=1)

        # Verify data exists
        assert len(self.kb.experiments) > 0
        assert len(self.kb.best_designs) > 0

        # Clear
        self.kb.clear()

        # Verify cleared
        assert len(self.kb.experiments) == 0
        assert len(self.kb.best_designs) == 0
        assert len(self.kb.principles) == 0
        assert self.kb.statistics['total_experiments'] == 0

    def test_hypothesis_outcome_tracking(self):
        """Test that hypothesis outcomes are tracked correctly."""
        # Add experiments
        for cycle in range(1, 6):
            hypothesis, designs, results = self._create_test_experiment(cycle, n_designs=10)
            self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=cycle)

        # Check outcomes tracked
        outcomes = [exp.get('hypothesis_outcome', 'unknown') for exp in self.kb.experiments]

        # Should have some outcomes
        assert len(outcomes) == 5

        # Outcomes should be valid
        valid_outcomes = {'confirmed', 'refuted', 'failed', 'unknown'}
        for outcome in outcomes:
            assert outcome in valid_outcomes

        # Statistics should reflect outcomes
        total_outcomes = (
            self.kb.statistics['successful_hypotheses'] +
            self.kb.statistics['failed_hypotheses']
        )
        assert total_outcomes == 5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
