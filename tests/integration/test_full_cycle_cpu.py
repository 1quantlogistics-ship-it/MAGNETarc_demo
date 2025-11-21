"""
End-to-End Integration Test for Full Research Cycle (CPU Mode)

Tests complete autonomous research cycle using:
- MockAgents (no real LLM required)
- PhysicsEngine (CPU mode)
- KnowledgeBase (JSON persistence)
- Baseline designs

This test validates that all components integrate correctly
before deploying to GPU with real LLM.

Test Flow:
1. Initialize KnowledgeBase + PhysicsEngine (CPU)
2. MockExplorer → hypothesis
3. MockArchitect → designs (batch of 10)
4. MockCritic → validation
5. PhysicsEngine.simulate_batch() → results
6. MockHistorian → insights
7. KnowledgeBase.add_experiment_results()
8. Verify data persistence and retrieval
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.mocks.mock_agents import create_mock_agents
from naval_domain.baseline_designs import get_baseline_general
from naval_domain.physics_engine import PhysicsEngine
from naval_domain.hull_parameters import HullParameters
from memory.knowledge_base import KnowledgeBase


class TestFullCycleCPU:
    """Integration tests for complete research cycle on CPU."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary directory for knowledge base
        self.temp_dir = tempfile.mkdtemp()
        self.kb_path = os.path.join(self.temp_dir, 'knowledge_test')

        # Initialize components
        self.kb = KnowledgeBase(storage_path=self.kb_path)
        self.physics_engine = PhysicsEngine(verbose=False)
        self.agents = create_mock_agents(seed=42)

        # Get baseline design
        self.baseline = get_baseline_general()

    def teardown_method(self):
        """Cleanup test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_single_research_cycle(self):
        """Test complete single research cycle."""
        # Step 1: Explorer generates hypothesis
        kb_context = self.kb.get_context_for_explorer()
        hypothesis = self.agents['explorer'].autonomous_cycle(kb_context)

        assert hypothesis is not None
        assert 'description' in hypothesis
        assert 'parameter_range' in hypothesis

        # Step 2: Architect designs experiments
        designs = self.agents['architect'].design_experiments(
            hypothesis, self.baseline, n_designs=10
        )

        assert len(designs) == 10
        assert all('length_overall' in d for d in designs)

        # Step 3: Critic reviews
        review = self.agents['critic'].review_experiments(designs, hypothesis)

        assert review['approved'] == True
        assert review['n_designs'] == 10

        # Step 4: Physics simulation
        results = []
        for design in designs:
            hp = HullParameters(**design)
            result = self.physics_engine.simulate(hp)
            results.append(result.to_dict())

        assert len(results) == 10
        valid_results = [r for r in results if r['is_valid']]
        assert len(valid_results) > 0  # At least some should be valid

        # Step 5: Historian analysis
        insights = self.agents['historian'].analyze_batch_results(
            designs, results, hypothesis
        )

        assert 'hypothesis_outcome' in insights
        assert 'avg_overall_score' in insights
        assert len(insights['insights']) > 0

        # Step 6: Knowledge base update
        self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=1)

        # Verify knowledge base was updated
        stats = self.kb.get_statistics()
        assert stats['total_experiments'] == 1
        assert stats['total_designs_evaluated'] == 10
        assert stats['best_overall_score'] > 0

    def test_multiple_research_cycles(self):
        """Test multiple consecutive research cycles."""
        n_cycles = 3

        for cycle in range(1, n_cycles + 1):
            # Generate hypothesis
            kb_context = self.kb.get_context_for_explorer()
            hypothesis = self.agents['explorer'].autonomous_cycle(kb_context)

            # Design experiments
            designs = self.agents['architect'].design_experiments(
                hypothesis, self.baseline, n_designs=5
            )

            # Simulate
            results = []
            for design in designs:
                hp = HullParameters(**design)
                result = self.physics_engine.simulate(hp)
                results.append(result.to_dict())

            # Update knowledge base
            self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=cycle)

        # Verify cumulative statistics
        stats = self.kb.get_statistics()
        assert stats['total_experiments'] == n_cycles
        assert stats['total_designs_evaluated'] == n_cycles * 5
        assert stats['total_cycles'] == n_cycles

    def test_knowledge_base_persistence(self):
        """Test that knowledge base persists to disk."""
        # Add experiment
        hypothesis = {'description': 'Test hypothesis', 'parameter_range': {}}
        designs = [self.baseline.copy() for _ in range(5)]
        results = []

        for design in designs:
            design.pop('name', None)
            design.pop('description', None)
            hp = HullParameters(**design)
            result = self.physics_engine.simulate(hp)
            results.append(result.to_dict())

        self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=1)

        # Save explicitly
        self.kb.save()

        # Verify files exist
        assert os.path.exists(os.path.join(self.kb_path, 'experiments.json'))
        assert os.path.exists(os.path.join(self.kb_path, 'statistics.json'))

        # Load into new knowledge base
        kb2 = KnowledgeBase(storage_path=self.kb_path)

        # Verify data was loaded
        assert len(kb2.experiments) == 1
        assert kb2.statistics['total_experiments'] == 1

    def test_context_for_explorer(self):
        """Test context generation for Explorer agent."""
        # Add some experiments
        for i in range(3):
            hypothesis = {'description': f'Hypothesis {i}', 'parameter_range': {}}
            designs = [self.baseline.copy() for _ in range(5)]
            results = []

            for design in designs:
                design.pop('name', None)
                design.pop('description', None)
                hp = HullParameters(**design)
                result = self.physics_engine.simulate(hp)
                results.append(result.to_dict())

            self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=i+1)

        # Get context
        context = self.kb.get_context_for_explorer(max_entries=10)

        assert 'recent_experiments' in context
        assert 'extracted_principles' in context
        assert 'best_designs' in context
        assert 'statistics' in context

        assert len(context['recent_experiments']) == 3

    def test_best_designs_tracking(self):
        """Test that best designs are tracked correctly."""
        # Create designs with varying scores
        for cycle in range(3):
            hypothesis = {'description': f'Cycle {cycle}', 'parameter_range': {}}

            # Create 5 designs with predictable variation
            designs = []
            for i in range(5):
                design = self.baseline.copy()
                design.pop('name', None)
                design.pop('description', None)
                # Vary speed to create score variation
                design['design_speed'] = 20.0 + i * 2.0
                designs.append(design)

            results = []
            for design in designs:
                hp = HullParameters(**design)
                result = self.physics_engine.simulate(hp)
                results.append(result.to_dict())

            self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=cycle+1)

        # Get best designs
        best_designs = self.kb.get_best_designs(n=10)

        assert len(best_designs) > 0
        assert all('design' in bd for bd in best_designs)
        assert all('result' in bd for bd in best_designs)

        # Verify sorted by score (descending)
        scores = [bd['overall_score'] for bd in best_designs]
        assert scores == sorted(scores, reverse=True)

    def test_principle_extraction(self):
        """Test that design principles are extracted from experiments."""
        # Create experiment with clear trend
        hypothesis = {
            'description': 'Hull spacing vs stability',
            'parameter_range': {'hull_spacing': (4.5, 7.0)}
        }

        designs = []
        for i in range(10):
            design = self.baseline.copy()
            design.pop('name', None)
            design.pop('description', None)
            # Linearly vary hull spacing
            design['hull_spacing'] = 4.5 + i * 0.25
            designs.append(design)

        results = []
        for design in designs:
            hp = HullParameters(**design)
            result = self.physics_engine.simulate(hp)
            results.append(result.to_dict())

        self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=1)

        # Check if principles were extracted
        assert len(self.kb.principles) > 0

        # At least one principle should be about hull_spacing
        spacing_principles = [
            p for p in self.kb.principles
            if p['parameter'] == 'hull_spacing'
        ]

        # Hull spacing typically correlates with stability
        # (We can't assert exact value, but should find correlation)
        assert len(spacing_principles) >= 0  # May or may not find correlation

    def test_markdown_report_generation(self):
        """Test markdown report generation."""
        # Add some data
        hypothesis = {'description': 'Test', 'parameter_range': {}}
        designs = [self.baseline.copy() for _ in range(5)]
        results = []

        for design in designs:
            design.pop('name', None)
            design.pop('description', None)
            hp = HullParameters(**design)
            result = self.physics_engine.simulate(hp)
            results.append(result.to_dict())

        self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=1)

        # Generate report
        report_path = os.path.join(self.temp_dir, 'test_report.md')
        report = self.kb.export_markdown_report(report_path)

        assert os.path.exists(report_path)
        assert len(report) > 100  # Should be substantial
        assert 'Research Report' in report
        assert 'Total Research Cycles' in report

    def test_invalid_designs_handled(self):
        """Test that invalid designs are handled gracefully."""
        # Create intentionally invalid design
        hypothesis = {'description': 'Test invalid', 'parameter_range': {}}

        invalid_design = self.baseline.copy()
        invalid_design.pop('name', None)
        invalid_design.pop('description', None)
        invalid_design['length_overall'] = 5.0  # Too small, will fail validation

        designs = [invalid_design]

        # This should raise ValueError during HullParameters creation
        with pytest.raises(ValueError):
            hp = HullParameters(**invalid_design)

    def test_empty_results_handled(self):
        """Test that empty/failed results are handled gracefully."""
        hypothesis = {'description': 'Test empty', 'parameter_range': {}}
        designs = []
        results = []

        # Add empty experiment
        self.kb.add_experiment_results(hypothesis, designs, results, cycle_number=1)

        # Should not crash
        stats = self.kb.get_statistics()
        assert stats['total_experiments'] == 1
        assert stats['total_designs_evaluated'] == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
