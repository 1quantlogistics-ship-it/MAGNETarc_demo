"""
Integration Tests for MAGNET Agent 2 (Architect Agent)
========================================================

Tests the complete flow:
1. Explorer generates hypothesis
2. Architect designs experiments
3. Validates output format for Physics Engine
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.base_naval_agent import NavalAgentConfig
from agents.explorer_agent import ExplorerAgent
from agents.experimental_architect_agent import ExperimentalArchitectAgent
from llm.local_client import MockLLMClient
from config.magnet_config import CONFIG_MOCK


class TestArchitectCycle:
    """Integration tests for Architect agent"""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client for testing"""
        return MockLLMClient()

    @pytest.fixture
    def explorer_agent(self, mock_llm_client):
        """Create Explorer agent with mock LLM"""
        config = NavalAgentConfig(
            agent_id="explorer_001",
            role="explorer",
            model="mock-llm",
            memory_path="/tmp/magnet_test/memory"
        )
        return ExplorerAgent(config, mock_llm_client)

    @pytest.fixture
    def architect_agent(self, mock_llm_client):
        """Create Architect agent with mock LLM"""
        config = NavalAgentConfig(
            agent_id="architect_001",
            role="architect",
            model="mock-llm",
            memory_path="/tmp/magnet_test/memory"
        )
        return ExperimentalArchitectAgent(config, mock_llm_client)

    def test_explorer_generates_hypothesis(self, explorer_agent):
        """Test that Explorer agent generates valid hypothesis"""
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
        response = explorer_agent.autonomous_cycle(context)

        # Validate response
        assert response.agent_id == "explorer_001"
        assert response.action == "submit_hypothesis"
        assert response.confidence > 0.0
        assert "hypothesis" in response.data

        hypothesis = response.data["hypothesis"]

        # Validate hypothesis structure
        assert "id" in hypothesis
        assert "statement" in hypothesis
        assert "type" in hypothesis
        assert "test_protocol" in hypothesis
        assert "expected_outcome" in hypothesis
        assert "success_criteria" in hypothesis

        # Validate test protocol
        protocol = hypothesis["test_protocol"]
        assert "parameters_to_vary" in protocol
        assert "ranges" in protocol
        assert "num_samples" in protocol
        assert len(protocol["parameters_to_vary"]) > 0
        assert len(protocol["ranges"]) == len(protocol["parameters_to_vary"])

        print(f"✓ Explorer generated hypothesis: {hypothesis['statement']}")

    def test_architect_designs_experiments(self, architect_agent):
        """Test that Architect agent designs valid experiments"""
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
        response = architect_agent.autonomous_cycle(context)

        # Validate response
        assert response.agent_id == "architect_001"
        assert response.action == "submit_experiments"
        assert response.confidence > 0.0
        assert "designs" in response.data
        assert "hypothesis" in response.data

        designs = response.data["designs"]

        # Validate number of designs
        assert len(designs) == 8  # Should match num_samples

        # Validate each design
        for design in designs:
            assert "design_id" in design
            assert "parameters" in design
            assert "hypothesis_id" in design
            assert design["hypothesis_id"] == "hyp_001_01"

            params = design["parameters"]

            # Check that varied parameter is present and in range
            assert "hull_spacing" in params
            assert 4.0 <= params["hull_spacing"] <= 6.0

            # Check that all required parameters are present
            required_params = [
                "length_overall", "beam", "hull_spacing", "hull_depth",
                "deadrise_angle", "freeboard", "lcb_position", "prismatic_coefficient"
            ]
            for param in required_params:
                assert param in params, f"Missing parameter: {param}"

        print(f"✓ Architect designed {len(designs)} experiments")

    def test_constraint_enforcement(self, architect_agent):
        """Test that physical constraints are enforced"""
        # Create hypothesis that might violate constraints
        hypothesis = {
            "id": "hyp_002_01",
            "statement": "Test extreme hull spacing",
            "type": "counter-intuitive",
            "test_protocol": {
                "parameters_to_vary": ["hull_spacing", "beam"],
                "ranges": [[5.0, 7.0], [5.0, 7.0]],  # Might violate spacing < beam
                "num_samples": 10,
                "fixed_parameters": {}
            },
            "expected_outcome": "Identify constraint boundaries",
            "success_criteria": "designs_valid == true"
        }

        context = {
            "hypothesis": hypothesis,
            "current_best_design": ExperimentalArchitectAgent.PARAMETER_DEFAULTS.copy()
        }

        # Design experiments
        response = architect_agent.autonomous_cycle(context)
        designs = response.data["designs"]

        # Validate constraints
        for design in designs:
            params = design["parameters"]

            # Constraint 1: hull_spacing < beam
            assert params["hull_spacing"] < params["beam"], \
                f"Constraint violated: spacing={params['hull_spacing']:.2f} >= beam={params['beam']:.2f}"

            # Constraint 2: freeboard < hull_depth
            assert params["freeboard"] < params["hull_depth"], \
                f"Constraint violated: freeboard={params['freeboard']:.2f} >= depth={params['hull_depth']:.2f}"

            # Constraint 3: All parameters in valid ranges
            for param, (min_val, max_val) in ExperimentalArchitectAgent.PARAMETER_RANGES.items():
                if param in params:
                    assert min_val <= params[param] <= max_val, \
                        f"Parameter {param}={params[param]:.2f} out of range [{min_val}, {max_val}]"

        print(f"✓ All {len(designs)} designs satisfy physical constraints")

    def test_sampling_strategies(self, architect_agent):
        """Test different sampling strategies"""
        strategies = [
            ("exploration", "Latin Hypercube"),
            ("exploitation", "Gaussian"),
            ("counter-intuitive", "Edge/corner")
        ]

        for hypothesis_type, expected_strategy in strategies:
            hypothesis = {
                "id": f"hyp_test_{hypothesis_type}",
                "statement": f"Test {hypothesis_type} sampling",
                "type": hypothesis_type,
                "test_protocol": {
                    "parameters_to_vary": ["hull_spacing"],
                    "ranges": [[4.0, 6.0]],
                    "num_samples": 6,
                    "fixed_parameters": {}
                },
                "expected_outcome": "Test sampling",
                "success_criteria": "valid"
            }

            context = {
                "hypothesis": hypothesis,
                "current_best_design": ExperimentalArchitectAgent.PARAMETER_DEFAULTS.copy()
            }

            response = architect_agent.autonomous_cycle(context)
            designs = response.data["designs"]

            assert len(designs) == 6
            print(f"✓ {hypothesis_type} sampling: {len(designs)} designs")

    def test_full_cycle_explorer_to_architect(self, explorer_agent, architect_agent):
        """Test full cycle: Explorer → Architect"""
        # Step 1: Explorer generates hypothesis
        explorer_context = {
            "knowledge_base": {},
            "experiment_history": [
                {
                    "parameters": {"length_overall": 18.0, "beam": 6.0, "hull_spacing": 4.5},
                    "results": {"stability_score": 75.0, "speed_score": 70.0}
                },
                {
                    "parameters": {"length_overall": 19.0, "beam": 6.5, "hull_spacing": 5.0},
                    "results": {"stability_score": 78.0, "speed_score": 68.0}
                }
            ],
            "current_best": {
                "length_overall": 19.0,
                "beam": 6.5,
                "hull_spacing": 5.0,
                "stability_score": 78.0,
                "speed_score": 68.0
            },
            "cycle_number": 2
        }

        explorer_response = explorer_agent.autonomous_cycle(explorer_context)
        hypothesis = explorer_response.data["hypothesis"]

        print(f"✓ Explorer hypothesis: {hypothesis['statement']}")

        # Step 2: Architect designs experiments for hypothesis
        architect_context = {
            "hypothesis": hypothesis,
            "current_best_design": {
                "length_overall": 19.0,
                "beam": 6.5,
                "hull_spacing": 5.0,
                "hull_depth": 2.5,
                "deadrise_angle": 12.0,
                "freeboard": 1.5,
                "lcb_position": 0.50,
                "prismatic_coefficient": 0.62
            },
            "knowledge_base": explorer_context["knowledge_base"]
        }

        architect_response = architect_agent.autonomous_cycle(architect_context)
        designs = architect_response.data["designs"]

        print(f"✓ Architect designed {len(designs)} experiments")

        # Step 3: Validate designs are ready for Physics Engine
        for design in designs:
            # Check design format matches what Physics Engine expects
            assert "design_id" in design
            assert "parameters" in design

            params = design["parameters"]

            # All 12 parameters should be present
            expected_params = [
                "length_overall", "beam", "hull_spacing", "hull_depth",
                "deadrise_angle", "freeboard", "lcb_position",
                "prismatic_coefficient", "waterline_beam", "block_coefficient",
                "wetted_surface_area", "displacement"
            ]

            for param in expected_params:
                assert param in params, f"Missing parameter for Physics Engine: {param}"
                assert isinstance(params[param], (int, float)), \
                    f"Parameter {param} should be numeric, got {type(params[param])}"

        print("✓ All designs ready for Physics Engine simulation")
        print(f"✓ Full cycle complete: {len(designs)} designs ready for testing")


class TestLLMClient:
    """Tests for LLM client"""

    def test_mock_client_generation(self):
        """Test mock LLM client"""
        client = MockLLMClient()

        # Test text generation
        response = client.generate("Test prompt", max_tokens=100)
        assert isinstance(response, str)
        assert len(response) > 0

        # Test JSON generation
        json_response = client.generate_json("Generate hypothesis", max_tokens=100)
        assert isinstance(json_response, dict)

        # Test health check
        assert client.health_check() is True

        # Test stats
        stats = client.get_stats()
        assert stats["total_calls"] > 0
        assert stats["success_rate"] == 1.0

        print("✓ Mock LLM client working correctly")

    def test_json_extraction(self):
        """Test JSON extraction from various formats"""
        from llm.local_client import LocalLLMClient

        # Test pure JSON
        text1 = '{"key": "value"}'
        assert LocalLLMClient.extract_json(text1) == {"key": "value"}

        # Test JSON in code block
        text2 = '```json\n{"key": "value"}\n```'
        assert LocalLLMClient.extract_json(text2) == {"key": "value"}

        # Test JSON with <think> tags
        text3 = '<think>Thinking...</think>\n{"key": "value"}'
        assert LocalLLMClient.extract_json(text3) == {"key": "value"}

        # Test JSON in backticks
        text4 = '`{"key": "value"}`'
        assert LocalLLMClient.extract_json(text4) == {"key": "value"}

        print("✓ JSON extraction working for all formats")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
