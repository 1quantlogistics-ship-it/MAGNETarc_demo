"""
ExplorerAgent: Parameter space exploration specialist
======================================================

The Explorer systematically explores the parameter space to discover
high-potential regions and identify boundaries.

Phase E Enhancements:
- Augmentation policy exploration (Task 1.6)
- Policy mutation and crossover for evolutionary search
- Integration with AugmentationPolicy schema
"""

from typing import Dict, Any, List, Optional
from agents.base import BaseAgent, AgentCapability
from llm.router import LLMRouter

# Phase E: Augmentation policy support
try:
    from schemas.augmentation_policy import (
        AugmentationPolicy, mutate_policy, crossover_policies,
        AugmentationOpType, PolicyEvolutionStrategy
    )
    AUGMENTATION_POLICY_AVAILABLE = True
except ImportError:
    AUGMENTATION_POLICY_AVAILABLE = False


class ExplorerAgent(BaseAgent):
    """
    Parameter space exploration agent.

    Responsibilities:
    - Systematic parameter space exploration
    - Identify high-potential regions
    - Map parameter boundaries
    - Generate exploratory proposals
    - Use exploration strategies (grid, random, bayesian)
    """

    def __init__(
        self,
        agent_id: str = "explorer_001",
        model: str = "qwen2.5-32b",
        llm_router: LLMRouter = None,
        voting_weight: float = 1.2,
        memory_path: str = "/workspace/arc/memory"
    ):
        """Initialize Explorer agent."""
        super().__init__(
            agent_id=agent_id,
            role="explorer",
            model=model,
            capabilities=[AgentCapability.EXPLORATION, AgentCapability.PROPOSAL_GENERATION],
            voting_weight=voting_weight,
            priority="medium",
            offline=False,
            memory_path=memory_path
        )
        self.llm_router = llm_router or LLMRouter(offline_mode=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate exploratory experiment proposals.

        Args:
            input_data: Contains exploration strategy and bounds

        Returns:
            List of exploratory proposals
        """
        import time
        start_time = time.time()

        try:
            # Read memory
            directive = self.read_memory("directive.json")
            history = self.read_memory("history_summary.json")
            constraints = self.read_memory("constraints.json")

            # Build prompt
            prompt = self._build_exploration_prompt(directive, history, constraints)

            # Get LLM client
            client = self.llm_router.get_client_for_role(self.role)

            # Generate exploration proposals
            response = client.generate_json(prompt, max_tokens=2500, temperature=0.9)

            # Track success
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("generate_exploration", success=True, duration_ms=duration_ms)

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_task("generate_exploration", success=False, duration_ms=duration_ms)
            raise e

    def vote_on_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explorers favor novel parameter combinations.

        Args:
            proposal: Experiment proposal

        Returns:
            Vote decision favoring novelty
        """
        novelty_category = proposal.get("novelty_category", "exploit")

        # Explorers prefer explore/wildcat over exploit
        if novelty_category in ["explore", "wildcat"]:
            return {
                "decision": "approve",
                "confidence": 0.9,
                "reasoning": "Proposal supports parameter space exploration"
            }
        else:
            return {
                "decision": "approve",
                "confidence": 0.6,
                "reasoning": "Exploit experiments have limited exploration value"
            }

    def propose_augmentation_policies(
        self,
        num_policies: int = 3,
        strategy: str = "evolutionary",
        parent_policies: Optional[List[AugmentationPolicy]] = None
    ) -> List[Dict[str, Any]]:
        """
        Propose augmentation policies for exploration.

        Phase E: Task 1.6 - Enable Explorer to generate augmentation policy proposals
        using evolutionary strategies (mutation, crossover).

        Args:
            num_policies: Number of policies to propose
            strategy: Evolution strategy ("random", "mutate", "crossover")
            parent_policies: Parent policies for mutation/crossover

        Returns:
            List of augmentation policy proposals
        """
        if not AUGMENTATION_POLICY_AVAILABLE:
            raise RuntimeError("Augmentation policy schema not available")

        import random

        proposals = []

        if strategy == "random":
            # Generate random policies from safe operation space
            for i in range(num_policies):
                policy = self._generate_random_policy(f"explore_random_{i}")
                proposals.append({
                    "augmentation_policy": policy.to_dict(),
                    "strategy": "random",
                    "novelty_category": "explore"
                })

        elif strategy == "mutate" and parent_policies:
            # Mutate existing policies
            for i in range(num_policies):
                parent = random.choice(parent_policies)
                mutated = mutate_policy(parent, mutation_rate=0.3)
                proposals.append({
                    "augmentation_policy": mutated.to_dict(),
                    "strategy": "mutate",
                    "parent": parent.name,
                    "novelty_category": "explore"
                })

        elif strategy == "crossover" and parent_policies and len(parent_policies) >= 2:
            # Crossover pairs of policies
            for i in range(num_policies):
                parent1, parent2 = random.sample(parent_policies, 2)
                offspring = crossover_policies(parent1, parent2)
                proposals.append({
                    "augmentation_policy": offspring.to_dict(),
                    "strategy": "crossover",
                    "parents": [parent1.name, parent2.name],
                    "novelty_category": "explore"
                })

        else:
            # Fallback: baseline policy
            baseline = AugmentationPolicy.baseline_policy()
            proposals.append({
                "augmentation_policy": baseline.to_dict(),
                "strategy": "baseline",
                "novelty_category": "exploit"
            })

        return proposals

    def _generate_random_policy(self, name: str) -> AugmentationPolicy:
        """
        Generate random augmentation policy from safe operation space.

        Args:
            name: Policy name

        Returns:
            Random policy with 2-5 operations
        """
        import random

        # Available safe operations
        available_ops = list(AugmentationOpType)

        # Random number of operations (2-5)
        num_ops = random.randint(2, 5)

        # Sample operations
        selected_op_types = random.sample(available_ops, num_ops)

        operations = []
        for op_type in selected_op_types:
            # Generate safe magnitude
            if op_type == AugmentationOpType.ROTATE:
                magnitude = random.uniform(-12.0, 12.0)
            elif op_type in [AugmentationOpType.BRIGHTNESS, AugmentationOpType.CONTRAST]:
                magnitude = random.uniform(-0.08, 0.08)
            elif op_type == AugmentationOpType.GAMMA:
                magnitude = random.uniform(0.92, 1.08)
            elif op_type == AugmentationOpType.SCALE:
                magnitude = random.uniform(0.92, 1.08)
            elif op_type == AugmentationOpType.GAUSSIAN_NOISE:
                magnitude = random.uniform(0.001, 0.008)
            elif op_type == AugmentationOpType.GAUSSIAN_BLUR:
                magnitude = random.uniform(0.5, 2.5)
            else:
                magnitude = random.uniform(0.0, 1.0)

            # Random probability (0.2-0.7)
            probability = random.uniform(0.2, 0.7)

            from schemas.augmentation_policy import AugmentationOp
            operations.append(AugmentationOp(
                op_type=op_type,
                magnitude=magnitude,
                probability=probability
            ))

        policy = AugmentationPolicy(
            name=name,
            operations=operations,
            evolution_strategy=PolicyEvolutionStrategy.RANDOM
        )

        return policy

    def _build_exploration_prompt(
        self,
        directive: Dict[str, Any],
        history: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Build prompt for exploration."""
        # Phase E: Add augmentation policy guidance
        augmentation_guidance = ""
        if AUGMENTATION_POLICY_AVAILABLE:
            augmentation_guidance = """

# Augmentation Policy Exploration (Phase E)

You can now propose augmentation policies using the following safe operations:

**Geometric Transforms** (Safe):
- "rotate": Rotation (±15° max)
- "horizontal_flip": Mirror flip
- "scale": Zoom (0.9-1.1 range)
- "translate": Shift (±10% max)

**Intensity Transforms** (Careful):
- "brightness": Brightness adjust (±10% max)
- "contrast": Contrast adjust (±10% max)
- "gamma": Gamma correction (0.9-1.1 range)

**Noise/Blur** (Careful):
- "gaussian_noise": Add noise (σ ≤ 0.01)
- "gaussian_blur": Blur (kernel ≤ 3)

**FORBIDDEN** (DO NOT USE):
- Color jitter (hue/saturation)
- Cutout / random erasing
- Strong elastic deformation

**Example Augmentation Policy Proposal**:
```json
{
  "augmentation_policy": {
    "name": "explore_intensity_v1",
    "operations": [
      {"op_type": "brightness", "magnitude": 0.08, "probability": 0.4},
      {"op_type": "contrast", "magnitude": 0.06, "probability": 0.3},
      {"op_type": "rotate", "magnitude": 10.0, "probability": 0.5}
    ],
    "evolution_strategy": "random",
    "dri_constraint": 0.6
  }
}
```

**Critical**: All policies must maintain DRI ≥ 0.6 (Disc Relevance Index).
"""

        return f"""You are the Explorer agent in ARC (Autonomous Research Collective).
Your role is to systematically explore the parameter space to discover optimal regions.

# Strategic Directive
{directive}

# Research History
{history}

# Safety Constraints
{constraints}
{augmentation_guidance}

# Your Task
Generate exploratory experiment proposals that:
1. Systematically sample unexplored parameter regions
2. Test parameter boundaries (within safety constraints)
3. Use diverse exploration strategies (grid, random, bayesian)
4. Maximize information gain about parameter space
5. (Phase E) Propose novel augmentation policies within safe operation space

Focus on NOVELTY and COVERAGE, not incremental improvements.

Return ONLY a valid JSON object:
{{
  "proposals": [
    {{
      "experiment_id": "exp_explore_XXX",
      "name": "descriptive_name",
      "hypothesis": "exploration hypothesis",
      "novelty_category": "explore" | "wildcat",
      "exploration_strategy": "grid" | "random" | "bayesian" | "boundary",
      "predicted_metrics": {{"auc": 0.XX, "sensitivity": 0.XX, "specificity": 0.XX}},
      "config_changes": {{"param1": value1, "param2": value2}},
      "augmentation_policy": {{...}} (optional, Phase E),
      "justification": "exploration rationale",
      "information_gain_estimate": 0.XX
    }}
  ]
}}"""
