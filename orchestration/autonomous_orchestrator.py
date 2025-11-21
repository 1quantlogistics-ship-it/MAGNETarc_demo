#!/usr/bin/env python3
"""
Autonomous Orchestrator for MAGNET
===================================

The AutonomousOrchestrator coordinates all naval agents in a continuous
research loop. It implements the 6-step autonomous research cycle.

Author: Agent 2
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from agents.base_naval_agent import NavalAgentResponse
from agents.explorer_agent import ExplorerAgent
from agents.experimental_architect_agent import ExperimentalArchitectAgent
from agents.critic_naval_agent import CriticNavalAgent
from agents.historian_naval_agent import HistorianNavalAgent
from agents.supervisor_naval_agent import SupervisorNavalAgent
from memory.knowledge_base import KnowledgeBase


@dataclass
class OrchestrationState:
    """State of the autonomous research loop"""
    cycle_number: int = 0
    total_experiments: int = 0
    total_valid_designs: int = 0
    best_overall_score: float = 0.0
    current_hypothesis: Optional[Dict[str, Any]] = None
    current_designs: List[Dict[str, Any]] = field(default_factory=list)
    current_results: List[Dict[str, Any]] = field(default_factory=list)
    exploration_strategy: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class AutonomousOrchestrator:
    """Orchestrates autonomous research cycles across all agents."""

    def __init__(
        self,
        agents: Dict[str, Any],
        knowledge_base: KnowledgeBase,
        physics_simulator: Callable,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.agents = agents
        self.knowledge_base = knowledge_base
        self.physics_simulator = physics_simulator
        self.config = config or {}
        self.logger = logger or self._create_logger()

        self.state = OrchestrationState()
        self.running = False

        self.state_file = Path(self.config.get("state_file", "memory/orchestrator_state.json"))
        self._load_state()

    def _create_logger(self) -> logging.Logger:
        """Create default logger"""
        logger = logging.getLogger("AutonomousOrchestrator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(logger)

        return logger

    def _load_state(self):
        """Load orchestrator state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self.state, key):
                            setattr(self.state, key, value)
                self.logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Save orchestrator state to disk"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state.last_updated = datetime.now().isoformat()

            state_dict = {
                "cycle_number": self.state.cycle_number,
                "total_experiments": self.state.total_experiments,
                "total_valid_designs": self.state.total_valid_designs,
                "best_overall_score": self.state.best_overall_score,
                "exploration_strategy": self.state.exploration_strategy,
                "error_count": self.state.error_count,
                "last_error": self.state.last_error,
                "started_at": self.state.started_at,
                "last_updated": self.state.last_updated
            }

            with open(self.state_file, 'w') as f:
                json.dump(state_dict, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    async def run_cycle(self) -> Dict[str, Any]:
        """Run a single autonomous research cycle."""
        self.state.cycle_number += 1
        cycle_num = self.state.cycle_number

        self.logger.info("="*70)
        self.logger.info(f"AUTONOMOUS CYCLE {cycle_num}")
        self.logger.info("="*70)

        cycle_results = {
            "cycle": cycle_num,
            "timestamp": datetime.now().isoformat(),
            "steps": {}
        }

        try:
            # STEP 1: Explorer
            self.logger.info(f"\n[Step 1/7] Explorer: Generating hypothesis...")
            explorer_response = await self._run_explorer()
            cycle_results["steps"]["explorer"] = {"success": True, "hypothesis": explorer_response.data.get("hypothesis")}
            self.state.current_hypothesis = explorer_response.data.get("hypothesis")

            # STEP 2: Architect
            self.logger.info(f"\n[Step 2/7] Architect: Designing experiments...")
            architect_response = await self._run_architect()
            cycle_results["steps"]["architect"] = {"success": True, "num_designs": len(architect_response.data.get("designs", []))}
            self.state.current_designs = architect_response.data.get("designs", [])

            # STEP 3: Critic pre-review
            self.logger.info(f"\n[Step 3/7] Critic: Pre-simulation review...")
            critic_pre_response = await self._run_critic_pre()
            cycle_results["steps"]["critic_pre"] = {"success": True, "verdict": critic_pre_response.data.get("verdict")}

            if critic_pre_response.data.get("verdict") == "reject":
                self.logger.warning("Critic rejected designs. Skipping simulation.")
                cycle_results["skipped"] = True
                return cycle_results

            # STEP 4: Physics
            self.logger.info(f"\n[Step 4/7] Physics: Running simulations...")
            simulation_results = await self._run_physics()
            cycle_results["steps"]["physics"] = {"success": True, "num_results": len(simulation_results)}
            self.state.current_results = simulation_results

            valid_results = [r for r in simulation_results if r.get("results", {}).get("is_valid", False)]
            self.state.total_experiments += len(simulation_results)
            self.state.total_valid_designs += len(valid_results)

            if valid_results:
                best_this_cycle = max(r["results"]["overall_score"] for r in valid_results)
                if best_this_cycle > self.state.best_overall_score:
                    self.state.best_overall_score = best_this_cycle

            # STEP 5: Critic post-critique
            self.logger.info(f"\n[Step 5/7] Critic: Analyzing results...")
            critic_post_response = await self._run_critic_post()
            cycle_results["steps"]["critic_post"] = {"success": True, "insights": critic_post_response.data.get("insights", [])}

            # STEP 6: Historian
            self.logger.info(f"\n[Step 6/7] Historian: Updating knowledge base...")
            historian_response = await self._run_historian()
            cycle_results["steps"]["historian"] = {"success": True, "patterns": len(historian_response.data.get("new_patterns", []))}

            # STEP 7: Supervisor
            self.logger.info(f"\n[Step 7/7] Supervisor: Adjusting strategy...")
            supervisor_response = await self._run_supervisor()
            cycle_results["steps"]["supervisor"] = {"success": True, "strategy": supervisor_response.data.get("strategy_adjustment", {})}
            self.state.exploration_strategy = supervisor_response.data.get("strategy_adjustment", {})

            # Persist
            self.knowledge_base.add_experiment_results(
                hypothesis=self.state.current_hypothesis,
                designs=self.state.current_designs,
                results=self.state.current_results,
                cycle_number=cycle_num
            )

            self._save_state()
            self.logger.info(f"\n✅ Cycle {cycle_num} complete")
            cycle_results["success"] = True

        except Exception as e:
            self.logger.error(f"❌ Cycle {cycle_num} failed: {e}")
            self.state.error_count += 1
            self.state.last_error = str(e)
            cycle_results["success"] = False
            cycle_results["error"] = str(e)

            import traceback
            self.logger.error(traceback.format_exc())

        return cycle_results

    async def _run_explorer(self) -> NavalAgentResponse:
        explorer = self.agents["explorer"]
        kb_context = self.knowledge_base.get_context_for_explorer()

        context = {
            "knowledge_base": kb_context,
            "experiment_history": self.knowledge_base.experiments,
            "current_best": self.knowledge_base.get_best_designs(n=1)[0] if self.knowledge_base.get_best_designs(n=1) else {},
            "cycle_number": self.state.cycle_number,
            "exploration_strategy": self.state.exploration_strategy
        }

        response = explorer.autonomous_cycle(context)
        self.logger.info(f"  ✓ Hypothesis: {response.data['hypothesis']['statement'][:60]}...")
        return response

    async def _run_architect(self) -> NavalAgentResponse:
        architect = self.agents["architect"]
        best_designs = self.knowledge_base.get_best_designs(n=1)
        current_best = best_designs[0] if best_designs else None

        context = {
            "hypothesis": self.state.current_hypothesis,
            "current_best_design": current_best,
            "exploration_strategy": self.state.exploration_strategy
        }

        response = architect.autonomous_cycle(context)
        self.logger.info(f"  ✓ Designed {len(response.data['designs'])} experiments")
        return response

    async def _run_critic_pre(self) -> NavalAgentResponse:
        critic = self.agents["critic"]

        context = {
            "designs": self.state.current_designs,
            "hypothesis": self.state.current_hypothesis,
            "experiment_history": self.knowledge_base.experiments
        }

        response = critic.autonomous_cycle(context)
        verdict = response.data["verdict"]
        self.logger.info(f"  ✓ Verdict: {verdict}")
        return response

    async def _run_physics(self) -> List[Dict[str, Any]]:
        results = self.physics_simulator(self.state.current_designs)
        valid = sum(1 for r in results if r.get("results", {}).get("is_valid", False))
        self.logger.info(f"  ✓ Simulated {len(results)} designs ({valid} valid)")
        return results

    async def _run_critic_post(self) -> NavalAgentResponse:
        critic = self.agents["critic"]

        context = {
            "experiment_results": self.state.current_results,
            "hypothesis": self.state.current_hypothesis,
            "experiment_history": self.knowledge_base.experiments
        }

        response = critic.autonomous_cycle(context)
        insights = len(response.data.get("insights", []))
        self.logger.info(f"  ✓ Insights: {insights}")
        return response

    async def _run_historian(self) -> NavalAgentResponse:
        historian = self.agents["historian"]

        context = {
            "new_results": self.state.current_results,
            "current_history": {"experiments": self.knowledge_base.experiments},
            "knowledge_base": {},
            "cycle_number": self.state.cycle_number
        }

        response = historian.autonomous_cycle(context)
        patterns = len(response.data.get("new_patterns", []))
        self.logger.info(f"  ✓ Patterns: {patterns}")
        return response

    async def _run_supervisor(self) -> NavalAgentResponse:
        supervisor = self.agents["supervisor"]

        context = {
            "experiment_history": self.knowledge_base.experiments,
            "knowledge_base": self.knowledge_base.get_statistics(),
            "cycle_number": self.state.cycle_number,
            "current_best_score": self.state.best_overall_score
        }

        response = supervisor.autonomous_cycle(context)
        strategy = response.data["strategy_adjustment"]
        self.logger.info(f"  ✓ Strategy: {strategy['mode']}")
        return response

    async def run(self, max_cycles: Optional[int] = None):
        """Run autonomous research loop."""
        self.running = True
        self.logger.info("\n" + "="*70)
        self.logger.info("🚀 MAGNET AUTONOMOUS RESEARCH SYSTEM")
        self.logger.info("="*70)

        try:
            while self.running:
                if max_cycles and self.state.cycle_number >= max_cycles:
                    break

                await self.run_cycle()

                delay = self.config.get("cycle_delay", 0)
                if delay > 0 and self.running:
                    await asyncio.sleep(delay)

        except KeyboardInterrupt:
            self.logger.info("\n⏸ Interrupted by user")
        except Exception as e:
            self.logger.error(f"\n❌ Fatal error: {e}")
        finally:
            self.running = False
            self._save_state()
            self._print_summary()

    def stop(self):
        self.running = False

    def _print_summary(self):
        self.logger.info("\n" + "="*70)
        self.logger.info("RESEARCH SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"Total cycles: {self.state.cycle_number}")
        self.logger.info(f"Total experiments: {self.state.total_experiments}")
        self.logger.info(f"Valid designs: {self.state.total_valid_designs}")
        self.logger.info(f"Best score: {self.state.best_overall_score:.1f}")
        self.logger.info("="*70)


def create_orchestrator(
    llm_client,
    physics_simulator,
    memory_path: str = "memory/knowledge",
    config: Optional[Dict[str, Any]] = None
) -> AutonomousOrchestrator:
    """Create autonomous orchestrator with all agents."""
    from agents.base_naval_agent import NavalAgentConfig

    kb = KnowledgeBase(storage_path=memory_path)

    agents = {
        "explorer": ExplorerAgent(
            NavalAgentConfig("explorer_001", "explorer", "mock", memory_path=memory_path),
            llm_client
        ),
        "architect": ExperimentalArchitectAgent(
            NavalAgentConfig("architect_001", "architect", "mock", memory_path=memory_path),
            llm_client
        ),
        "critic": CriticNavalAgent(
            NavalAgentConfig("critic_001", "critic", "mock", memory_path=memory_path),
            llm_client
        ),
        "historian": HistorianNavalAgent(
            NavalAgentConfig("historian_001", "historian", "mock", memory_path=memory_path),
            llm_client
        ),
        "supervisor": SupervisorNavalAgent(
            NavalAgentConfig("supervisor_001", "supervisor", "mock", memory_path=memory_path),
            llm_client
        )
    }

    return AutonomousOrchestrator(
        agents=agents,
        knowledge_base=kb,
        physics_simulator=physics_simulator,
        config=config or {}
    )
