"""
MAGNET Configuration
====================

Configuration for MAGNET autonomous naval design system.
Defines agent configurations, model parameters, and system settings.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseSettings
import os


# Agent Configurations
@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    agent_id: str
    role: str
    model: str
    voting_weight: float = 1.0
    priority: str = "medium"
    offline: bool = False
    gpu_id: Optional[int] = None
    memory_path: str = "/workspace/magnet/memory"


# System Configuration Classes
class MAGNETSettings(BaseSettings):
    """
    MAGNET system settings using Pydantic for validation.

    Loads from environment variables or .env file.
    """

    # Environment
    environment: str = "dev"  # dev, test, prod

    # Directories
    workspace_dir: str = "/workspace/magnet"
    memory_dir: str = "/workspace/magnet/memory"
    experiments_dir: str = "/workspace/magnet/experiments"
    logs_dir: str = "/workspace/magnet/logs"

    # LLM Configuration
    llm_endpoint: str = "http://localhost:8000/v1/completions"
    llm_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    llm_timeout: int = 120
    llm_max_retries: int = 3
    llm_quantization: str = "awq"  # 4-bit quantization

    # GPU Configuration
    gpu_devices: str = "0,1"  # Comma-separated GPU IDs for 2x A40
    tensor_parallel_size: int = 2  # Number of GPUs for model parallelism
    gpu_memory_utilization: float = 0.85

    # Agent Configuration
    cycle_delay: int = 5  # Seconds between autonomous cycles
    max_designs_per_cycle: int = 10  # Max designs to generate per hypothesis
    require_llm_validation: bool = False  # LLM validation of designs (slower)

    # Safety & Control
    mode: str = "SEMI"  # SEMI, AUTO, FULL, OFF
    require_approval_for_experiments: bool = False  # Human approval before simulation

    # WebSocket (for UI integration)
    websocket_port: int = 8765
    enable_websocket: bool = True

    class Config:
        env_file = ".env"
        env_prefix = "MAGNET_"


# Predefined Agent Configurations
def get_explorer_config(memory_path: str = "/workspace/magnet/memory") -> AgentConfig:
    """Get configuration for Explorer agent (Agent 1)"""
    return AgentConfig(
        agent_id="explorer_001",
        role="explorer",
        model="local-deepseek",
        voting_weight=1.0,
        priority="high",
        offline=False,
        gpu_id=0,  # GPU 0
        memory_path=memory_path
    )


def get_architect_config(memory_path: str = "/workspace/magnet/memory") -> AgentConfig:
    """Get configuration for Architect agent (Agent 2)"""
    return AgentConfig(
        agent_id="architect_001",
        role="architect",
        model="local-deepseek",
        voting_weight=1.5,  # Higher weight for experimental design
        priority="high",
        offline=False,
        gpu_id=1,  # GPU 1
        memory_path=memory_path
    )


def get_critic_config(memory_path: str = "/workspace/magnet/memory") -> AgentConfig:
    """Get configuration for Critic agent (Agent 3)"""
    return AgentConfig(
        agent_id="critic_001",
        role="critic",
        model="local-deepseek",
        voting_weight=1.2,
        priority="medium",
        offline=False,
        gpu_id=0,  # Share GPU 0 with Explorer
        memory_path=memory_path
    )


def get_supervisor_config(memory_path: str = "/workspace/magnet/memory") -> AgentConfig:
    """Get configuration for Supervisor agent (Agent 4)"""
    return AgentConfig(
        agent_id="supervisor_001",
        role="supervisor",
        model="local-deepseek",
        voting_weight=2.0,  # Highest weight for meta-level oversight
        priority="critical",
        offline=False,
        gpu_id=1,  # Share GPU 1 with Architect
        memory_path=memory_path
    )


def get_director_config(memory_path: str = "/workspace/magnet/memory") -> AgentConfig:
    """Get configuration for Director agent"""
    return AgentConfig(
        agent_id="director_001",
        role="director",
        model="local-deepseek",
        voting_weight=1.8,
        priority="high",
        offline=False,
        gpu_id=0,  # Share GPU 0
        memory_path=memory_path
    )


# System Profiles
@dataclass
class MAGNETProfile:
    """Complete system profile with all agent configurations"""
    name: str
    description: str
    settings: MAGNETSettings
    agents: Dict[str, AgentConfig]


def get_2xA40_profile() -> MAGNETProfile:
    """
    Configuration profile for 2x A40 GPUs (48GB each).

    GPU 0: Explorer + Critic + Director (shared, ~20GB total)
    GPU 1: Architect + Supervisor (shared, ~20GB total)
    """
    settings = MAGNETSettings(
        environment="dev",
        workspace_dir="/workspace/magnet",
        memory_dir="/workspace/magnet/memory",
        llm_endpoint="http://localhost:8000/v1/completions",
        llm_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        gpu_devices="0,1",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        cycle_delay=5,
        max_designs_per_cycle=10,
        require_llm_validation=False,
        mode="SEMI"
    )

    memory_path = settings.memory_dir

    agents = {
        "explorer": get_explorer_config(memory_path),
        "architect": get_architect_config(memory_path),
        "critic": get_critic_config(memory_path),
        "supervisor": get_supervisor_config(memory_path),
        "director": get_director_config(memory_path)
    }

    return MAGNETProfile(
        name="2xA40_profile",
        description="Configuration for 2x NVIDIA A40 GPUs (48GB each)",
        settings=settings,
        agents=agents
    )


def get_1xA40_profile() -> MAGNETProfile:
    """
    Configuration profile for 1x A40 GPU (48GB).

    GPU 0: All agents share single GPU
    """
    settings = MAGNETSettings(
        environment="dev",
        workspace_dir="/workspace/magnet",
        memory_dir="/workspace/magnet/memory",
        llm_endpoint="http://localhost:8000/v1/completions",
        llm_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        gpu_devices="0",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        cycle_delay=10,  # Slower cycles due to single GPU
        max_designs_per_cycle=8,
        require_llm_validation=False,
        mode="SEMI"
    )

    memory_path = settings.memory_dir

    # All agents use GPU 0
    agents = {
        "explorer": get_explorer_config(memory_path),
        "architect": get_architect_config(memory_path),
        "critic": get_critic_config(memory_path),
        "supervisor": get_supervisor_config(memory_path),
        "director": get_director_config(memory_path)
    }

    # Override GPU IDs to 0
    for agent_config in agents.values():
        agent_config.gpu_id = 0

    return MAGNETProfile(
        name="1xA40_profile",
        description="Configuration for 1x NVIDIA A40 GPU (48GB)",
        settings=settings,
        agents=agents
    )


def get_mock_profile() -> MAGNETProfile:
    """
    Configuration profile for testing without GPUs (mock LLM).

    Uses MockLLMClient instead of real vLLM.
    """
    settings = MAGNETSettings(
        environment="test",
        workspace_dir="/tmp/magnet_test",
        memory_dir="/tmp/magnet_test/memory",
        llm_endpoint="mock://localhost",
        llm_model="mock-llm",
        gpu_devices="",
        tensor_parallel_size=1,
        cycle_delay=1,
        max_designs_per_cycle=5,
        require_llm_validation=False,
        mode="SEMI"
    )

    memory_path = settings.memory_dir

    agents = {
        "explorer": get_explorer_config(memory_path),
        "architect": get_architect_config(memory_path),
        "critic": get_critic_config(memory_path)
    }

    # Set all agents to offline mode (no GPU)
    for agent_config in agents.values():
        agent_config.offline = True
        agent_config.gpu_id = None

    return MAGNETProfile(
        name="mock_profile",
        description="Mock configuration for testing without GPUs",
        settings=settings,
        agents=agents
    )


# Default configuration getter
def get_default_profile() -> MAGNETProfile:
    """
    Get default configuration profile based on environment.

    Checks MAGNET_PROFILE environment variable.
    """
    profile_name = os.getenv("MAGNET_PROFILE", "2xA40")

    if profile_name == "2xA40":
        return get_2xA40_profile()
    elif profile_name == "1xA40":
        return get_1xA40_profile()
    elif profile_name == "mock":
        return get_mock_profile()
    else:
        print(f"Unknown profile '{profile_name}', using 2xA40 as default")
        return get_2xA40_profile()


# Export commonly used config
CONFIG_2xA40 = get_2xA40_profile()
CONFIG_1xA40 = get_1xA40_profile()
CONFIG_MOCK = get_mock_profile()
