"""
Role-based LLM Response Fixtures

Pre-defined realistic responses for each ARC role to use in testing.
These fixtures simulate actual LLM outputs for different scenarios.
"""

from datetime import datetime
from typing import Dict, Any, List


# ============================================================================
# Historian Responses
# ============================================================================

HISTORIAN_UPDATE_BASELINE = {
    "history_summary": {
        "total_cycles": 1,
        "total_experiments": 3,
        "best_metrics": {
            "auc": 0.78,
            "sensitivity": 0.75,
            "specificity": 0.81
        },
        "recent_experiments": [
            {
                "experiment_id": "exp_1_1",
                "auc": 0.78,
                "sensitivity": 0.75,
                "specificity": 0.81,
                "training_time": 120.5,
                "timestamp": datetime.utcnow().isoformat(),
                "success": True
            }
        ],
        "failed_configs": [],
        "successful_patterns": [],
        "performance_trends": {
            "auc_trend": "unknown",
            "sensitivity_trend": "unknown",
            "specificity_trend": "unknown",
            "cycles_without_improvement": 0,
            "consecutive_regressions": 0
        },
        "last_updated": datetime.utcnow().isoformat()
    },
    "constraints": {
        "forbidden_ranges": [],
        "unstable_configs": [],
        "safe_baselines": [],
        "max_learning_rate": 1.0,
        "min_batch_size": 1,
        "max_batch_size": 512,
        "last_updated": datetime.utcnow().isoformat()
    }
}

HISTORIAN_UPDATE_WITH_IMPROVEMENT = {
    "history_summary": {
        "total_cycles": 5,
        "total_experiments": 15,
        "best_metrics": {
            "auc": 0.89,
            "sensitivity": 0.87,
            "specificity": 0.91
        },
        "recent_experiments": [
            {
                "experiment_id": "exp_5_3",
                "auc": 0.89,
                "sensitivity": 0.87,
                "specificity": 0.91,
                "training_time": 135.2,
                "timestamp": datetime.utcnow().isoformat(),
                "success": True
            },
            {
                "experiment_id": "exp_5_2",
                "auc": 0.86,
                "sensitivity": 0.84,
                "specificity": 0.88,
                "training_time": 128.7,
                "timestamp": datetime.utcnow().isoformat(),
                "success": True
            }
        ],
        "failed_configs": [],
        "successful_patterns": [
            {"learning_rate": 0.001, "batch_size": 32},
            {"learning_rate": 0.0008, "batch_size": 32}
        ],
        "performance_trends": {
            "auc_trend": "improving",
            "sensitivity_trend": "improving",
            "specificity_trend": "stable",
            "cycles_without_improvement": 0,
            "consecutive_regressions": 0
        },
        "last_updated": datetime.utcnow().isoformat()
    },
    "constraints": {
        "forbidden_ranges": [],
        "unstable_configs": [],
        "safe_baselines": [{"learning_rate": 0.001, "batch_size": 32}],
        "max_learning_rate": 0.1,
        "min_batch_size": 8,
        "max_batch_size": 256,
        "last_updated": datetime.utcnow().isoformat()
    }
}

HISTORIAN_UPDATE_STAGNANT = {
    "history_summary": {
        "total_cycles": 15,
        "total_experiments": 45,
        "best_metrics": {
            "auc": 0.85,
            "sensitivity": 0.83,
            "specificity": 0.87
        },
        "performance_trends": {
            "auc_trend": "stable",
            "sensitivity_trend": "stable",
            "specificity_trend": "stable",
            "cycles_without_improvement": 12,
            "consecutive_regressions": 0
        },
        "last_updated": datetime.utcnow().isoformat()
    }
}

HISTORIAN_UPDATE_REGRESSING = {
    "history_summary": {
        "total_cycles": 8,
        "total_experiments": 24,
        "best_metrics": {
            "auc": 0.88,
            "sensitivity": 0.86,
            "specificity": 0.90
        },
        "performance_trends": {
            "auc_trend": "declining",
            "sensitivity_trend": "declining",
            "specificity_trend": "stable",
            "cycles_without_improvement": 3,
            "consecutive_regressions": 3
        },
        "last_updated": datetime.utcnow().isoformat()
    },
    "constraints": {
        "forbidden_ranges": [
            {
                "param": "learning_rate",
                "min": 0.01,
                "max": 1.0,
                "reason": "Recent failures with high learning rates"
            }
        ],
        "unstable_configs": [
            {"learning_rate": 0.1, "batch_size": 16}
        ],
        "safe_baselines": [
            {"learning_rate": 0.001, "batch_size": 32}
        ],
        "last_updated": datetime.utcnow().isoformat()
    }
}


# ============================================================================
# Director Responses
# ============================================================================

DIRECTOR_DIRECTIVE_EXPLORE = {
    "cycle_id": 5,
    "mode": "explore",
    "objective": "improve_auc",
    "novelty_budget": {
        "exploit": 2,
        "explore": 3,
        "wildcat": 1
    },
    "focus_areas": ["architecture", "optimizer", "learning_rate"],
    "forbidden_axes": ["dataset"],
    "encouraged_axes": ["dropout", "weight_decay", "augmentation"],
    "notes": "Strong upward trend. Safe to explore architectural changes while maintaining exploit baseline.",
    "timestamp": datetime.utcnow().isoformat()
}

DIRECTOR_DIRECTIVE_EXPLOIT = {
    "cycle_id": 6,
    "mode": "exploit",
    "objective": "improve_sensitivity",
    "novelty_budget": {
        "exploit": 5,
        "explore": 1,
        "wildcat": 0
    },
    "focus_areas": ["learning_rate", "batch_size"],
    "forbidden_axes": ["architecture", "dataset"],
    "encouraged_axes": ["learning_rate", "optimizer_params"],
    "notes": "Reached good AUC plateau. Focus on sensitivity refinement with low-risk experiments.",
    "timestamp": datetime.utcnow().isoformat()
}

DIRECTOR_DIRECTIVE_RECOVER = {
    "cycle_id": 9,
    "mode": "recover",
    "objective": "improve_auc",
    "novelty_budget": {
        "exploit": 4,
        "explore": 0,
        "wildcat": 0
    },
    "focus_areas": ["learning_rate"],
    "forbidden_axes": ["architecture", "loss_function", "optimizer"],
    "encouraged_axes": [],
    "notes": "Three consecutive regressions detected. Switching to recovery mode. Revert to safe baselines.",
    "timestamp": datetime.utcnow().isoformat()
}

DIRECTOR_DIRECTIVE_WILDCAT = {
    "cycle_id": 20,
    "mode": "wildcat",
    "objective": "improve_auc",
    "novelty_budget": {
        "exploit": 1,
        "explore": 2,
        "wildcat": 3
    },
    "focus_areas": ["architecture", "loss_function", "data_pipeline"],
    "forbidden_axes": [],
    "encouraged_axes": ["novel_architectures", "custom_losses", "advanced_augmentation"],
    "notes": "Prolonged stagnation (15 cycles). Time for aggressive exploration. High-risk acceptable.",
    "timestamp": datetime.utcnow().isoformat()
}


# ============================================================================
# Architect Responses
# ============================================================================

ARCHITECT_PROPOSALS_CONSERVATIVE = {
    "cycle_id": 6,
    "proposals": [
        {
            "experiment_id": "exp_6_1",
            "novelty_class": "exploit",
            "hypothesis": "Slightly reducing learning rate will improve convergence stability",
            "changes": {"learning_rate": 0.0009},
            "expected_impact": {
                "auc": "up",
                "sensitivity": "same",
                "specificity": "same"
            },
            "resource_cost": "low",
            "rationale": "Current LR=0.001 performs well, minor reduction may reduce variance"
        },
        {
            "experiment_id": "exp_6_2",
            "novelty_class": "exploit",
            "hypothesis": "Increasing batch size will stabilize gradients",
            "changes": {"batch_size": 48},
            "expected_impact": {
                "auc": "up",
                "sensitivity": "same",
                "specificity": "same"
            },
            "resource_cost": "low",
            "rationale": "Larger batches reduce gradient noise without architecture changes"
        }
    ],
    "timestamp": datetime.utcnow().isoformat()
}

ARCHITECT_PROPOSALS_BALANCED = {
    "cycle_id": 7,
    "proposals": [
        {
            "experiment_id": "exp_7_1",
            "novelty_class": "exploit",
            "hypothesis": "Fine-tune weight decay for better regularization",
            "changes": {"weight_decay": 0.0001},
            "expected_impact": {
                "auc": "up",
                "sensitivity": "up",
                "specificity": "same"
            },
            "resource_cost": "low",
            "rationale": "Literature shows this range effective for medical imaging"
        },
        {
            "experiment_id": "exp_7_2",
            "novelty_class": "explore",
            "hypothesis": "Adding dropout will reduce overfitting",
            "changes": {"dropout": 0.3, "learning_rate": 0.001},
            "expected_impact": {
                "auc": "up",
                "sensitivity": "up",
                "specificity": "up"
            },
            "resource_cost": "medium",
            "rationale": "Model may be overfitting based on train/val gap"
        },
        {
            "experiment_id": "exp_7_3",
            "novelty_class": "explore",
            "hypothesis": "Switching to cosine annealing schedule may improve final performance",
            "changes": {"lr_scheduler": "cosine", "learning_rate": 0.001, "epochs": 75},
            "expected_impact": {
                "auc": "up",
                "sensitivity": "unknown",
                "specificity": "unknown"
            },
            "resource_cost": "medium",
            "rationale": "Cosine schedules often yield better convergence in vision tasks"
        }
    ],
    "timestamp": datetime.utcnow().isoformat()
}

ARCHITECT_PROPOSALS_AGGRESSIVE = {
    "cycle_id": 20,
    "proposals": [
        {
            "experiment_id": "exp_20_1",
            "novelty_class": "wildcat",
            "hypothesis": "EfficientNetV2 may significantly boost performance",
            "changes": {
                "architecture": "efficientnet_v2_s",
                "learning_rate": 0.001,
                "batch_size": 64,
                "input_size": 384
            },
            "expected_impact": {
                "auc": "up",
                "sensitivity": "up",
                "specificity": "up"
            },
            "resource_cost": "high",
            "rationale": "Modern architecture with proven medical imaging results"
        },
        {
            "experiment_id": "exp_20_2",
            "novelty_class": "wildcat",
            "hypothesis": "Custom focal loss may better handle class imbalance",
            "changes": {
                "loss_function": "focal_loss",
                "focal_alpha": 0.25,
                "focal_gamma": 2.0
            },
            "expected_impact": {
                "auc": "unknown",
                "sensitivity": "up",
                "specificity": "up"
            },
            "resource_cost": "medium",
            "rationale": "Focal loss specifically designed for imbalanced datasets"
        },
        {
            "experiment_id": "exp_20_3",
            "novelty_class": "explore",
            "hypothesis": "MixUp augmentation will improve robustness",
            "changes": {
                "use_mixup": True,
                "mixup_alpha": 0.2,
                "learning_rate": 0.001
            },
            "expected_impact": {
                "auc": "up",
                "sensitivity": "same",
                "specificity": "up"
            },
            "resource_cost": "low",
            "rationale": "MixUp reduces overfitting in medical imaging"
        }
    ],
    "timestamp": datetime.utcnow().isoformat()
}


# ============================================================================
# Critic Responses
# ============================================================================

CRITIC_REVIEWS_ALL_APPROVED = {
    "cycle_id": 6,
    "reviews": [
        {
            "proposal_id": "exp_6_1",
            "decision": "approve",
            "issues": [],
            "reasoning": "Safe, incremental change. Well-justified by historical data.",
            "risk_level": "low"
        },
        {
            "proposal_id": "exp_6_2",
            "decision": "approve",
            "issues": [],
            "reasoning": "Low-risk exploit with clear theoretical basis.",
            "risk_level": "low"
        }
    ],
    "approved": ["exp_6_1", "exp_6_2"],
    "rejected": [],
    "revise": [],
    "timestamp": datetime.utcnow().isoformat()
}

CRITIC_REVIEWS_MIXED = {
    "cycle_id": 7,
    "reviews": [
        {
            "proposal_id": "exp_7_1",
            "decision": "approve",
            "issues": [],
            "reasoning": "Well-researched parameter choice. Low risk.",
            "risk_level": "low"
        },
        {
            "proposal_id": "exp_7_2",
            "decision": "approve",
            "issues": ["May slow training by 10%"],
            "reasoning": "Good hypothesis. Resource cost acceptable given potential benefit.",
            "risk_level": "medium"
        },
        {
            "proposal_id": "exp_7_3",
            "decision": "revise",
            "issues": [
                "Requires 50% more epochs",
                "Untested on this dataset",
                "May interact poorly with current optimizer"
            ],
            "reasoning": "Interesting idea but needs pilot study. Suggest shorter trial first.",
            "risk_level": "medium"
        }
    ],
    "approved": ["exp_7_1", "exp_7_2"],
    "rejected": [],
    "revise": ["exp_7_3"],
    "timestamp": datetime.utcnow().isoformat()
}

CRITIC_REVIEWS_STRICT = {
    "cycle_id": 20,
    "reviews": [
        {
            "proposal_id": "exp_20_1",
            "decision": "reject",
            "issues": [
                "Extremely high resource cost",
                "Requires 3x memory",
                "Training time >4 hours",
                "No guarantee of improvement"
            ],
            "reasoning": "Too risky given resource constraints. Suggest starting with EfficientNet-B0.",
            "risk_level": "high"
        },
        {
            "proposal_id": "exp_20_2",
            "decision": "revise",
            "issues": [
                "Focal loss hyperparameters not justified",
                "May destabilize training",
                "Requires careful tuning"
            ],
            "reasoning": "Good concept but needs more conservative parameters. Suggest alpha=0.5, gamma=1.0 first.",
            "risk_level": "high"
        },
        {
            "proposal_id": "exp_20_3",
            "decision": "approve",
            "issues": ["Slight increase in training time"],
            "reasoning": "Well-justified augmentation technique. Low risk, proven approach.",
            "risk_level": "low"
        }
    ],
    "approved": ["exp_20_3"],
    "rejected": ["exp_20_1"],
    "revise": ["exp_20_2"],
    "timestamp": datetime.utcnow().isoformat()
}


# ============================================================================
# Executor Responses
# ============================================================================

EXECUTOR_JOBS_SIMPLE = {
    "experiments": [
        {
            "experiment_id": "exp_6_1",
            "config_patch": {"learning_rate": 0.0009},
            "commands": [
                "python train.py --config experiments/exp_6_1/config.yaml --gpu 0",
                "python evaluate.py --experiment exp_6_1"
            ],
            "status": "ready",
            "estimated_time": 120
        }
    ],
    "validation": "passed",
    "timestamp": datetime.utcnow().isoformat()
}

EXECUTOR_JOBS_MULTIPLE = {
    "experiments": [
        {
            "experiment_id": "exp_7_1",
            "config_patch": {"weight_decay": 0.0001},
            "commands": [
                "python train.py --config experiments/exp_7_1/config.yaml --gpu 0",
                "python evaluate.py --experiment exp_7_1"
            ],
            "status": "ready",
            "estimated_time": 125
        },
        {
            "experiment_id": "exp_7_2",
            "config_patch": {"dropout": 0.3, "learning_rate": 0.001},
            "commands": [
                "python train.py --config experiments/exp_7_2/config.yaml --gpu 0",
                "python evaluate.py --experiment exp_7_2"
            ],
            "status": "ready",
            "estimated_time": 140
        }
    ],
    "validation": "passed",
    "timestamp": datetime.utcnow().isoformat()
}


# ============================================================================
# Fixture Registry
# ============================================================================

FIXTURES = {
    "historian": {
        "baseline": HISTORIAN_UPDATE_BASELINE,
        "improvement": HISTORIAN_UPDATE_WITH_IMPROVEMENT,
        "stagnant": HISTORIAN_UPDATE_STAGNANT,
        "regressing": HISTORIAN_UPDATE_REGRESSING
    },
    "director": {
        "explore": DIRECTOR_DIRECTIVE_EXPLORE,
        "exploit": DIRECTOR_DIRECTIVE_EXPLOIT,
        "recover": DIRECTOR_DIRECTIVE_RECOVER,
        "wildcat": DIRECTOR_DIRECTIVE_WILDCAT
    },
    "architect": {
        "conservative": ARCHITECT_PROPOSALS_CONSERVATIVE,
        "balanced": ARCHITECT_PROPOSALS_BALANCED,
        "aggressive": ARCHITECT_PROPOSALS_AGGRESSIVE
    },
    "critic": {
        "all_approved": CRITIC_REVIEWS_ALL_APPROVED,
        "mixed": CRITIC_REVIEWS_MIXED,
        "strict": CRITIC_REVIEWS_STRICT
    },
    "executor": {
        "simple": EXECUTOR_JOBS_SIMPLE,
        "multiple": EXECUTOR_JOBS_MULTIPLE
    }
}


def get_fixture(role: str, scenario: str) -> Dict[str, Any]:
    """
    Get a pre-defined fixture for a role and scenario.

    Args:
        role: ARC role (historian, director, architect, critic, executor)
        scenario: Scenario name (varies by role)

    Returns:
        Dictionary with fixture data

    Raises:
        KeyError: If role or scenario not found
    """
    return FIXTURES[role][scenario]


def list_fixtures(role: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List available fixtures.

    Args:
        role: Optional role to filter by

    Returns:
        Dictionary mapping roles to available scenarios
    """
    if role:
        return {role: list(FIXTURES.get(role, {}).keys())}
    return {role: list(scenarios.keys()) for role, scenarios in FIXTURES.items()}
