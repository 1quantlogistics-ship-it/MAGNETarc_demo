"""
Failure Prediction and Recovery System for ARC.

This module implements predictive failure detection and automated recovery mechanisms
to prevent wasted compute and enable resilient training cycles.

Key Features:
- Gradient explosion/vanishing detection
- Loss spike prediction via exponential moving average
- Training instability detection (oscillation, plateaus)
- Automatic checkpoint recovery
- Resource exhaustion prediction (GPU memory, disk space)
- FDA-compliant logging of all failure events

Author: ARC Team
Created: 2025-11-18
"""

import json
import logging
import shutil
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy import stats

from config import get_settings
from tools.dev_logger import get_dev_logger

logger = logging.getLogger(__name__)


@dataclass
class FailurePrediction:
    """Result of failure prediction analysis."""
    failure_predicted: bool
    failure_type: str  # "gradient_explosion", "loss_spike", "instability", "resource_exhaustion"
    confidence: float  # 0.0 to 1.0
    severity: str  # "low", "medium", "high", "critical"
    details: Dict[str, Any]
    recommended_action: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RecoveryAction:
    """Recovery action result."""
    action_type: str  # "checkpoint_restore", "lr_reduction", "gradient_clip", "early_stop"
    success: bool
    details: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class FailurePredictor:
    """
    Predictive failure detection and automated recovery for ARC training.

    Monitors training metrics in real-time to predict failures before they occur,
    enabling proactive intervention and checkpoint-based recovery.
    """

    def __init__(
        self,
        window_size: int = 20,
        loss_spike_threshold: float = 2.0,
        gradient_explosion_threshold: float = 10.0,
        oscillation_threshold: float = 0.3,
        plateau_threshold: float = 0.001,
        min_gpu_memory_mb: int = 1024,
        min_disk_space_gb: int = 10
    ):
        """
        Initialize failure predictor.

        Args:
            window_size: Number of recent training steps to monitor
            loss_spike_threshold: Multiplier for loss spike detection (vs. EMA)
            gradient_explosion_threshold: Max gradient norm before explosion
            oscillation_threshold: Max std dev for stable training
            plateau_threshold: Min improvement rate to avoid plateau
            min_gpu_memory_mb: Minimum free GPU memory required (MB)
            min_disk_space_gb: Minimum free disk space required (GB)
        """
        self.window_size = window_size
        self.loss_spike_threshold = loss_spike_threshold
        self.gradient_explosion_threshold = gradient_explosion_threshold
        self.oscillation_threshold = oscillation_threshold
        self.plateau_threshold = plateau_threshold
        self.min_gpu_memory_mb = min_gpu_memory_mb
        self.min_disk_space_gb = min_disk_space_gb

        # Metric histories (circular buffers)
        self.loss_history: deque = deque(maxlen=window_size)
        self.gradient_norm_history: deque = deque(maxlen=window_size)
        self.accuracy_history: deque = deque(maxlen=window_size)

        # Exponential moving average for loss
        self.loss_ema: Optional[float] = None
        self.ema_alpha: float = 0.1  # Smoothing factor

        # Recovery state
        self.checkpoint_dir: Path = Path(get_settings().workspace_path) / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # FDA logging
        self.dev_logger = get_dev_logger()

        logger.info(f"FailurePredictor initialized with window_size={window_size}")

    def predict_gradient_explosion(
        self,
        gradient_norm: float,
        step: int,
        cycle_id: int
    ) -> FailurePrediction:
        """
        Predict gradient explosion before it crashes training.

        Uses gradient norm history and threshold-based detection.

        Args:
            gradient_norm: Current gradient norm
            step: Training step number
            cycle_id: Current research cycle ID

        Returns:
            FailurePrediction with explosion prediction and recommended action
        """
        self.gradient_norm_history.append(gradient_norm)

        # Check if gradient norm exceeds threshold
        explosion_detected = gradient_norm > self.gradient_explosion_threshold

        # Check for rapid gradient growth trend
        if len(self.gradient_norm_history) >= 5:
            recent_grads = list(self.gradient_norm_history)[-5:]
            slope, _, _, _, _ = stats.linregress(range(len(recent_grads)), recent_grads)
            rapid_growth = slope > 1.0  # Gradient growing by >1.0 per step
        else:
            rapid_growth = False

        failure_predicted = explosion_detected or rapid_growth

        if failure_predicted:
            severity = "critical" if explosion_detected else "high"
            confidence = 0.95 if explosion_detected else 0.75

            details = {
                "gradient_norm": float(gradient_norm),
                "threshold": self.gradient_explosion_threshold,
                "recent_norms": [float(g) for g in list(self.gradient_norm_history)[-5:]],
                "explosion_detected": explosion_detected,
                "rapid_growth": rapid_growth,
                "step": step
            }

            recommended_action = (
                "CRITICAL: Apply gradient clipping immediately and restore last checkpoint"
                if explosion_detected else
                "Reduce learning rate by 0.5x and enable gradient clipping"
            )

            # Log to FDA
            self._log_failure_to_fda(
                failure_type="gradient_explosion",
                severity=severity,
                cycle_id=cycle_id,
                details=details
            )

            return FailurePrediction(
                failure_predicted=True,
                failure_type="gradient_explosion",
                confidence=confidence,
                severity=severity,
                details=details,
                recommended_action=recommended_action
            )

        return FailurePrediction(
            failure_predicted=False,
            failure_type="gradient_explosion",
            confidence=0.0,
            severity="low",
            details={"gradient_norm": float(gradient_norm), "step": step},
            recommended_action="Continue training normally"
        )

    def predict_loss_spike(
        self,
        current_loss: float,
        step: int,
        cycle_id: int
    ) -> FailurePrediction:
        """
        Predict loss spike using exponential moving average.

        Detects sudden loss increases that indicate training instability.

        Args:
            current_loss: Current training loss value
            step: Training step number
            cycle_id: Current research cycle ID

        Returns:
            FailurePrediction with spike detection and recommended action
        """
        self.loss_history.append(current_loss)

        # Initialize EMA on first loss
        if self.loss_ema is None:
            self.loss_ema = current_loss
            return FailurePrediction(
                failure_predicted=False,
                failure_type="loss_spike",
                confidence=0.0,
                severity="low",
                details={"current_loss": float(current_loss), "step": step},
                recommended_action="Continue training (EMA initialized)"
            )

        # Update EMA: EMA_t = α * Loss_t + (1 - α) * EMA_{t-1}
        self.loss_ema = self.ema_alpha * current_loss + (1 - self.ema_alpha) * self.loss_ema

        # Detect spike: current loss significantly exceeds EMA
        spike_ratio = current_loss / self.loss_ema if self.loss_ema > 0 else 1.0
        spike_detected = spike_ratio > self.loss_spike_threshold

        if spike_detected:
            severity = "critical" if spike_ratio > 3.0 else "high"
            confidence = min(0.95, 0.5 + (spike_ratio - self.loss_spike_threshold) / 5.0)

            details = {
                "current_loss": float(current_loss),
                "loss_ema": float(self.loss_ema),
                "spike_ratio": float(spike_ratio),
                "threshold": self.loss_spike_threshold,
                "recent_losses": [float(l) for l in list(self.loss_history)[-5:]],
                "step": step
            }

            recommended_action = (
                "CRITICAL: Restore last checkpoint immediately"
                if spike_ratio > 3.0 else
                "Reduce learning rate by 0.5x and monitor next 10 steps"
            )

            # Log to FDA
            self._log_failure_to_fda(
                failure_type="loss_spike",
                severity=severity,
                cycle_id=cycle_id,
                details=details
            )

            return FailurePrediction(
                failure_predicted=True,
                failure_type="loss_spike",
                confidence=confidence,
                severity=severity,
                details=details,
                recommended_action=recommended_action
            )

        return FailurePrediction(
            failure_predicted=False,
            failure_type="loss_spike",
            confidence=0.0,
            severity="low",
            details={
                "current_loss": float(current_loss),
                "loss_ema": float(self.loss_ema),
                "step": step
            },
            recommended_action="Continue training normally"
        )

    def predict_instability(
        self,
        current_metric: float,
        metric_name: str,
        step: int,
        cycle_id: int
    ) -> FailurePrediction:
        """
        Predict training instability via oscillation or plateau detection.

        Monitors metric variance (oscillation) and improvement rate (plateau).

        Args:
            current_metric: Current metric value (e.g., accuracy, AUC)
            metric_name: Name of metric being monitored
            step: Training step number
            cycle_id: Current research cycle ID

        Returns:
            FailurePrediction with instability detection and recommended action
        """
        self.accuracy_history.append(current_metric)

        if len(self.accuracy_history) < 10:
            return FailurePrediction(
                failure_predicted=False,
                failure_type="instability",
                confidence=0.0,
                severity="low",
                details={
                    "metric_name": metric_name,
                    "current_value": float(current_metric),
                    "step": step
                },
                recommended_action="Continue training (collecting history)"
            )

        recent_metrics = list(self.accuracy_history)[-10:]

        # Oscillation detection: high variance in recent metrics
        std_dev = np.std(recent_metrics)
        mean_val = np.mean(recent_metrics)
        cv = std_dev / mean_val if mean_val > 0 else 0  # Coefficient of variation
        oscillation_detected = cv > self.oscillation_threshold

        # Plateau detection: minimal improvement over window
        slope, _, _, _, _ = stats.linregress(range(len(recent_metrics)), recent_metrics)
        plateau_detected = abs(slope) < self.plateau_threshold

        instability_detected = oscillation_detected or plateau_detected

        if instability_detected:
            if oscillation_detected:
                failure_type_detail = "oscillation"
                severity = "medium"
                confidence = 0.7
                recommended_action = "Reduce learning rate by 0.5x to stabilize training"
            else:  # plateau_detected
                failure_type_detail = "plateau"
                severity = "medium"
                confidence = 0.65
                recommended_action = "Consider early stopping or learning rate warmup restart"

            details = {
                "metric_name": metric_name,
                "current_value": float(current_metric),
                "mean": float(mean_val),
                "std_dev": float(std_dev),
                "cv": float(cv),
                "slope": float(slope),
                "oscillation_detected": oscillation_detected,
                "plateau_detected": plateau_detected,
                "recent_values": [float(m) for m in recent_metrics],
                "step": step
            }

            # Log to FDA
            self._log_failure_to_fda(
                failure_type=f"instability_{failure_type_detail}",
                severity=severity,
                cycle_id=cycle_id,
                details=details
            )

            return FailurePrediction(
                failure_predicted=True,
                failure_type="instability",
                confidence=confidence,
                severity=severity,
                details=details,
                recommended_action=recommended_action
            )

        return FailurePrediction(
            failure_predicted=False,
            failure_type="instability",
            confidence=0.0,
            severity="low",
            details={
                "metric_name": metric_name,
                "current_value": float(current_metric),
                "cv": float(cv),
                "slope": float(slope),
                "step": step
            },
            recommended_action="Continue training normally"
        )

    def predict_resource_exhaustion(
        self,
        cycle_id: int
    ) -> FailurePrediction:
        """
        Predict resource exhaustion (GPU memory, disk space).

        Checks current resource availability against safety thresholds.

        Args:
            cycle_id: Current research cycle ID

        Returns:
            FailurePrediction with resource exhaustion prediction
        """
        try:
            import torch

            # Check GPU memory if available
            gpu_exhaustion = False
            gpu_details = {}

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    free_memory_mb = torch.cuda.mem_get_info(i)[0] / (1024 ** 2)
                    total_memory_mb = torch.cuda.mem_get_info(i)[1] / (1024 ** 2)
                    used_memory_mb = total_memory_mb - free_memory_mb

                    gpu_details[f"gpu_{i}"] = {
                        "free_mb": float(free_memory_mb),
                        "used_mb": float(used_memory_mb),
                        "total_mb": float(total_memory_mb),
                        "utilization": float(used_memory_mb / total_memory_mb)
                    }

                    if free_memory_mb < self.min_gpu_memory_mb:
                        gpu_exhaustion = True

            # Check disk space
            workspace_stat = shutil.disk_usage(get_settings().workspace_path)
            free_disk_gb = workspace_stat.free / (1024 ** 3)
            total_disk_gb = workspace_stat.total / (1024 ** 3)
            used_disk_gb = workspace_stat.used / (1024 ** 3)

            disk_exhaustion = free_disk_gb < self.min_disk_space_gb

            disk_details = {
                "free_gb": float(free_disk_gb),
                "used_gb": float(used_disk_gb),
                "total_gb": float(total_disk_gb),
                "utilization": float(used_disk_gb / total_disk_gb)
            }

            exhaustion_detected = gpu_exhaustion or disk_exhaustion

            if exhaustion_detected:
                severity = "critical" if (gpu_exhaustion and disk_exhaustion) else "high"
                confidence = 0.9

                if gpu_exhaustion and disk_exhaustion:
                    recommended_action = "CRITICAL: Free GPU memory and disk space immediately"
                elif gpu_exhaustion:
                    recommended_action = "Free GPU memory: clear cache, reduce batch size"
                else:
                    recommended_action = "Free disk space: clean old checkpoints and logs"

                details = {
                    "gpu_exhaustion": gpu_exhaustion,
                    "disk_exhaustion": disk_exhaustion,
                    "gpu_details": gpu_details,
                    "disk_details": disk_details
                }

                # Log to FDA
                self._log_failure_to_fda(
                    failure_type="resource_exhaustion",
                    severity=severity,
                    cycle_id=cycle_id,
                    details=details
                )

                return FailurePrediction(
                    failure_predicted=True,
                    failure_type="resource_exhaustion",
                    confidence=confidence,
                    severity=severity,
                    details=details,
                    recommended_action=recommended_action
                )

            return FailurePrediction(
                failure_predicted=False,
                failure_type="resource_exhaustion",
                confidence=0.0,
                severity="low",
                details={
                    "gpu_details": gpu_details,
                    "disk_details": disk_details
                },
                recommended_action="Continue training normally"
            )

        except Exception as e:
            logger.warning(f"Resource exhaustion check failed: {e}")
            return FailurePrediction(
                failure_predicted=False,
                failure_type="resource_exhaustion",
                confidence=0.0,
                severity="low",
                details={"error": str(e)},
                recommended_action="Unable to check resources"
            )

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        cycle_id: int,
        step: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save training checkpoint for recovery.

        Args:
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            cycle_id: Current research cycle ID
            step: Training step number
            metadata: Optional additional metadata

        Returns:
            Path to saved checkpoint
        """
        import torch

        checkpoint_name = f"checkpoint_cycle{cycle_id}_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint_data = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "cycle_id": cycle_id,
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        torch.save(checkpoint_data, checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Log to FDA
        self.dev_logger.log_experiment(
            experiment_type="checkpoint_save",
            cycle_id=cycle_id,
            details={
                "checkpoint_path": str(checkpoint_path),
                "step": step,
                "size_mb": checkpoint_path.stat().st_size / (1024 ** 2)
            }
        )

        return str(checkpoint_path)

    def restore_checkpoint(
        self,
        checkpoint_path: str,
        cycle_id: int
    ) -> RecoveryAction:
        """
        Restore training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            cycle_id: Current research cycle ID

        Returns:
            RecoveryAction with restoration result
        """
        import torch

        try:
            checkpoint_data = torch.load(checkpoint_path)

            details = {
                "checkpoint_path": checkpoint_path,
                "restored_cycle": checkpoint_data["cycle_id"],
                "restored_step": checkpoint_data["step"],
                "timestamp": checkpoint_data["timestamp"]
            }

            logger.info(f"Checkpoint restored from {checkpoint_path}")

            # Log to FDA
            self.dev_logger.log_experiment(
                experiment_type="checkpoint_restore",
                cycle_id=cycle_id,
                details=details
            )

            return RecoveryAction(
                action_type="checkpoint_restore",
                success=True,
                details=details
            )

        except Exception as e:
            logger.error(f"Failed to restore checkpoint from {checkpoint_path}: {e}")

            return RecoveryAction(
                action_type="checkpoint_restore",
                success=False,
                details={
                    "checkpoint_path": checkpoint_path,
                    "error": str(e)
                }
            )

    def _log_failure_to_fda(
        self,
        failure_type: str,
        severity: str,
        cycle_id: int,
        details: Dict[str, Any]
    ) -> None:
        """Log failure prediction to FDA development logs."""
        self.dev_logger.log_risk_event(
            event_type=f"failure_prediction_{failure_type}",
            severity=severity,
            description=f"Predicted {failure_type} failure during training",
            cycle_id=cycle_id,
            context=details
        )


# Singleton instance
_failure_predictor_instance: Optional[FailurePredictor] = None


def get_failure_predictor(
    window_size: int = 20,
    loss_spike_threshold: float = 2.0,
    gradient_explosion_threshold: float = 10.0
) -> FailurePredictor:
    """
    Get singleton failure predictor instance.

    Args:
        window_size: Number of recent steps to monitor
        loss_spike_threshold: Multiplier for loss spike detection
        gradient_explosion_threshold: Max gradient norm

    Returns:
        Global FailurePredictor instance
    """
    global _failure_predictor_instance

    if _failure_predictor_instance is None:
        _failure_predictor_instance = FailurePredictor(
            window_size=window_size,
            loss_spike_threshold=loss_spike_threshold,
            gradient_explosion_threshold=gradient_explosion_threshold
        )

    return _failure_predictor_instance
