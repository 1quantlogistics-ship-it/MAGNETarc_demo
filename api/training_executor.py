"""
Training Job Executor: Submit and monitor autonomous training experiments
===========================================================================

Bridges multi-agent decisions to actual training execution:
- Submits approved proposals as training jobs
- Monitors job progress (polling or async)
- Collects experiment results
- Feeds results back to Historian for learning

This is the critical "hands" that enable autonomous operation.
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from config.experiment_config_generator import get_config_generator, ConfigValidationError


class JobStatus(Enum):
    """Training job states."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Represents a submitted training job."""
    job_id: str
    experiment_id: str
    config: Dict[str, Any]
    status: JobStatus
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class TrainingExecutionError(Exception):
    """Raised when training job execution fails."""
    pass


class TrainingExecutor:
    """
    Executes approved training jobs and monitors completion.

    Workflow:
    1. Generate config from proposal
    2. Submit to control plane /train endpoint
    3. Monitor job status
    4. Collect results when complete
    5. Return metrics for Historian
    """

    def __init__(
        self,
        control_plane_url: str = "http://localhost:8000",
        experiments_dir: str = "/Users/bengibson/Desktop/ARC/arc_clean/experiments",
        memory_path: str = "/Users/bengibson/Desktop/ARC/arc_clean/memory",
        poll_interval: int = 10,
        max_concurrent_jobs: int = 3
    ):
        """
        Initialize training executor.

        Args:
            control_plane_url: URL of control plane API
            experiments_dir: Directory for experiment configs/results
            memory_path: Path to memory for constraints
            poll_interval: Seconds between status polls
            max_concurrent_jobs: Maximum parallel training jobs
        """
        self.control_plane_url = control_plane_url.rstrip('/')
        self.experiments_dir = Path(experiments_dir)
        self.memory_path = Path(memory_path)
        self.poll_interval = poll_interval
        self.max_concurrent_jobs = max_concurrent_jobs

        # Initialize config generator
        self.config_generator = get_config_generator(
            experiments_dir=str(self.experiments_dir),
            memory_path=str(self.memory_path)
        )

        # Track active jobs
        self.active_jobs: Dict[str, TrainingJob] = {}

        # Ensure directories exist
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def submit_job(
        self,
        proposal: Dict[str, Any],
        requires_approval: bool = False
    ) -> TrainingJob:
        """
        Submit training job from proposal.

        Args:
            proposal: Agent proposal with experiment_id and config_changes
            requires_approval: Whether to require human approval in SEMI mode

        Returns:
            TrainingJob object with submission details

        Raises:
            TrainingExecutionError: If submission fails
            ConfigValidationError: If config generation/validation fails
        """
        experiment_id = proposal.get("experiment_id")
        if not experiment_id:
            raise TrainingExecutionError("Proposal missing experiment_id")

        # Check concurrent job limit
        running_jobs = sum(1 for job in self.active_jobs.values()
                           if job.status in [JobStatus.QUEUED, JobStatus.RUNNING])
        if running_jobs >= self.max_concurrent_jobs:
            raise TrainingExecutionError(
                f"Max concurrent jobs ({self.max_concurrent_jobs}) reached. "
                f"Currently running: {running_jobs}"
            )

        try:
            # Generate validated config
            config = self.config_generator.generate_config(
                experiment_id=experiment_id,
                proposal=proposal,
                validate=True
            )

            # Submit to control plane
            response = requests.post(
                f"{self.control_plane_url}/train",
                json={
                    "experiment_id": experiment_id,
                    "config": config,
                    "requires_approval": requires_approval
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                job_status = JobStatus(result.get("status", "queued"))

                # Create job object
                job = TrainingJob(
                    job_id=experiment_id,  # Use experiment_id as job_id
                    experiment_id=experiment_id,
                    config=config,
                    status=job_status,
                    submitted_at=datetime.now().isoformat()
                )

                # Track job
                self.active_jobs[experiment_id] = job

                return job

            elif response.status_code == 400:
                # Validation error from control plane
                error_detail = response.json().get("detail", {})
                if isinstance(error_detail, dict):
                    issues = error_detail.get("issues", [])
                    raise ConfigValidationError(f"Control plane validation failed:\n" + "\n".join(issues))
                else:
                    raise ConfigValidationError(f"Control plane validation failed: {error_detail}")

            else:
                raise TrainingExecutionError(
                    f"Failed to submit job: HTTP {response.status_code} - {response.text}"
                )

        except requests.RequestException as e:
            raise TrainingExecutionError(f"Failed to connect to control plane: {e}")

    def submit_batch(
        self,
        proposals: List[Dict[str, Any]],
        requires_approval: bool = False
    ) -> List[TrainingJob]:
        """
        Submit multiple training jobs.

        Args:
            proposals: List of agent proposals
            requires_approval: Whether to require human approval

        Returns:
            List of submitted TrainingJob objects
        """
        submitted_jobs = []
        errors = []

        for proposal in proposals:
            try:
                job = self.submit_job(proposal, requires_approval=requires_approval)
                submitted_jobs.append(job)
            except (TrainingExecutionError, ConfigValidationError) as e:
                exp_id = proposal.get("experiment_id", "unknown")
                errors.append(f"{exp_id}: {str(e)}")

        if errors and not submitted_jobs:
            # All submissions failed
            raise TrainingExecutionError(f"All job submissions failed:\n" + "\n".join(errors))

        # Some succeeded
        return submitted_jobs

    def get_job_status(self, experiment_id: str) -> Optional[JobStatus]:
        """
        Get current status of training job.

        Args:
            experiment_id: Experiment identifier

        Returns:
            JobStatus or None if not found
        """
        if experiment_id in self.active_jobs:
            return self.active_jobs[experiment_id].status

        # Check if completed (results exist)
        exp_dir = self.experiments_dir / experiment_id
        results_path = exp_dir / "results.json"
        if results_path.exists():
            return JobStatus.COMPLETED

        return None

    def poll_job_status(self, experiment_id: str) -> JobStatus:
        """
        Poll control plane for job status update.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Updated JobStatus
        """
        try:
            # Check if results file exists (simplest status check)
            exp_dir = self.experiments_dir / experiment_id
            results_path = exp_dir / "results.json"

            if results_path.exists():
                # Job completed
                job = self.active_jobs.get(experiment_id)
                if job:
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now().isoformat()
                return JobStatus.COMPLETED

            # Check for error files
            error_path = exp_dir / "error.log"
            if error_path.exists():
                job = self.active_jobs.get(experiment_id)
                if job:
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now().isoformat()
                    with open(error_path, 'r') as f:
                        job.error = f.read()
                return JobStatus.FAILED

            # Otherwise assume still running
            job = self.active_jobs.get(experiment_id)
            if job and job.status == JobStatus.QUEUED:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now().isoformat()

            return job.status if job else JobStatus.RUNNING

        except Exception as e:
            print(f"Warning: Failed to poll status for {experiment_id}: {e}")
            return self.active_jobs.get(experiment_id).status if experiment_id in self.active_jobs else JobStatus.FAILED

    def wait_for_completion(
        self,
        experiment_ids: List[str],
        timeout: Optional[int] = 3600  # 1 hour default
    ) -> Dict[str, JobStatus]:
        """
        Wait for multiple jobs to complete.

        Args:
            experiment_ids: List of experiment identifiers to wait for
            timeout: Maximum seconds to wait (None = no timeout)

        Returns:
            Dict mapping experiment_id to final JobStatus
        """
        start_time = time.time()
        completed = {}

        while len(completed) < len(experiment_ids):
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                # Mark remaining as failed due to timeout
                for exp_id in experiment_ids:
                    if exp_id not in completed:
                        completed[exp_id] = JobStatus.FAILED
                        if exp_id in self.active_jobs:
                            self.active_jobs[exp_id].status = JobStatus.FAILED
                            self.active_jobs[exp_id].error = "Timeout"
                break

            # Poll each job
            for exp_id in experiment_ids:
                if exp_id in completed:
                    continue

                status = self.poll_job_status(exp_id)

                if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    completed[exp_id] = status

            # Sleep before next poll
            if len(completed) < len(experiment_ids):
                time.sleep(self.poll_interval)

        return completed

    def collect_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Collect results from completed experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dict containing experiment results and metrics

        Raises:
            TrainingExecutionError: If results not found or job not complete
        """
        job = self.active_jobs.get(experiment_id)
        if not job:
            raise TrainingExecutionError(f"Job {experiment_id} not found in active jobs")

        if job.status != JobStatus.COMPLETED:
            raise TrainingExecutionError(f"Job {experiment_id} not completed (status: {job.status.value})")

        # Load results from file
        exp_dir = self.experiments_dir / experiment_id
        results_path = exp_dir / "results.json"

        if not results_path.exists():
            raise TrainingExecutionError(f"Results file not found: {results_path}")

        try:
            with open(results_path, 'r') as f:
                results = json.load(f)

            # Update job with metrics
            job.metrics = results.get("metrics", {})

            # Return comprehensive results
            return {
                "experiment_id": experiment_id,
                "config": job.config,
                "metrics": job.metrics,
                "status": job.status.value,
                "submitted_at": job.submitted_at,
                "completed_at": job.completed_at,
                "duration_seconds": self._compute_duration(job),
                "proposal_type": job.config.get("proposal_type", "unknown"),
                "risk_level": job.config.get("risk_level", "unknown")
            }

        except json.JSONDecodeError as e:
            raise TrainingExecutionError(f"Failed to parse results JSON: {e}")

    def collect_batch_results(
        self,
        experiment_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Collect results from multiple experiments.

        Args:
            experiment_ids: List of experiment identifiers

        Returns:
            List of result dicts (skips failed experiments)
        """
        results = []
        for exp_id in experiment_ids:
            try:
                result = self.collect_results(exp_id)
                results.append(result)
            except TrainingExecutionError as e:
                print(f"Warning: Failed to collect results for {exp_id}: {e}")
                # Log failed result
                job = self.active_jobs.get(exp_id)
                if job:
                    results.append({
                        "experiment_id": exp_id,
                        "status": job.status.value,
                        "error": str(e),
                        "metrics": {}
                    })

        return results

    def get_active_jobs(self) -> List[TrainingJob]:
        """Get list of currently active jobs."""
        return [
            job for job in self.active_jobs.values()
            if job.status in [JobStatus.QUEUED, JobStatus.RUNNING]
        ]

    def get_completed_jobs(self) -> List[TrainingJob]:
        """Get list of completed jobs."""
        return [
            job for job in self.active_jobs.values()
            if job.status == JobStatus.COMPLETED
        ]

    def get_failed_jobs(self) -> List[TrainingJob]:
        """Get list of failed jobs."""
        return [
            job for job in self.active_jobs.values()
            if job.status == JobStatus.FAILED
        ]

    def cancel_job(self, experiment_id: str) -> bool:
        """
        Cancel running job (if supported by training infrastructure).

        Args:
            experiment_id: Experiment identifier

        Returns:
            True if cancelled successfully
        """
        job = self.active_jobs.get(experiment_id)
        if not job:
            return False

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            # Already finished
            return False

        # Mark as cancelled
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now().isoformat()

        # TODO: Implement actual process termination if training runs locally
        # For now, just mark as cancelled

        return True

    def cleanup_job(self, experiment_id: str) -> None:
        """Remove job from active tracking."""
        if experiment_id in self.active_jobs:
            del self.active_jobs[experiment_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics."""
        total_jobs = len(self.active_jobs)
        completed = sum(1 for j in self.active_jobs.values() if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self.active_jobs.values() if j.status == JobStatus.FAILED)
        running = sum(1 for j in self.active_jobs.values() if j.status in [JobStatus.QUEUED, JobStatus.RUNNING])

        return {
            "total_jobs": total_jobs,
            "completed": completed,
            "failed": failed,
            "running": running,
            "success_rate": completed / total_jobs if total_jobs > 0 else 0.0,
            "max_concurrent": self.max_concurrent_jobs
        }

    def _compute_duration(self, job: TrainingJob) -> Optional[float]:
        """Compute job duration in seconds."""
        if not job.completed_at or not job.submitted_at:
            return None

        try:
            start = datetime.fromisoformat(job.submitted_at)
            end = datetime.fromisoformat(job.completed_at)
            return (end - start).total_seconds()
        except Exception:
            return None


def get_training_executor(
    control_plane_url: Optional[str] = None,
    experiments_dir: Optional[str] = None,
    memory_path: Optional[str] = None,
    poll_interval: Optional[int] = None,
    max_concurrent_jobs: Optional[int] = None
) -> TrainingExecutor:
    """
    Factory function to get training executor instance.

    Args:
        control_plane_url: Optional custom control plane URL
        experiments_dir: Optional custom experiments directory
        memory_path: Optional custom memory path
        poll_interval: Optional custom poll interval
        max_concurrent_jobs: Optional custom concurrent job limit

    Returns:
        TrainingExecutor instance
    """
    kwargs = {}
    if control_plane_url is not None:
        kwargs["control_plane_url"] = control_plane_url
    if experiments_dir is not None:
        kwargs["experiments_dir"] = experiments_dir
    if memory_path is not None:
        kwargs["memory_path"] = memory_path
    if poll_interval is not None:
        kwargs["poll_interval"] = poll_interval
    if max_concurrent_jobs is not None:
        kwargs["max_concurrent_jobs"] = max_concurrent_jobs

    return TrainingExecutor(**kwargs)
