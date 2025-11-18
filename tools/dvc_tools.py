"""
DVC (Data Version Control) Tools

Manages dataset versioning, tracking, and synchronization using DVC.

Features:
- Dataset registration with DVC
- SHA256 hash tracking
- Remote push/pull operations
- Data registry management
- Integration with tool governance

DVC enables:
- Dataset reproducibility
- Version control for large files
- Remote storage synchronization
- Audit trail for datasets
"""

import os
import subprocess
import json
import hashlib
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DVCError(Exception):
    """Raised when DVC operations fail."""
    pass


def register_dataset_with_dvc(
    dataset_dir: str,
    dataset_name: str,
    remote_name: str = "origin",
    push: bool = True
) -> Dict[str, Any]:
    """
    Register dataset with DVC and optionally push to remote.

    Args:
        dataset_dir: Path to dataset directory
        dataset_name: Dataset identifier
        remote_name: DVC remote name (default: "origin")
        push: Whether to push to remote after adding

    Returns:
        Dict with registration results

    Raises:
        DVCError: If DVC operations fail
    """
    logger.info(f"Registering dataset with DVC: {dataset_name}")

    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise DVCError(f"Dataset directory does not exist: {dataset_dir}")

    try:
        # Initialize DVC if not already initialized
        _ensure_dvc_initialized(dataset_path.parent)

        # Add dataset to DVC
        logger.info(f"Adding {dataset_dir} to DVC...")
        result = subprocess.run(
            ["dvc", "add", str(dataset_path)],
            cwd=dataset_path.parent,
            capture_output=True,
            text=True,
            check=True
        )

        # Calculate SHA256 hash
        sha256_hash = _calculate_directory_hash(dataset_path)

        # Commit .dvc file to git
        dvc_file = dataset_path.with_suffix(dataset_path.suffix + ".dvc")
        if dvc_file.exists():
            logger.info("Committing .dvc file to git...")
            subprocess.run(
                ["git", "add", str(dvc_file)],
                cwd=dataset_path.parent,
                check=False  # Don't fail if not a git repo
            )

        # Push to DVC remote if requested
        if push:
            logger.info(f"Pushing to DVC remote: {remote_name}...")
            push_result = subprocess.run(
                ["dvc", "push", "-r", remote_name],
                cwd=dataset_path.parent,
                capture_output=True,
                text=True,
                check=False  # Don't fail if remote not configured
            )

            if push_result.returncode != 0:
                logger.warning(f"DVC push failed: {push_result.stderr}")

        # Update data registry
        _update_data_registry(
            dataset_name=dataset_name,
            dataset_path=str(dataset_path),
            sha256=sha256_hash
        )

        logger.info(f"Successfully registered dataset: {dataset_name}")

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "sha256": sha256_hash,
            "dvc_file": str(dvc_file) if dvc_file.exists() else None,
            "pushed": push and push_result.returncode == 0
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"DVC command failed: {e.stderr}")
        raise DVCError(f"DVC registration failed: {e.stderr}") from e
    except Exception as e:
        logger.error(f"Dataset registration failed: {e}")
        raise DVCError(f"Registration failed: {str(e)}") from e


def pull_dataset_from_dvc(
    dataset_name: str,
    remote_name: str = "origin"
) -> Dict[str, Any]:
    """
    Pull dataset from DVC remote.

    Args:
        dataset_name: Dataset to pull
        remote_name: DVC remote name

    Returns:
        Dict with pull results
    """
    logger.info(f"Pulling dataset from DVC: {dataset_name}")

    try:
        # Get dataset path from registry
        registry = _load_data_registry()

        if dataset_name not in registry.get("datasets", {}):
            raise DVCError(f"Dataset not found in registry: {dataset_name}")

        dataset_info = registry["datasets"][dataset_name]
        dataset_path = Path(dataset_info["path"])

        # Pull from DVC
        result = subprocess.run(
            ["dvc", "pull", "-r", remote_name, str(dataset_path.with_suffix(".dvc"))],
            cwd=dataset_path.parent,
            capture_output=True,
            text=True,
            check=True
        )

        logger.info(f"Successfully pulled dataset: {dataset_name}")

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "sha256": dataset_info.get("sha256")
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"DVC pull failed: {e.stderr}")
        raise DVCError(f"Pull failed: {e.stderr}") from e
    except Exception as e:
        logger.error(f"Dataset pull failed: {e}")
        raise DVCError(f"Pull failed: {str(e)}") from e


def _ensure_dvc_initialized(directory: Path):
    """
    Ensure DVC is initialized in directory.

    Args:
        directory: Directory to initialize DVC in
    """
    dvc_dir = directory / ".dvc"

    if not dvc_dir.exists():
        logger.info(f"Initializing DVC in {directory}...")
        try:
            subprocess.run(
                ["dvc", "init"],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"DVC init failed (may already be initialized): {e.stderr}")


def _calculate_directory_hash(directory: Path) -> str:
    """
    Calculate SHA256 hash of all files in directory.

    Args:
        directory: Directory to hash

    Returns:
        SHA256 hash string
    """
    hasher = hashlib.sha256()

    # Get sorted list of all files
    files = sorted(directory.rglob('*'))

    for file_path in files:
        if file_path.is_file():
            # Include file path in hash
            hasher.update(str(file_path.relative_to(directory)).encode())

            # Include file content in hash
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)

    return hasher.hexdigest()


def _load_data_registry() -> Dict[str, Any]:
    """
    Load data registry from YAML file.

    Returns:
        Data registry dict
    """
    registry_path = Path(__file__).parent.parent / "data_registry.yaml"

    if registry_path.exists():
        with open(registry_path, 'r') as f:
            return yaml.safe_load(f) or {"datasets": {}}

    return {"datasets": {}}


def _update_data_registry(
    dataset_name: str,
    dataset_path: str,
    sha256: str
):
    """
    Update data registry with new dataset info.

    Args:
        dataset_name: Dataset identifier
        dataset_path: Path to dataset
        sha256: SHA256 hash
    """
    registry_path = Path(__file__).parent.parent / "data_registry.yaml"

    # Load existing registry
    registry = _load_data_registry()

    # Update dataset entry
    if "datasets" not in registry:
        registry["datasets"] = {}

    registry["datasets"][dataset_name] = {
        "path": dataset_path,
        "sha256": sha256,
        "registered_at": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat()
    }

    # Save registry
    with open(registry_path, 'w') as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated data registry: {registry_path}")


def get_dataset_info(dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Get dataset information from registry.

    Args:
        dataset_name: Dataset identifier

    Returns:
        Dataset info dict or None if not found
    """
    registry = _load_data_registry()

    return registry.get("datasets", {}).get(dataset_name)


def list_registered_datasets() -> List[Dict[str, Any]]:
    """
    List all registered datasets.

    Returns:
        List of dataset info dicts
    """
    registry = _load_data_registry()

    datasets = []
    for name, info in registry.get("datasets", {}).items():
        datasets.append({
            "name": name,
            **info
        })

    return datasets


def validate_dataset_integrity(dataset_name: str) -> Dict[str, Any]:
    """
    Validate dataset integrity by comparing current hash with registered hash.

    Args:
        dataset_name: Dataset to validate

    Returns:
        Dict with validation results
    """
    logger.info(f"Validating dataset integrity: {dataset_name}")

    dataset_info = get_dataset_info(dataset_name)

    if not dataset_info:
        return {
            "valid": False,
            "error": f"Dataset not found in registry: {dataset_name}"
        }

    dataset_path = Path(dataset_info["path"])

    if not dataset_path.exists():
        return {
            "valid": False,
            "error": f"Dataset directory does not exist: {dataset_path}"
        }

    # Calculate current hash
    current_hash = _calculate_directory_hash(dataset_path)
    registered_hash = dataset_info.get("sha256")

    if current_hash == registered_hash:
        return {
            "valid": True,
            "dataset_name": dataset_name,
            "sha256": current_hash
        }
    else:
        return {
            "valid": False,
            "error": "Hash mismatch - dataset may have been modified",
            "current_hash": current_hash,
            "registered_hash": registered_hash
        }
