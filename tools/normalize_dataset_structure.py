"""
Dataset Structure Normalizer

Normalizes arbitrary dataset structures to ARC's standard format.

Standard Format:
    dataset_root/
        images/
            *.jpg or *.png
        masks/ (if segmentation)
            *.png
        metadata.json

Handles:
- Nested folder structures
- Separate train/val/test splits
- MATLAB .mat files (common in RIM-ONE)
- CSV metadata
- Various naming conventions
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


class DatasetNormalizerError(Exception):
    """Raised when dataset normalization fails."""
    pass


def normalize_dataset_structure(
    input_dir: str,
    output_dir: str,
    dataset_name: str,
    mode: str = "copy"  # or "move"
) -> Dict[str, Any]:
    """
    Normalize dataset structure to ARC standard format.

    Args:
        input_dir: Input dataset directory (potentially messy)
        output_dir: Output directory for normalized dataset
        dataset_name: Dataset name for metadata
        mode: "copy" or "move" files

    Returns:
        Dict with normalization results

    Raises:
        DatasetNormalizerError: If normalization fails
    """
    logger.info(f"Normalizing dataset structure: {input_dir} → {output_dir}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise DatasetNormalizerError(f"Input directory does not exist: {input_dir}")

    # Create output structure
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Find all images and masks
    images_found, masks_found = _find_images_and_masks(input_path)

    logger.info(f"Found {len(images_found)} images and {len(masks_found)} masks")

    # Copy/move images
    images_copied = 0
    for src_path in images_found:
        dest_path = images_dir / src_path.name
        if mode == "copy":
            shutil.copy2(src_path, dest_path)
        else:
            shutil.move(str(src_path), dest_path)
        images_copied += 1

    # Copy/move masks
    masks_copied = 0
    for src_path in masks_found:
        dest_path = masks_dir / src_path.name
        if mode == "copy":
            shutil.copy2(src_path, dest_path)
        else:
            shutil.move(str(src_path), dest_path)
        masks_copied += 1

    # Process MATLAB files if present
    mat_files = list(input_path.rglob("*.mat"))
    mat_processed = 0
    if mat_files and HAS_SCIPY:
        for mat_file in mat_files:
            try:
                _process_matlab_file(mat_file, output_path)
                mat_processed += 1
            except Exception as e:
                logger.warning(f"Failed to process MATLAB file {mat_file}: {e}")

    # Create metadata.json
    metadata = {
        "name": dataset_name,
        "description": f"Normalized dataset from {input_dir}",
        "source": str(input_path),
        "normalized_at": datetime.utcnow().isoformat(),
        "statistics": {
            "total_images": images_copied,
            "total_masks": masks_copied,
            "has_segmentation": masks_copied > 0,
            "matlab_files_processed": mat_processed
        },
        "structure": {
            "format": "arc_standard",
            "images_dir": "images/",
            "masks_dir": "masks/" if masks_copied > 0 else None
        }
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Normalization complete: {images_copied} images, {masks_copied} masks")

    return {
        "status": "success",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "images_copied": images_copied,
        "masks_copied": masks_copied,
        "matlab_files_processed": mat_processed,
        "metadata_path": str(metadata_path)
    }


def _find_images_and_masks(root_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find all images and masks in directory tree.

    Args:
        root_dir: Root directory to search

    Returns:
        Tuple of (image_paths, mask_paths)
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    images = []
    masks = []

    for file_path in root_dir.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()

            if ext in image_extensions:
                # Heuristic detection of masks
                path_lower = str(file_path).lower()

                if any(keyword in path_lower for keyword in ['mask', 'segmentation', 'label', 'ground_truth', 'gt']):
                    masks.append(file_path)
                else:
                    images.append(file_path)

    return images, masks


def _process_matlab_file(mat_file: Path, output_dir: Path):
    """
    Process MATLAB .mat file (common in RIM-ONE dataset).

    Args:
        mat_file: Path to .mat file
        output_dir: Output directory for extracted data
    """
    if not HAS_SCIPY:
        logger.warning("scipy not available, skipping MATLAB file processing")
        return

    try:
        mat_data = sio.loadmat(str(mat_file))

        # Extract relevant data
        # (Structure depends on specific dataset format)
        logger.info(f"Loaded MATLAB file: {mat_file.name}")
        logger.debug(f"Keys: {list(mat_data.keys())}")

        # Save metadata about MATLAB file
        mat_metadata = {
            "filename": mat_file.name,
            "keys": [k for k in mat_data.keys() if not k.startswith('__')],
            "processed_at": datetime.utcnow().isoformat()
        }

        metadata_path = output_dir / f"matlab_{mat_file.stem}.json"
        with open(metadata_path, 'w') as f:
            json.dump(mat_metadata, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to process MATLAB file: {e}")
        raise


def merge_dataset_splits(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    output_dir: str,
    dataset_name: str
) -> Dict[str, Any]:
    """
    Merge separate train/val/test splits into single normalized dataset.

    Args:
        train_dir: Training set directory
        val_dir: Validation set directory
        test_dir: Test set directory
        output_dir: Output directory
        dataset_name: Dataset name

    Returns:
        Dict with merge results
    """
    logger.info(f"Merging dataset splits into {output_dir}")

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_dir,
        "val": val_dir,
        "test": test_dir
    }

    total_images = 0
    total_masks = 0
    split_info = {}

    for split_name, split_dir in splits.items():
        if not split_dir or not Path(split_dir).exists():
            continue

        images, masks = _find_images_and_masks(Path(split_dir))

        # Copy images with split prefix
        for img_path in images:
            dest_name = f"{split_name}_{img_path.name}"
            shutil.copy2(img_path, images_dir / dest_name)
            total_images += 1

        # Copy masks with split prefix
        for mask_path in masks:
            dest_name = f"{split_name}_{mask_path.name}"
            shutil.copy2(mask_path, masks_dir / dest_name)
            total_masks += 1

        split_info[split_name] = {
            "images": len(images),
            "masks": len(masks)
        }

    # Create metadata
    metadata = {
        "name": dataset_name,
        "description": f"Merged dataset from train/val/test splits",
        "merged_at": datetime.utcnow().isoformat(),
        "splits": split_info,
        "statistics": {
            "total_images": total_images,
            "total_masks": total_masks,
            "has_segmentation": total_masks > 0
        }
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Merge complete: {total_images} images, {total_masks} masks")

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "total_images": total_images,
        "total_masks": total_masks,
        "split_info": split_info
    }


def detect_dataset_format(dataset_dir: str) -> Dict[str, Any]:
    """
    Auto-detect dataset format and structure.

    Args:
        dataset_dir: Directory to analyze

    Returns:
        Dict with detected format information
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        return {"detected": False, "error": "Directory does not exist"}

    # Check for standard ARC format
    has_images_dir = (dataset_path / "images").exists()
    has_masks_dir = (dataset_path / "masks").exists()
    has_metadata = (dataset_path / "metadata.json").exists()

    if has_images_dir and has_metadata:
        return {
            "detected": True,
            "format": "arc_standard",
            "ready": True,
            "needs_normalization": False
        }

    # Check for split-based format (train/val/test)
    has_train = (dataset_path / "train").exists()
    has_val = (dataset_path / "val").exists()
    has_test = (dataset_path / "test").exists()

    if has_train or has_val or has_test:
        return {
            "detected": True,
            "format": "split_based",
            "ready": False,
            "needs_normalization": True,
            "splits_found": {
                "train": has_train,
                "val": has_val,
                "test": has_test
            }
        }

    # Check for flat structure (all images in root)
    images, masks = _find_images_and_masks(dataset_path)

    if len(images) > 0:
        return {
            "detected": True,
            "format": "flat",
            "ready": False,
            "needs_normalization": True,
            "image_count": len(images),
            "mask_count": len(masks)
        }

    return {
        "detected": False,
        "format": "unknown",
        "ready": False,
        "needs_normalization": True
    }
