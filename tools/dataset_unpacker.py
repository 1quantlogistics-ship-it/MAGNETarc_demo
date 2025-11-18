"""
Dataset Unpacker

Extracts and validates dataset archives (zip, tar.gz) for ARC training.

Features:
- Automatic format detection
- Safe extraction with validation
- Integrity checks
- Integration with tool governance
- Audit logging

Supports:
- ZIP archives
- TAR/TAR.GZ archives
- Nested folder structures
- Multiple image formats (JPG, PNG)
"""

import os
import zipfile
import tarfile
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetUnpackerError(Exception):
    """Raised when dataset unpacking fails."""
    pass


def unpack_zip_to_dataset(
    zip_path: str,
    output_dir: str,
    validate: bool = True,
    max_size_mb: int = 10240  # 10GB default limit
) -> Dict[str, Any]:
    """
    Unpack ZIP archive to dataset directory.

    Args:
        zip_path: Path to ZIP file
        output_dir: Destination directory
        validate: Whether to validate archive before extraction
        max_size_mb: Maximum archive size in MB

    Returns:
        Dict with extraction results

    Raises:
        DatasetUnpackerError: If extraction fails
    """
    logger.info(f"Unpacking dataset: {zip_path} → {output_dir}")

    zip_file = Path(zip_path)
    output_path = Path(output_dir)

    # Validate ZIP file exists
    if not zip_file.exists():
        raise DatasetUnpackerError(f"ZIP file not found: {zip_path}")

    # Check file size
    file_size_mb = zip_file.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise DatasetUnpackerError(
            f"ZIP file too large: {file_size_mb:.1f}MB > {max_size_mb}MB"
        )

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Validate ZIP integrity
            if validate:
                logger.info("Validating ZIP integrity...")
                bad_file = zip_ref.testzip()
                if bad_file:
                    raise DatasetUnpackerError(f"Corrupt file in ZIP: {bad_file}")

            # Get file list
            file_list = zip_ref.namelist()
            logger.info(f"ZIP contains {len(file_list)} files")

            # Extract all files
            logger.info("Extracting files...")
            zip_ref.extractall(output_dir)

        # Analyze extracted content
        result = _analyze_extracted_content(output_path)

        logger.info(f"Successfully extracted dataset to {output_dir}")

        return {
            "status": "success",
            "zip_path": str(zip_path),
            "output_dir": str(output_dir),
            "files_extracted": len(file_list),
            "size_mb": file_size_mb,
            **result
        }

    except zipfile.BadZipFile as e:
        raise DatasetUnpackerError(f"Invalid ZIP file: {str(e)}") from e
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise DatasetUnpackerError(f"Extraction failed: {str(e)}") from e


def unpack_tar_to_dataset(
    tar_path: str,
    output_dir: str,
    validate: bool = True,
    max_size_mb: int = 10240
) -> Dict[str, Any]:
    """
    Unpack TAR/TAR.GZ archive to dataset directory.

    Args:
        tar_path: Path to TAR file
        output_dir: Destination directory
        validate: Whether to validate archive
        max_size_mb: Maximum archive size in MB

    Returns:
        Dict with extraction results

    Raises:
        DatasetUnpackerError: If extraction fails
    """
    logger.info(f"Unpacking TAR dataset: {tar_path} → {output_dir}")

    tar_file = Path(tar_path)
    output_path = Path(output_dir)

    if not tar_file.exists():
        raise DatasetUnpackerError(f"TAR file not found: {tar_path}")

    # Check file size
    file_size_mb = tar_file.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise DatasetUnpackerError(
            f"TAR file too large: {file_size_mb:.1f}MB > {max_size_mb}MB"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_path, 'r:*') as tar_ref:
            # Get member list
            members = tar_ref.getmembers()
            logger.info(f"TAR contains {len(members)} files")

            # Extract all files
            logger.info("Extracting files...")
            tar_ref.extractall(output_dir)

        # Analyze extracted content
        result = _analyze_extracted_content(output_path)

        logger.info(f"Successfully extracted TAR dataset to {output_dir}")

        return {
            "status": "success",
            "tar_path": str(tar_path),
            "output_dir": str(output_dir),
            "files_extracted": len(members),
            "size_mb": file_size_mb,
            **result
        }

    except tarfile.TarError as e:
        raise DatasetUnpackerError(f"Invalid TAR file: {str(e)}") from e
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise DatasetUnpackerError(f"Extraction failed: {str(e)}") from e


def unpack_dataset(
    archive_path: str,
    output_dir: str,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Automatically detect archive type and unpack.

    Args:
        archive_path: Path to archive file
        output_dir: Destination directory
        validate: Whether to validate archive

    Returns:
        Dict with extraction results
    """
    archive_file = Path(archive_path)

    # Detect archive type by extension
    if archive_path.endswith('.zip'):
        return unpack_zip_to_dataset(archive_path, output_dir, validate)
    elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
        return unpack_tar_to_dataset(archive_path, output_dir, validate)
    else:
        raise DatasetUnpackerError(
            f"Unsupported archive format: {archive_file.suffix}. "
            f"Supported: .zip, .tar, .tar.gz, .tgz"
        )


def _analyze_extracted_content(output_dir: Path) -> Dict[str, Any]:
    """
    Analyze extracted dataset content.

    Args:
        output_dir: Directory to analyze

    Returns:
        Dict with analysis results
    """
    # Count files by type
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    mask_extensions = {'.png', '.bmp'}
    metadata_extensions = {'.json', '.csv', '.xml', '.mat'}

    images = []
    masks = []
    metadata_files = []
    other_files = []

    for file_path in output_dir.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()

            if ext in image_extensions:
                # Heuristic: if 'mask' in path, it's probably a mask
                if 'mask' in str(file_path).lower():
                    masks.append(str(file_path.relative_to(output_dir)))
                else:
                    images.append(str(file_path.relative_to(output_dir)))

            elif ext in metadata_extensions:
                metadata_files.append(str(file_path.relative_to(output_dir)))

            else:
                other_files.append(str(file_path.relative_to(output_dir)))

    # Detect folder structure
    has_images_folder = (output_dir / 'images').exists()
    has_masks_folder = (output_dir / 'masks').exists()

    return {
        "image_count": len(images),
        "mask_count": len(masks),
        "metadata_count": len(metadata_files),
        "other_count": len(other_files),
        "has_images_folder": has_images_folder,
        "has_masks_folder": has_masks_folder,
        "sample_images": images[:5],
        "sample_masks": masks[:5],
        "metadata_files": metadata_files
    }


def create_metadata_json(
    dataset_dir: str,
    dataset_name: str,
    description: str = "",
    source: str = ""
) -> str:
    """
    Create metadata.json for dataset.

    Args:
        dataset_dir: Dataset directory
        dataset_name: Dataset name
        description: Dataset description
        source: Dataset source/origin

    Returns:
        Path to created metadata.json
    """
    dataset_path = Path(dataset_dir)

    # Analyze dataset content
    analysis = _analyze_extracted_content(dataset_path)

    metadata = {
        "name": dataset_name,
        "description": description,
        "source": source,
        "created_at": datetime.utcnow().isoformat(),
        "statistics": {
            "total_images": analysis["image_count"],
            "total_masks": analysis["mask_count"],
            "has_segmentation": analysis["mask_count"] > 0
        },
        "structure": {
            "has_images_folder": analysis["has_images_folder"],
            "has_masks_folder": analysis["has_masks_folder"]
        }
    }

    metadata_path = dataset_path / "metadata.json"

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created metadata: {metadata_path}")

    return str(metadata_path)


def validate_dataset_structure(dataset_dir: str) -> Dict[str, Any]:
    """
    Validate dataset has required structure.

    Expected structure:
        dataset_dir/
            images/
                *.jpg or *.png
            masks/ (optional, for segmentation)
                *.png
            metadata.json

    Args:
        dataset_dir: Directory to validate

    Returns:
        Dict with validation results
    """
    dataset_path = Path(dataset_dir)

    errors = []
    warnings = []

    # Check directory exists
    if not dataset_path.exists():
        errors.append(f"Dataset directory does not exist: {dataset_dir}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check images/ folder
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        errors.append("Missing images/ directory")
    else:
        # Count images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if len(image_files) == 0:
            errors.append("No images found in images/ directory")
        logger.info(f"Found {len(image_files)} images")

    # Check masks/ folder (optional for segmentation tasks)
    masks_dir = dataset_path / "masks"
    if masks_dir.exists():
        mask_files = list(masks_dir.glob("*.png"))
        logger.info(f"Found {len(mask_files)} masks")

        # Check if image and mask counts match
        if len(mask_files) != len(image_files):
            warnings.append(
                f"Image count ({len(image_files)}) != mask count ({len(mask_files)})"
            )

    # Check metadata.json
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        warnings.append("Missing metadata.json (recommended)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "image_count": len(image_files) if 'image_files' in locals() else 0,
        "mask_count": len(mask_files) if 'mask_files' in locals() else 0
    }
