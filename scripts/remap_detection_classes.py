"""
YOLO11 Class Remapping Script
Remaps all classes in a YOLO dataset to a single specified class
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Optional


def find_dataset_yaml(dataset_path: Path) -> Optional[Path]:
    """Find the dataset.yaml file in the dataset directory."""
    possible_names = ['dataset.yaml', 'data.yaml', 'config.yaml']
    for name in possible_names:
        yaml_path = dataset_path / name
        if yaml_path.exists():
            return yaml_path
    return None


def backup_dataset(dataset_path: Path, backup_suffix: str = "_backup") -> Path:
    """Create a backup of the original dataset."""
    backup_path = Path(str(dataset_path) + backup_suffix)
    if backup_path.exists():
        print(f"Backup already exists at {backup_path}")
        return backup_path
    
    print(f"Creating backup at {backup_path}...")
    shutil.copytree(dataset_path, backup_path)
    return backup_path


def remap_label_file(label_path: Path) -> None:
    """Remap all class IDs in a label file to class 0."""
    if not label_path.exists():
        return
    
    lines = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split the line and change the class ID to 0
                parts = line.split()
                if len(parts) >= 5:  # Valid YOLO format: class x y w h
                    parts[0] = '0'  # Set class to 0
                    lines.append(' '.join(parts))
    
    # Write back the modified lines
    with open(label_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def update_dataset_yaml(yaml_path: Path, class_name: str) -> None:
    """Update the dataset.yaml file to reflect the new single class."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update the classes
    data['nc'] = 1  # Number of classes
    data['names'] = [class_name]  # Class names
    
    # Write back the updated YAML
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def remap_yolo_dataset(dataset_path: str, class_name: str, create_backup: bool = True) -> None:
    """
    Remap all classes in a YOLO dataset to a single specified class.
    
    Args:
        dataset_path: Path to the dataset directory
        class_name: Name of the new single class
        create_backup: Whether to create a backup before modifying
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    print(f"Processing dataset at: {dataset_path}")
    print(f"Remapping all classes to: {class_name}")
    
    # Create backup if requested
    if create_backup:
        backup_path = backup_dataset(dataset_path)
        print(f"Backup created at: {backup_path}")
    
    # Find and update dataset.yaml
    yaml_path = find_dataset_yaml(dataset_path)
    if yaml_path:
        print(f"Updating dataset configuration: {yaml_path}")
        update_dataset_yaml(yaml_path, class_name)
    else:
        print("Warning: No dataset.yaml file found. You may need to create one manually.")
    
    # Process label files
    labels_processed = 0
    
    # Look for labels in common YOLO directory structures
    possible_label_dirs = [
        dataset_path / "train" / "labels",
        dataset_path / "labels" / "train",
        dataset_path / "labels",
    ]
    
    label_dirs_found = []
    for label_dir in possible_label_dirs:
        if label_dir.exists():
            label_dirs_found.append(label_dir)
    
    if not label_dirs_found:
        print("Warning: No label directories found. Looking for .txt files recursively...")
        # Fallback: search for all .txt files in the dataset
        for txt_file in dataset_path.rglob("*.txt"):
            # Skip dataset.yaml and similar files
            if txt_file.suffix == '.txt' and 'yaml' not in txt_file.name.lower():
                remap_label_file(txt_file)
                labels_processed += 1
    else:
        # Process found label directories
        for label_dir in label_dirs_found:
            print(f"Processing labels in: {label_dir}")
            for label_file in label_dir.glob("*.txt"):
                remap_label_file(label_file)
                labels_processed += 1
    
    print(f"\nProcessing complete!")
    print(f"- Labels processed: {labels_processed}")
    print(f"- All classes remapped to: {class_name} (class ID: 0)")
    
    if yaml_path:
        print(f"- Dataset configuration updated: {yaml_path}")
    else:
        print(f"\nNote: You may need to create a dataset.yaml file with:")
        print("nc: 1")
        print(f"names: ['{class_name}']")
        print("train: path/to/train/images")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Remap all classes in a YOLO11 dataset to a single specified class"
    )
    parser.add_argument(
        "dataset_path",
        help="Path to the YOLO dataset directory"
    )
    parser.add_argument(
        "class_name",
        help="Name of the new single class (e.g., 'diptera', 'insect', 'object')"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup of the original dataset"
    )
    
    args = parser.parse_args()
    
    try:
        remap_yolo_dataset(args.dataset_path, args.class_name, create_backup=not args.no_backup)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    
    exit(main())

# Usage
# python ./scripts/remap_detection_classes.py D:/Dropbox/data/dipteraID/imgs/hymenoptera-g-13 hymenoptera --no-backup
