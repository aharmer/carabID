import os
from pathlib import Path
from collections import defaultdict
import pandas as pd


def count_images_per_class(labels_dir, class_names=None, splits=None):
    """
    Count the number of images per class in a YOLO detection dataset.
    
    Args:
        labels_dir: Path to the directory containing YOLO format label files (.txt)
                   or root directory containing train/val/test splits
        class_names: Optional list of class names. If None, will use class indices.
        splits: List of split names to process (e.g., ['train', 'val', 'test']).
               If None, will search for labels directly in labels_dir.
    
    Returns:
        pandas DataFrame with class counts and percentages
    """
    labels_path = Path(labels_dir)
    
    if not labels_path.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Dictionary to track which images contain each class
    class_to_images = defaultdict(set)
    total_images = 0
    
    # Determine which directories to search
    label_files = []
    
    if splits:
        # Search in train/val/test subdirectories
        for split in splits:
            split_labels_dir = labels_path / split / 'labels'
            if split_labels_dir.exists():
                label_files.extend(split_labels_dir.glob("*.txt"))
                print(f"Found {len(list(split_labels_dir.glob('*.txt')))} label files in {split}")
            else:
                # Try without 'labels' subdirectory
                split_dir = labels_path / split
                if split_dir.exists():
                    label_files.extend(split_dir.glob("*.txt"))
                    print(f"Found {len(list(split_dir.glob('*.txt')))} label files in {split}")
    else:
        # Search directly in the provided directory
        label_files = list(labels_path.glob("*.txt"))
    
    if not label_files:
        print(f"Warning: No label files found in {labels_dir}")
        if not splits:
            print("Hint: If your dataset has train/val/test splits, use --splits parameter")
    
    # Process each label file
    for label_file in label_files:
        total_images += 1
        image_name = label_file.stem
        
        # Read the label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Extract unique class IDs from this image
        classes_in_image = set()
        for line in lines:
            parts = line.strip().split()
            if parts:  # Skip empty lines
                class_id = int(parts[0])
                classes_in_image.add(class_id)
        
        # Add this image to each class it contains
        for class_id in classes_in_image:
            class_to_images[class_id].add(image_name)
    
    # Create results dictionary
    results = []
    for class_id in sorted(class_to_images.keys()):
        count = len(class_to_images[class_id])
        percentage = (count / total_images * 100) if total_images > 0 else 0
        
        # Use class name if provided, otherwise use class ID
        class_label = class_names[class_id] if (class_names and class_id < len(class_names)) else f"Class {class_id}"
        
        results.append({
            'Class ID': class_id,
            'Class Name': class_label,
            'Image Count': count,
            'Percentage': f"{percentage:.2f}%"
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add summary row
    summary = pd.DataFrame([{
        'Class ID': '',
        'Class Name': 'TOTAL',
        'Image Count': total_images,
        'Percentage': '100.00%'
    }])
    df = pd.concat([df, summary], ignore_index=True)
    
    return df


def load_class_names_from_yaml(yaml_path):
    """
    Load class names from a YOLO dataset YAML file.
    
    Args:
        yaml_path: Path to the dataset.yaml file
    
    Returns:
        List of class names
    """
    import yaml
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data.get('names', [])


def save_class_distribution(labels_dir, output_csv, class_names=None, yaml_path=None, splits=None):
    """
    Count classes and save distribution to CSV.
    
    Args:
        labels_dir: Path to the directory containing YOLO label files or root directory
        output_csv: Path where CSV file will be saved
        class_names: Optional list of class names
        yaml_path: Optional path to dataset YAML file (overrides class_names if provided)
        splits: List of split names to process (e.g., ['train', 'val', 'test'])
    
    Returns:
        pandas DataFrame with the class distribution
    """
    # Load class names from YAML if provided
    if yaml_path and os.path.exists(yaml_path):
        class_names = load_class_names_from_yaml(yaml_path)
        print(f"Loaded {len(class_names)} class names from {yaml_path}")
    
    # Count classes
    df = count_images_per_class(labels_dir, class_names, splits)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Class distribution saved to {output_csv}")
    
    # Print summary
    total_images = df.iloc[-1]['Image Count']
    num_classes = len(df) - 1
    print(f"\nSummary:")
    print(f"  Total images: {total_images}")
    print(f"  Number of classes: {num_classes}")
    if num_classes > 0:
        print(f"  Average images per class: {total_images/num_classes:.1f}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Count images per class in YOLO dataset')
    parser.add_argument('labels_dir', help='Path to labels directory or dataset root')
    parser.add_argument('output_csv', help='Path for output CSV file')
    parser.add_argument('--yaml', help='Path to dataset YAML file (optional)')
    parser.add_argument('--classes', help='Comma-separated class names (optional)', default=None)
    parser.add_argument('--splits', help='Comma-separated split names (e.g., train,val,test)', default=None)
    
    args = parser.parse_args()
    
    # Parse class names if provided
    class_names = None
    if args.classes:
        class_names = [name.strip() for name in args.classes.split(',')]
    
    # Parse splits if provided
    splits = None
    if args.splits:
        splits = [s.strip() for s in args.splits.split(',')]
    
    # Run the analysis
    save_class_distribution(
        labels_dir=args.labels_dir,
        output_csv=args.output_csv,
        class_names=class_names,
        yaml_path=args.yaml,
        splits=splits
    )

    