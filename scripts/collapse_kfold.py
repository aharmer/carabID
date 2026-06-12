"""
Consolidate cross-validation folds into a single train/val dataset for YOLO classification.

This script collects all unique images from CV folds (avoiding duplicates) and redistributes
them into a final train/validation split for deployment training.
"""

import shutil
from pathlib import Path
from collections import defaultdict
import random

def consolidate_cv_folds(
    cv_root: str,
    output_root: str,
    num_folds: int = 5,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Consolidate CV folds into a single dataset for final deployment training.
    Collects all unique images (avoiding duplicates across folds) and redistributes
    them into train/val splits.
    
    Args:
        cv_root: Root directory containing fold subdirectories (fold1, fold2, etc.)
        output_root: Output directory for consolidated dataset
        num_folds: Number of folds to process
        train_split: Name of training split folder
        val_split: Name of validation split folder
        test_split: Name of test split folder
        val_ratio: Proportion of images to use for validation split
        seed: Random seed for reproducible train/val split
    
    Expected input structure:
        cv_root/
            fold1/
                train/
                    class1/
                        img1.jpg
                    class2/
                        img2.jpg
                valid/
                    class1/
                    class2/
                test/
                    class1/
                    class2/
            fold2/
                ...
    
    Output structure:
        output_root/
            train/
                class1/
                class2/
            valid/
                class1/
                class2/
    """
    cv_root = Path(cv_root)
    output_root = Path(output_root)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directories
    output_train = output_root / train_split
    output_val = output_root / val_split
    
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)
    
    # Collect all unique images by class
    # Key: class_name, Value: dict of {filename: source_path}
    unique_images = defaultdict(dict)
    
    print(f"Collecting unique images from {num_folds} folds...")
    print("-" * 60)
    
    # Process each fold and collect unique images
    for fold_idx in range(1, num_folds + 1):
        fold_dir = cv_root / f"fold_{fold_idx}"
        
        if not fold_dir.exists():
            print(f"Warning: {fold_dir} not found, skipping...")
            continue
        
        print(f"\nProcessing fold {fold_idx}...")
        fold_image_count = 0
        
        # Process all splits (train, valid, test)
        for split_name in [train_split, val_split, test_split]:
            split_dir = fold_dir / split_name
            
            if not split_dir.exists():
                continue
            
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    
                    for img_path in class_dir.iterdir():
                        if img_path.is_file():
                            # Use filename as unique identifier
                            filename = img_path.name
                            
                            # Only add if we haven't seen this image before
                            if filename not in unique_images[class_name]:
                                unique_images[class_name][filename] = img_path
                                fold_image_count += 1
        
        print(f"  Found {fold_image_count} new unique images")
    
    # Summary of collected images
    print("\n" + "=" * 60)
    print("UNIQUE IMAGES COLLECTED")
    print("=" * 60)
    total_unique = 0
    for class_name in sorted(unique_images.keys()):
        count = len(unique_images[class_name])
        total_unique += count
        print(f"  {class_name}: {count} unique images")
    print(f"  Total: {total_unique} unique images")
    
    # Redistribute into train/val splits
    print("\n" + "=" * 60)
    print(f"REDISTRIBUTING INTO TRAIN/VAL SPLITS ({val_ratio:.0%} validation)")
    print("=" * 60)
    
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    
    for class_name, images_dict in unique_images.items():
        # Create class directories
        (output_train / class_name).mkdir(exist_ok=True)
        (output_val / class_name).mkdir(exist_ok=True)
        
        # Get list of (filename, path) pairs and shuffle
        image_items = list(images_dict.items())
        random.shuffle(image_items)
        
        # Calculate split point
        num_val = int(len(image_items) * val_ratio)
        num_train = len(image_items) - num_val
        
        # Split into val and train
        val_items = image_items[:num_val]
        train_items = image_items[num_val:]
        
        # Copy to validation
        for filename, source_path in val_items:
            dest_path = output_val / class_name / filename
            shutil.copy2(source_path, dest_path)
            val_counts[class_name] += 1
        
        # Copy to train
        for filename, source_path in train_items:
            dest_path = output_train / class_name / filename
            shutil.copy2(source_path, dest_path)
            train_counts[class_name] += 1
        
        print(f"  {class_name}: {num_train} train, {num_val} val")
    
    # Final summary
    print("\n" + "=" * 60)
    print("CONSOLIDATION COMPLETE")
    print("=" * 60)
    
    print(f"\nOutput directory: {output_root}")
    print(f"\nTrain set:")
    total_train = 0
    for class_name in sorted(train_counts.keys()):
        count = train_counts[class_name]
        total_train += count
        print(f"  {class_name}: {count} images")
    print(f"  Total: {total_train} images")
    
    print(f"\nValidation set:")
    total_val = 0
    for class_name in sorted(val_counts.keys()):
        count = val_counts[class_name]
        total_val += count
        print(f"  {class_name}: {count} images")
    print(f"  Total: {total_val} images")
    
    print(f"\nGrand total: {total_train + total_val} images")
    print(f"Train/Val split: {total_train/(total_train+total_val):.1%} / {total_val/(total_train+total_val):.1%}")


if __name__ == "__main__":
    # Example usage
    CV_ROOT = "D:/Dropbox/data/carabID/imgs/cv_classification_set"  # Directory containing fold1, fold2, etc.
    OUTPUT_ROOT = "D:/Dropbox/data/carabID/imgs/final_classification_set"
    NUM_FOLDS = 5
    
    consolidate_cv_folds(
        cv_root=CV_ROOT,
        output_root=OUTPUT_ROOT,
        num_folds=NUM_FOLDS,
        train_split="train",
        val_split="val",
        test_split="test",
        val_ratio=0.15,  # 20% validation, 80% train
        seed=42  # For reproducible splits
    )