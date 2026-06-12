import os
import shutil
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import argparse

def get_class_images(dataset_dir: str) -> Dict[str, List[str]]:
    """
    Get images organized by class from a classification dataset directory structure.
    
    Args:
        dataset_dir: Path to dataset directory containing class subdirectories
    
    Returns:
        Dictionary mapping class_name to list of image paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    class_to_images = {}
    
    dataset_path = Path(dataset_dir)
    
    # Find all subdirectories (classes)
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"Warning: No class directories found in {dataset_dir}")
        return class_to_images
    
    print(f"Found {len(class_dirs)} class directories")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = []
        
        # Get all image files in this class directory
        for ext in image_extensions:
            image_files.extend(class_dir.glob(f'*{ext}'))
            image_files.extend(class_dir.glob(f'*{ext.upper()}'))
        
        # Remove duplicates and convert to strings
        unique_images = list(set(str(img) for img in image_files))
        
        if unique_images:
            class_to_images[class_name] = unique_images
            print(f"  Class '{class_name}': {len(unique_images)} images")
        else:
            print(f"  Warning: No images found in class directory '{class_name}'")
    
    return class_to_images

def analyze_class_distribution(class_to_images: Dict[str, List[str]]) -> Dict:
    """
    Analyze class distribution in the classification dataset.
    
    Args:
        class_to_images: Dictionary mapping class_name to list of image paths
    
    Returns:
        Dictionary with class distribution statistics
    """
    print("Analyzing class distribution...")
    
    # Count images per class
    class_counts = {class_name: len(images) for class_name, images in class_to_images.items()}
    total_images = sum(class_counts.values())
    unique_classes = set(class_counts.keys())
    
    print(f"Found {len(unique_classes)} unique classes with {total_images} total images")
    print("Class distribution:")
    
    # Sort classes by count for better readability
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, count in sorted_classes:
        percentage = (count / total_images) * 100
        print(f"  Class '{class_name}': {count} images ({percentage:.1f}%)")
    
    return {
        'class_counts': class_counts,
        'class_to_images': class_to_images,
        'unique_classes': unique_classes,
        'total_images': total_images
    }

def stratified_kfold_split(class_to_images: Dict[str, List[str]], 
                          n_splits: int = 3,
                          test_ratio: float = 0.0) -> List[Tuple[Dict, Dict, Dict]]:
    """
    Perform stratified k-fold cross-validation split for classification dataset.
    
    Args:
        class_to_images: Dictionary mapping class_name to list of image paths
        n_splits: Number of folds for cross-validation
        test_ratio: Proportion for held-out test set (applied before k-fold split)
    
    Returns:
        List of (train_dict, valid_dict, test_dict) tuples for each fold
    """
    print(f"\nPerforming stratified {n_splits}-fold cross-validation split...")
    
    all_folds = []
    held_out_test = defaultdict(list)
    
    # First, hold out test set if test_ratio > 0
    fold_class_images = {}
    for class_name, images in class_to_images.items():
        class_images = images.copy()
        random.shuffle(class_images)
        
        if test_ratio > 0:
            class_size = len(class_images)
            test_size = max(0, int(class_size * test_ratio))
            
            held_out_test[class_name] = class_images[:test_size]
            fold_class_images[class_name] = class_images[test_size:]
            
            print(f"  Class '{class_name}': {len(held_out_test[class_name])} images held out for test set")
        else:
            fold_class_images[class_name] = class_images
    
    # Create k-fold splits from remaining data
    for fold_idx in range(n_splits):
        print(f"\n--- Creating Fold {fold_idx + 1}/{n_splits} ---")
        
        train_dict = defaultdict(list)
        valid_dict = defaultdict(list)
        test_dict = defaultdict(list) if test_ratio > 0 else None
        
        for class_name, images in fold_class_images.items():
            class_size = len(images)
            fold_size = class_size // n_splits
            
            # Calculate validation indices for this fold
            valid_start = fold_idx * fold_size
            valid_end = valid_start + fold_size if fold_idx < n_splits - 1 else class_size
            
            # Split into train and validation
            valid_items = images[valid_start:valid_end]
            train_items = images[:valid_start] + images[valid_end:]
            
            train_dict[class_name] = train_items
            valid_dict[class_name] = valid_items
            
            # Add held-out test set
            if test_ratio > 0:
                test_dict[class_name] = held_out_test[class_name]
            
            print(f"  Class '{class_name}': train={len(train_items)}, val={len(valid_items)}" + 
                  (f", test={len(test_dict[class_name])}" if test_ratio > 0 else ""))
        
        all_folds.append((dict(train_dict), dict(valid_dict), dict(test_dict) if test_ratio > 0 else {}))
    
    return all_folds

def create_directory_structure(output_dir: str, fold_idx: int, class_names: List[str], include_test: bool = True):
    """
    Create the train/valid/test directory structure for a specific fold.
    
    Args:
        output_dir: Base output directory
        fold_idx: Fold index
        class_names: List of class names to create subdirectories for
        include_test: Whether to create test directory
    """
    fold_dir = Path(output_dir) / f"fold_{fold_idx}"
    splits = ['train', 'val'] + (['test'] if include_test else [])
    
    for split in splits:
        for class_name in class_names:
            dir_path = fold_dir / split / class_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created fold_{fold_idx} directory structure with {len(class_names)} classes")

def copy_files(split_dict: Dict[str, List[str]], output_dir: str, fold_idx: int, split_name: str):
    """
    Copy image files to the appropriate split directory for a specific fold.
    
    Args:
        split_dict: Dictionary mapping class_name to list of image paths
        output_dir: Base output directory
        fold_idx: Fold index
        split_name: Name of the split (train, val, test)
    """
    fold_dir = Path(output_dir) / f"fold_{fold_idx}"
    split_dest = fold_dir / split_name
    
    successful_copies = 0
    failed_copies = []
    total_expected = sum(len(images) for images in split_dict.values())
    
    for class_name, images in split_dict.items():
        class_dest = split_dest / class_name
        
        for image_path in images:
            try:
                image_name = Path(image_path).name
                image_dest_path = class_dest / image_name
                
                # Check for filename conflicts
                if image_dest_path.exists():
                    print(f"Warning: Image file {image_name} already exists in fold_{fold_idx}/{split_name}/{class_name}, skipping...")
                    failed_copies.append(f"Conflict: {class_name}/{image_name}")
                    continue
                
                shutil.copy2(image_path, image_dest_path)
                successful_copies += 1
                
            except Exception as e:
                print(f"Error copying {image_path}: {e}")
                failed_copies.append(f"Copy error: {class_name}/{Path(image_path).name} - {str(e)}")
    
    # Verify actual file counts
    actual_files = 0
    class_counts = {}
    for class_name in split_dict.keys():
        class_path = split_dest / class_name
        if class_path.exists():
            class_file_count = len(list(class_path.glob('*')))
            class_counts[class_name] = class_file_count
            actual_files += class_file_count
        else:
            class_counts[class_name] = 0
    
    print(f"Fold {fold_idx} - {split_name}:")
    print(f"  Expected: {total_expected} files")
    print(f"  Successful copies: {successful_copies}")
    print(f"  Actual files: {actual_files}")
    
    if failed_copies:
        print(f"  Failed copies ({len(failed_copies)}):")
        for failure in failed_copies[:3]:
            print(f"    - {failure}")
        if len(failed_copies) > 3:
            print(f"    ... and {len(failed_copies) - 3} more")
    
    return successful_copies, actual_files, class_counts

def diagnose_dataset(dataset_dir: str, n_splits: int):
    """
    Diagnose potential issues in the classification dataset before splitting.
    """
    print("="*50)
    print("DATASET DIAGNOSIS")
    print("="*50)
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist")
        return False
    
    # Find class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found {len(class_dirs)} class directories")
    
    if not class_dirs:
        print("Error: No class directories found")
        return False
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    issues_found = False
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(class_dir.glob(f'*{ext}'))
            image_files.extend(class_dir.glob(f'*{ext.upper()}'))
        
        print(f"Class '{class_name}': {len(image_files)} images")
        
        # Check if class has enough images for k-fold
        if len(image_files) < n_splits:
            print(f"  ⚠️  WARNING: Class '{class_name}' has only {len(image_files)} images")
            print(f"    This is less than n_splits={n_splits}, which may cause empty validation sets!")
            issues_found = True
        elif len(image_files) < n_splits * 2:
            print(f"  ⚠️  Warning: Class '{class_name}' has very few images ({len(image_files)})")
            print(f"    With {n_splits} folds, some folds may have very small validation sets")
            issues_found = True
    
    print("="*50)
    
    if issues_found:
        print("⚠️  Issues found that may affect k-fold splitting quality")
    else:
        print("✓ Dataset looks suitable for k-fold cross-validation")
    
    return not issues_found

def main():
    parser = argparse.ArgumentParser(description='Split classification dataset into k-fold cross-validation splits')
    parser.add_argument('--dataset_dir', required=True, help='Path to classification dataset directory containing class subdirectories')
    parser.add_argument('--output_dir', required=True, help='Output directory for split dataset')
    parser.add_argument('--n_splits', type=int, default=3, help='Number of folds for cross-validation (default: 3)')
    parser.add_argument('--test_ratio', type=float, default=0.0, help='Proportion for held-out test set (default: 0.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits (default: 42)')
    parser.add_argument('--diagnose', action='store_true', help='Run dataset diagnosis before splitting')
    
    args = parser.parse_args()
    
    # Validate n_splits
    if args.n_splits < 2:
        print(f"Error: n_splits must be at least 2, got {args.n_splits}")
        return
    
    # Validate test_ratio
    if args.test_ratio < 0 or args.test_ratio >= 1.0:
        print(f"Error: test_ratio must be between 0 and 1.0, got {args.test_ratio}")
        return
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate input directory
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory '{args.dataset_dir}' does not exist")
        return
    
    # Run diagnosis if requested
    if args.diagnose:
        is_clean = diagnose_dataset(args.dataset_dir, args.n_splits)
        if not is_clean:
            print("Dataset has issues that may affect splitting quality.")
            response = input("Continue anyway? (y/n): ").lower()
            if response != 'y':
                return
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION DATASET SPLIT")
    print("="*60)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of folds: {args.n_splits}")
    print(f"Test set ratio: {args.test_ratio}")
    print(f"Random seed: {args.seed}")
    
    # Get class images
    class_to_images = get_class_images(args.dataset_dir)
    
    if not class_to_images:
        print("Error: No images found in class directories")
        return
    
    # Analyze class distribution
    class_data = analyze_class_distribution(class_to_images)
    
    # Create k-fold splits
    all_folds = stratified_kfold_split(class_to_images, args.n_splits, args.test_ratio)
    
    # Process each fold
    class_names = list(class_to_images.keys())
    fold_summaries = []
    
    for fold_idx, (train_dict, valid_dict, test_dict) in enumerate(all_folds):
        print(f"\n{'='*60}")
        print(f"PROCESSING FOLD {fold_idx + 1}/{args.n_splits}")
        print(f"{'='*60}")
        
        # Create directory structure for this fold
        create_directory_structure(args.output_dir, fold_idx + 1, class_names, 
                                  include_test=(args.test_ratio > 0))
        
        # Copy files
        train_success, train_actual, train_class_counts = copy_files(
            train_dict, args.output_dir, fold_idx + 1, 'train'
        )
        valid_success, valid_actual, valid_class_counts = copy_files(
            valid_dict, args.output_dir, fold_idx + 1, 'val'
        )
        
        if args.test_ratio > 0 and test_dict:
            test_success, test_actual, test_class_counts = copy_files(
                test_dict, args.output_dir, fold_idx + 1, 'test'
            )
        else:
            test_success, test_actual, test_class_counts = 0, 0, {}
        
        fold_summaries.append({
            'fold': fold_idx + 1,
            'train': train_actual,
            'val': valid_actual,
            'test': test_actual,
            'train_classes': train_class_counts,
            'val_classes': valid_class_counts,
            'test_classes': test_class_counts
        })
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - ALL FOLDS")
    print("="*60)
    print(f"Original dataset: {class_data['total_images']} images across {len(class_names)} classes")
    print(f"Number of folds created: {len(all_folds)}")
    print()
    
    for summary in fold_summaries:
        print(f"Fold {summary['fold']}:")
        print(f"  Train: {summary['train']} images")
        print(f"  Val:   {summary['val']} images")
        if args.test_ratio > 0:
            print(f"  Test:  {summary['test']} images")
        print()
    
    # Show class distribution for each fold
    print("Class distribution per fold:")
    print(f"{'Class':<20} " + " ".join([f"F{i+1}_train" for i in range(args.n_splits)]) + 
          " " + " ".join([f"F{i+1}_val" for i in range(args.n_splits)]))
    print("-" * (20 + 10 * args.n_splits * 2))
    
    for class_name in sorted(class_names):
        train_counts = [str(fold_summaries[i]['train_classes'].get(class_name, 0)) for i in range(args.n_splits)]
        val_counts = [str(fold_summaries[i]['val_classes'].get(class_name, 0)) for i in range(args.n_splits)]
        
        train_str = " ".join([f"{c:>8}" for c in train_counts])
        val_str = " ".join([f"{c:>6}" for c in val_counts])
        print(f"{class_name:<20} {train_str} {val_str}")
    
    print("\n" + "="*60)
    print("✓ Cross-validation dataset split completed!")
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    for fold_idx in range(1, args.n_splits + 1):
        print(f"    ├── fold_{fold_idx}/")
        print(f"    │   ├── train/")
        for class_name in sorted(class_names)[:2]:
            print(f"    │   │   ├── {class_name}/")
        if len(class_names) > 2:
            print(f"    │   │   └── ... ({len(class_names)} classes total)")
        print(f"    │   ├── val/")
        for class_name in sorted(class_names)[:2]:
            print(f"    │   │   ├── {class_name}/")
        if len(class_names) > 2:
            print(f"    │   │   └── ... ({len(class_names)} classes total)")
        if args.test_ratio > 0:
            print(f"    │   └── test/")
    
    print("\n💡 Training tip:")
    print(f"   Train {args.n_splits} separate models, one for each fold:")
    for fold_idx in range(1, args.n_splits + 1):
        print(f"   - Model {fold_idx}: train on fold_{fold_idx}/train, validate on fold_{fold_idx}/val")
    print(f"   Then average the validation metrics across all {args.n_splits} folds for final performance.")

if __name__ == "__main__":
    main()

### Usage ###
# Example: Create 3-fold cross-validation splits
# python split_classification_cv.py --dataset_dir /path/to/dataset --output_dir /path/to/output --n_splits 3
#
# Example: Create 5-fold CV with a held-out test set (10%)
# python ./scripts/kfold_split_classification_dataset.py --dataset_dir D:/Dropbox/data/carabID/imgs/c1_augmented --output_dir D:/Dropbox/data/carabID/imgs/cv_classification_Set --n_splits 5 --test_ratio 0.1
#
# Example: With diagnosis to check dataset suitability
# python split_classification_cv.py --dataset_dir /path/to/dataset --output_dir /path/to/output --n_splits 3 --diagnose
