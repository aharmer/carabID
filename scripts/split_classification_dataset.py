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

def stratified_split_classification(class_to_images: Dict[str, List[str]], 
                                  train_ratio: float = 0.8, 
                                  valid_ratio: float = 0.15, 
                                  test_ratio: float = 0.05) -> Tuple[Dict, Dict, Dict]:
    """
    Perform stratified split for classification dataset.
    
    Args:
        class_to_images: Dictionary mapping class_name to list of image paths
        train_ratio: Proportion for training set
        valid_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
    
    Returns:
        Tuple of (train_dict, valid_dict, test_dict) where each dict maps class_name to image_paths
    """
    print("Performing stratified classification split...")
    
    train_dict = defaultdict(list)
    valid_dict = defaultdict(list)
    test_dict = defaultdict(list)
    
    # For each class, split proportionally
    for class_name, images in class_to_images.items():
        # Shuffle images within each class
        class_images = images.copy()
        random.shuffle(class_images)
        
        class_size = len(class_images)
        train_size = max(1, int(class_size * train_ratio))  # Ensure at least 1 item for training
        valid_size = int(class_size * valid_ratio)
        test_size = class_size - train_size - valid_size
        
        # Adjust if we have rounding issues
        if test_size < 0:
            test_size = 0
            valid_size = class_size - train_size
        
        # Split the class
        train_items = class_images[:train_size]
        valid_items = class_images[train_size:train_size + valid_size]
        test_items = class_images[train_size + valid_size:]
        
        # Add to respective splits
        train_dict[class_name] = train_items
        valid_dict[class_name] = valid_items
        test_dict[class_name] = test_items
        
        print(f"  Class '{class_name}': {class_size} total -> "
              f"train:{len(train_items)}, valid:{len(valid_items)}, test:{len(test_items)}")
    
    return dict(train_dict), dict(valid_dict), dict(test_dict)

def random_split_classification(class_to_images: Dict[str, List[str]], 
                              train_ratio: float = 0.8, 
                              valid_ratio: float = 0.15, 
                              test_ratio: float = 0.05) -> Tuple[Dict, Dict, Dict]:
    """
    Perform random split for classification dataset.
    
    Args:
        class_to_images: Dictionary mapping class_name to list of image paths
        train_ratio: Proportion for training set
        valid_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
    
    Returns:
        Tuple of (train_dict, valid_dict, test_dict) where each dict maps class_name to image_paths
    """
    print("Performing random classification split...")
    
    # Flatten all images into a single list with their class labels
    all_images = []
    for class_name, images in class_to_images.items():
        for image_path in images:
            all_images.append((image_path, class_name))
    
    # Shuffle all images
    random.shuffle(all_images)
    
    # Calculate split sizes
    total_images = len(all_images)
    train_size = int(total_images * train_ratio)
    valid_size = int(total_images * valid_ratio)
    
    # Split the images
    train_items = all_images[:train_size]
    valid_items = all_images[train_size:train_size + valid_size]
    test_items = all_images[train_size + valid_size:]
    
    # Reorganize back into class dictionaries
    def organize_by_class(items):
        class_dict = defaultdict(list)
        for image_path, class_name in items:
            class_dict[class_name].append(image_path)
        return dict(class_dict)
    
    train_dict = organize_by_class(train_items)
    valid_dict = organize_by_class(valid_items)
    test_dict = organize_by_class(test_items)
    
    # Print split information
    for class_name in class_to_images.keys():
        train_count = len(train_dict.get(class_name, []))
        valid_count = len(valid_dict.get(class_name, []))
        test_count = len(test_dict.get(class_name, []))
        total_count = len(class_to_images[class_name])
        
        print(f"  Class '{class_name}': {total_count} total -> "
              f"train:{train_count}, valid:{valid_count}, test:{test_count}")
    
    return train_dict, valid_dict, test_dict

def validate_stratification(original_data: Dict, 
                          train_dict: Dict[str, List[str]], 
                          valid_dict: Dict[str, List[str]], 
                          test_dict: Dict[str, List[str]]):
    """
    Validate that the stratification maintained class proportions.
    """
    print("\nValidating stratification...")
    
    original_counts = original_data['class_counts']
    
    def get_split_counts(split_dict):
        return {class_name: len(images) for class_name, images in split_dict.items()}
    
    train_counts = get_split_counts(train_dict)
    valid_counts = get_split_counts(valid_dict)
    test_counts = get_split_counts(test_dict)
    
    print("Class distribution validation:")
    print(f"{'Class':<20} {'Original %':<12} {'Train %':<12} {'Valid %':<12} {'Test %':<12}")
    print("-" * 80)
    
    total_train = sum(train_counts.values())
    total_valid = sum(valid_counts.values())
    total_test = sum(test_counts.values())
    
    max_deviation = 0
    
    for class_name in sorted(original_counts.keys()):
        orig_count = original_counts[class_name]
        train_count = train_counts.get(class_name, 0)
        valid_count = valid_counts.get(class_name, 0)
        test_count = test_counts.get(class_name, 0)
        
        # Calculate percentages within each split
        train_pct = (train_count / total_train * 100) if total_train > 0 else 0
        valid_pct = (valid_count / total_valid * 100) if total_valid > 0 else 0
        test_pct = (test_count / total_test * 100) if total_test > 0 else 0
        orig_pct = (orig_count / original_data['total_images'] * 100)
        
        print(f"{class_name:<20} {orig_pct:<8.1f}%    "
              f"{train_pct:<8.1f}%    "
              f"{valid_pct:<8.1f}%    "
              f"{test_pct:<8.1f}%")
        
        # Track maximum deviation for stratification quality
        deviation = abs(orig_pct - train_pct)
        max_deviation = max(max_deviation, deviation)
    
    # Check stratification quality
    print("\nStratification quality:")
    print(f"Maximum deviation from original distribution: {max_deviation:.2f}%")
    if max_deviation < 2.0:
        print("✓ Excellent stratification!")
    elif max_deviation < 5.0:
        print("✓ Good stratification")
    else:
        print("⚠ Stratification could be improved (may be due to very small classes)")

def create_directory_structure(output_dir: str, class_names: List[str]):
    """
    Create the train/valid/test directory structure with class subdirectories.
    
    Args:
        output_dir: Base output directory
        class_names: List of class names to create subdirectories for
    """
    splits = ['train', 'val', 'test']
    
    for split in splits:
        for class_name in class_names:
            dir_path = Path(output_dir) / split / class_name
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created {split} directory with {len(class_names)} class subdirectories")

def copy_files(split_dict: Dict[str, List[str]], output_dir: str, split_name: str):
    """
    Copy image files to the appropriate split directory maintaining class structure.
    
    Args:
        split_dict: Dictionary mapping class_name to list of image paths
        output_dir: Base output directory
        split_name: Name of the split (train, valid, test)
    """
    split_dest = Path(output_dir) / split_name
    
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
                    print(f"Warning: Image file {image_name} already exists in {split_name}/{class_name}, skipping...")
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
    
    print(f"Split {split_name}:")
    print(f"  Expected: {total_expected} files")
    print(f"  Successful copies: {successful_copies}")
    print(f"  Actual files: {actual_files}")
    
    # Show per-class counts
    for class_name, count in class_counts.items():
        expected_count = len(split_dict.get(class_name, []))
        print(f"    {class_name}: {count}/{expected_count} files")
    
    if failed_copies:
        print(f"  Failed copies ({len(failed_copies)}):")
        for failure in failed_copies[:5]:  # Show first 5 failures
            print(f"    - {failure}")
        if len(failed_copies) > 5:
            print(f"    ... and {len(failed_copies) - 5} more")
    
    if successful_copies != total_expected:
        print(f"  WARNING: Expected {total_expected} successful copies, got {successful_copies}")
    
    return successful_copies, actual_files, class_counts

def diagnose_dataset(dataset_dir: str):
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
        
        # Check for very small classes
        if len(image_files) < 3:
            print(f"  Warning: Class '{class_name}' has very few images ({len(image_files)})")
            print(f"    This may cause issues with stratified splitting")
            issues_found = True
        
        # Check for duplicate filenames
        stems = {}
        for img_file in image_files:
            stem = img_file.stem
            if stem not in stems:
                stems[stem] = []
            stems[stem].append(img_file.name)
        
        duplicates = {k: v for k, v in stems.items() if len(v) > 1}
        if duplicates:
            print(f"  Warning: Found duplicate stems in class '{class_name}':")
            for stem, files in list(duplicates.items())[:3]:
                print(f"    '{stem}': {files}")
            if len(duplicates) > 3:
                print(f"    ... and {len(duplicates) - 3} more")
            issues_found = True
    
    print("="*50)
    
    if issues_found:
        print("⚠ Issues found that may affect splitting quality")
    else:
        print("✓ Dataset looks clean")
    
    return not issues_found

def main():
    parser = argparse.ArgumentParser(description='Split YOLO classification dataset into train/valid/test sets using stratified or random sampling')
    parser.add_argument('--dataset_dir', required=True, help='Path to classification dataset directory containing class subdirectories')
    parser.add_argument('--output_dir', required=True, help='Output directory for split dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio (default: 0.8)')
    parser.add_argument('--valid_ratio', type=float, default=0.15, help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.05, help='Test set ratio (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits (default: 42)')
    parser.add_argument('--diagnose', action='store_true', help='Run dataset diagnosis before splitting')
    parser.add_argument('--no_stratify', action='store_true', help='Use random splitting instead of stratified (for comparison)')
    
    args = parser.parse_args()
    
    # Validate ratios sum to 1
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0, got {total_ratio}")
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
        is_clean = diagnose_dataset(args.dataset_dir)
        if not is_clean:
            print("Dataset has issues that may affect splitting quality.")
            response = input("Continue anyway? (y/n): ").lower()
            if response != 'y':
                return
    
    print("Starting classification dataset split...")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios - Train: {args.train_ratio}, Valid: {args.valid_ratio}, Test: {args.test_ratio}")
    print(f"Stratified splitting: {'Disabled' if args.no_stratify else 'Enabled'}")
    
    # Get class images
    class_to_images = get_class_images(args.dataset_dir)
    
    if not class_to_images:
        print("Error: No images found in class directories")
        return
    
    # Analyze class distribution
    class_data = analyze_class_distribution(class_to_images)
    
    # Create output directory structure
    class_names = list(class_to_images.keys())
    create_directory_structure(args.output_dir, class_names)
    
    # Split the dataset
    if args.no_stratify:
        train_dict, valid_dict, test_dict = random_split_classification(
            class_to_images, args.train_ratio, args.valid_ratio, args.test_ratio
        )
        print("Using random (non-stratified) split")
    else:
        train_dict, valid_dict, test_dict = stratified_split_classification(
            class_to_images, args.train_ratio, args.valid_ratio, args.test_ratio
        )
        print("Using stratified split")
    
    # Calculate total split sizes
    train_total = sum(len(images) for images in train_dict.values())
    valid_total = sum(len(images) for images in valid_dict.values())
    test_total = sum(len(images) for images in test_dict.values())
    
    print(f"Split sizes - Train: {train_total}, Valid: {valid_total}, Test: {test_total}")
    
    # Validate stratification
    validate_stratification(class_data, train_dict, valid_dict, test_dict)
    
    # Copy files to respective directories
    train_success, train_actual, train_class_counts = copy_files(train_dict, args.output_dir, 'train')
    valid_success, valid_actual, valid_class_counts = copy_files(valid_dict, args.output_dir, 'val')
    test_success, test_actual, test_class_counts = copy_files(test_dict, args.output_dir, 'test')
    
    print("\n" + "="*50)
    print("FINAL SUMMARY:")
    print("="*50)
    print(f"Original dataset: {class_data['total_images']} images across {len(class_names)} classes")
    print(f"Expected splits - Train: {train_total}, Valid: {valid_total}, Test: {test_total}")
    print(f"Successful copies - Train: {train_success}, Valid: {valid_success}, Test: {test_success}")
    print(f"Actual file counts - Train: {train_actual}, Valid: {valid_actual}, Test: {test_actual}")
    
    total_expected = train_total + valid_total + test_total
    total_actual = train_actual + valid_actual + test_actual
    
    if total_actual != total_expected:
        print(f"\nWARNING: File count mismatch!")
        print(f"Expected total: {total_expected}, Actual total: {total_actual}")
        print(f"Difference: {total_expected - total_actual}")
    else:
        print(f"\n✓ All files copied successfully!")
    
    print("\nDataset split completed!")
    print(f"Output structure:")
    print(f"  {args.output_dir}/")
    print(f"    ├── train/ ({train_actual} files)")
    for class_name in sorted(class_names):
        count = train_class_counts.get(class_name, 0)
        print(f"    │   ├── {class_name}/ ({count} files)")
    print(f"    ├── val/ ({valid_actual} files)")
    for class_name in sorted(class_names):
        count = valid_class_counts.get(class_name, 0)
        print(f"    │   ├── {class_name}/ ({count} files)")
    print(f"    └── test/ ({test_actual} files)")
    for class_name in sorted(class_names):
        count = test_class_counts.get(class_name, 0)
        print(f"        ├── {class_name}/ ({count} files)")

if __name__ == "__main__":
    main()

### Usage ###
# Example: Split a classification dataset with stratified sampling
# python ./scripts/split_classification_dataset.py --dataset_dir D:/Dropbox/data/carabID/imgs/c1_augmented --output_dir D:/Dropbox/data/carabID/imgs/c1_split
#
# Example: Use random sampling instead of stratified
# python split_classification_dataset.py --dataset_dir /path/to/classification/dataset --output_dir /path/to/output --no_stratify
#
# Example: Custom split ratios with diagnosis
# python split_classification_dataset.py --dataset_dir /path/to/dataset --output_dir /path/to/output --train_ratio 0.7 --valid_ratio 0.2 --test_ratio 0.1 --diagnose
