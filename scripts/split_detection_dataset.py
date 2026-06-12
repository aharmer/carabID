import os
import shutil
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import argparse

def parse_yolo_label(label_path: str) -> int:
    """
    Parse YOLO label file and return the class ID (assuming single class per image).
    
    Args:
        label_path: Path to YOLO label file
    
    Returns:
        Class ID of the first (and presumably only) object in the label file
    """
    try:
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                # First number in the line is the class ID
                class_id = int(first_line.split()[0])
                return class_id
            else:
                print(f"Warning: Empty label file {label_path}")
                return -1
    except Exception as e:
        print(f"Warning: Could not parse label file {label_path}: {e}")
        return -1

def analyze_class_distribution(pairs: List[Tuple[str, str]]) -> Dict:
    """
    Analyze class distribution in the dataset.
    
    Args:
        pairs: List of (image_path, label_path) tuples
    
    Returns:
        Dictionary with class distribution statistics
    """
    print("Analyzing class distribution...")
    
    # Count class occurrences and group images by class
    class_counts = Counter()
    class_to_pairs = defaultdict(list)
    
    for image_path, label_path in pairs:
        class_id = parse_yolo_label(label_path)
        if class_id >= 0:  # Valid class ID
            class_counts[class_id] += 1
            class_to_pairs[class_id].append((image_path, label_path))
    
    # Calculate statistics
    total_images = sum(class_counts.values())
    unique_classes = set(class_counts.keys())
    
    print(f"Found {len(unique_classes)} unique classes in {total_images} images")
    print("Class distribution:")
    for class_id in sorted(unique_classes):
        count = class_counts[class_id]
        percentage = (count / total_images) * 100
        print(f"  Class {class_id}: {count} images ({percentage:.1f}%)")
    
    return {
        'class_counts': class_counts,
        'class_to_pairs': class_to_pairs,
        'unique_classes': unique_classes,
        'total_images': total_images
    }

def stratified_split_single_label(class_to_pairs: Dict[int, List[Tuple[str, str]]], 
                                 train_ratio: float = 0.8, 
                                 valid_ratio: float = 0.15, 
                                 test_ratio: float = 0.05) -> Tuple[List, List, List]:
    """
    Perform stratified split for single-label dataset.
    
    Args:
        class_to_pairs: Dictionary mapping class_id to list of (image_path, label_path) tuples
        train_ratio: Proportion for training set
        valid_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
    
    Returns:
        Tuple of (train_pairs, valid_pairs, test_pairs)
    """
    print("Performing stratified single-label split...")
    
    train_pairs = []
    valid_pairs = []
    test_pairs = []
    
    # For each class, split proportionally
    for class_id, pairs in class_to_pairs.items():
        # Shuffle pairs within each class
        class_pairs = pairs.copy()
        random.shuffle(class_pairs)
        
        class_size = len(class_pairs)
        train_size = max(1, int(class_size * train_ratio))  # Ensure at least 1 item for training
        valid_size = int(class_size * valid_ratio)
        test_size = class_size - train_size - valid_size
        
        # Adjust if we have rounding issues
        if test_size < 0:
            test_size = 0
            valid_size = class_size - train_size
        
        # Split the class
        train_items = class_pairs[:train_size]
        valid_items = class_pairs[train_size:train_size + valid_size]
        test_items = class_pairs[train_size + valid_size:]
        
        # Add to respective splits
        train_pairs.extend(train_items)
        valid_pairs.extend(valid_items)
        test_pairs.extend(test_items)
        
        print(f"  Class {class_id}: {class_size} total -> "
              f"train:{len(train_items)}, valid:{len(valid_items)}, test:{len(test_items)}")
    
    # Shuffle the final splits to mix classes
    random.shuffle(train_pairs)
    random.shuffle(valid_pairs)
    random.shuffle(test_pairs)
    
    return train_pairs, valid_pairs, test_pairs

def validate_stratification(original_data: Dict, 
                          train_pairs: List[Tuple[str, str]], 
                          valid_pairs: List[Tuple[str, str]], 
                          test_pairs: List[Tuple[str, str]]):
    """
    Validate that the stratification maintained class proportions.
    """
    print("\nValidating stratification...")
    
    def get_split_class_counts(pairs):
        counts = Counter()
        for _, label_path in pairs:
            class_id = parse_yolo_label(label_path)
            if class_id >= 0:
                counts[class_id] += 1
        return counts
    
    original_counts = original_data['class_counts']
    train_counts = get_split_class_counts(train_pairs)
    valid_counts = get_split_class_counts(valid_pairs)
    test_counts = get_split_class_counts(test_pairs)
    
    print("Class distribution validation:")
    print(f"{'Class':<8} {'Original':<12} {'Train %':<12} {'Valid %':<12} {'Test %':<12}")
    print("-" * 60)
    
    total_train = sum(train_counts.values())
    total_valid = sum(valid_counts.values())
    total_test = sum(test_counts.values())
    
    for class_id in sorted(original_counts.keys()):
        orig_count = original_counts[class_id]
        train_count = train_counts.get(class_id, 0)
        valid_count = valid_counts.get(class_id, 0)
        test_count = test_counts.get(class_id, 0)
        
        # Calculate percentages within each split
        train_pct = (train_count / total_train * 100) if total_train > 0 else 0
        valid_pct = (valid_count / total_valid * 100) if total_valid > 0 else 0
        test_pct = (test_count / total_test * 100) if total_test > 0 else 0
        orig_pct = (orig_count / original_data['total_images'] * 100)
        
        print(f"{class_id:<8} {orig_pct:<8.1f}%    "
              f"{train_pct:<8.1f}%    "
              f"{valid_pct:<8.1f}%    "
              f"{test_pct:<8.1f}%")
    
    # Check how well proportions are maintained
    print("\nStratification quality:")
    max_deviation = 0
    for class_id in sorted(original_counts.keys()):
        orig_pct = (original_counts[class_id] / original_data['total_images'] * 100)
        train_pct = (train_counts.get(class_id, 0) / total_train * 100) if total_train > 0 else 0
        deviation = abs(orig_pct - train_pct)
        max_deviation = max(max_deviation, deviation)
    
    print(f"Maximum deviation from original distribution: {max_deviation:.2f}%")
    if max_deviation < 2.0:
        print("✓ Excellent stratification!")
    elif max_deviation < 5.0:
        print("✓ Good stratification")
    else:
        print("⚠ Stratification could be improved (may be due to very small classes)")

def get_image_label_pairs(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    """
    Get matching image-label pairs from the dataset.
    
    Args:
        images_dir: Path to images directory
        labels_dir: Path to labels directory
    
    Returns:
        List of tuples containing (image_path, label_path)
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Use a set to automatically deduplicate files (handles case-insensitive filesystems)
    image_files_set = set()
    for ext in image_extensions:
        image_files_set.update(Path(images_dir).glob(f'*{ext}'))
        image_files_set.update(Path(images_dir).glob(f'*{ext.upper()}'))
    
    image_files = list(image_files_set)
    print(f"Found {len(image_files)} unique image files after deduplication")
    
    pairs = []
    seen_stems = set()  # Track stems to detect duplicates
    
    for image_path in image_files:
        stem = image_path.stem
        
        # Check for duplicate stems (same name, different extension)
        if stem in seen_stems:
            print(f"Warning: Duplicate image stem '{stem}' found - {image_path.name}")
            print(f"  This could cause filename conflicts during copying!")
            continue
        
        seen_stems.add(stem)
        
        # Corresponding label file should have same name but .txt extension
        label_name = stem + '.txt'
        label_path = Path(labels_dir) / label_name
        
        if label_path.exists():
            pairs.append((str(image_path), str(label_path)))
        else:
            print(f"Warning: No label file found for {image_path.name}")
    
    print(f"Found {len(pairs)} unique image-label pairs")
    if len(image_files) != len(pairs):
        print(f"Note: {len(image_files) - len(pairs)} image files had no matching labels")
    
    return pairs

def create_directory_structure(output_dir: str):
    """
    Create the train/valid/test directory structure with images and labels subdirectories.
    
    Args:
        output_dir: Base output directory
    """
    splits = ['train', 'valid', 'test']
    subdirs = ['images', 'labels']
    
    for split in splits:
        for subdir in subdirs:
            dir_path = Path(output_dir) / split / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")

def copy_files(pairs: List[Tuple[str, str]], output_dir: str, split_name: str):
    """
    Copy image and label files to the appropriate split directory.
    
    Args:
        pairs: List of (image_path, label_path) tuples
        output_dir: Base output directory
        split_name: Name of the split (train, valid, test)
    """
    images_dest = Path(output_dir) / split_name / 'images'
    labels_dest = Path(output_dir) / split_name / 'labels'
    
    successful_copies = 0
    failed_copies = []
    
    for i, (image_path, label_path) in enumerate(pairs):
        try:
            # Copy image file
            image_name = Path(image_path).name
            image_dest_path = images_dest / image_name
            
            # Check for filename conflicts
            if image_dest_path.exists():
                print(f"Warning: Image file {image_name} already exists in {split_name}, skipping...")
                failed_copies.append(f"Image conflict: {image_name}")
                continue
            
            shutil.copy2(image_path, image_dest_path)
            
            # Copy label file
            label_name = Path(label_path).name
            label_dest_path = labels_dest / label_name
            
            if label_dest_path.exists():
                print(f"Warning: Label file {label_name} already exists in {split_name}, skipping...")
                failed_copies.append(f"Label conflict: {label_name}")
                continue
                
            shutil.copy2(label_path, label_dest_path)
            successful_copies += 1
            
        except Exception as e:
            print(f"Error copying pair {i+1}: {e}")
            failed_copies.append(f"Copy error: {Path(image_path).name} - {str(e)}")
    
    # Verify actual file counts
    actual_images = len(list(images_dest.glob('*')))
    actual_labels = len(list(labels_dest.glob('*')))
    
    print(f"Split {split_name}:")
    print(f"  Expected: {len(pairs)} pairs")
    print(f"  Successful copies: {successful_copies}")
    print(f"  Actual files - Images: {actual_images}, Labels: {actual_labels}")
    
    if failed_copies:
        print(f"  Failed copies ({len(failed_copies)}):")
        for failure in failed_copies[:5]:  # Show first 5 failures
            print(f"    - {failure}")
        if len(failed_copies) > 5:
            print(f"    ... and {len(failed_copies) - 5} more")
    
    if successful_copies != len(pairs):
        print(f"  WARNING: Expected {len(pairs)} successful copies, got {successful_copies}")
    
    if actual_images != actual_labels:
        print(f"  WARNING: Mismatch between images ({actual_images}) and labels ({actual_labels})")
    
    return successful_copies, actual_images, actual_labels

def diagnose_dataset(images_dir: str, labels_dir: str):
    """
    Diagnose potential issues in the dataset before splitting.
    """
    print("="*50)
    print("DATASET DIAGNOSIS")
    print("="*50)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    print(f"Total image files found: {len(image_files)}")
    
    # Check for duplicate stems
    stems = {}
    for img in image_files:
        stem = img.stem
        if stem not in stems:
            stems[stem] = []
        stems[stem].append(img.name)
    
    duplicates = {k: v for k, v in stems.items() if len(v) > 1}
    if duplicates:
        print(f"Found {len(duplicates)} duplicate stems:")
        for stem, files in list(duplicates.items())[:5]:
            print(f"  '{stem}': {files}")
        if len(duplicates) > 5:
            print(f"  ... and {len(duplicates) - 5} more")
    
    # Check label files
    label_files = list(Path(labels_dir).glob('*.txt'))
    print(f"Total label files found: {len(label_files)}")
    
    # Check matching
    matched = 0
    unmatched_images = []
    unmatched_labels = []
    
    label_stems = {f.stem for f in label_files}
    
    for img in image_files:
        if img.stem in label_stems:
            matched += 1
        else:
            unmatched_images.append(img.name)
    
    for label in label_files:
        img_found = any(img.stem == label.stem for img in image_files)
        if not img_found:
            unmatched_labels.append(label.name)
    
    print(f"Matched pairs: {matched}")
    if unmatched_images:
        print(f"Images without labels: {len(unmatched_images)}")
        for img in unmatched_images[:3]:
            print(f"  - {img}")
        if len(unmatched_images) > 3:
            print(f"  ... and {len(unmatched_images) - 3} more")
    
    if unmatched_labels:
        print(f"Labels without images: {len(unmatched_labels)}")
        for lbl in unmatched_labels[:3]:
            print(f"  - {lbl}")
        if len(unmatched_labels) > 3:
            print(f"  ... and {len(unmatched_labels) - 3} more")
    
    print("="*50)
    return len(duplicates) == 0

def main():
    parser = argparse.ArgumentParser(description='Split single-class YOLO dataset into train/valid/test sets using stratified sampling')
    parser.add_argument('--images_dir', required=True, help='Path to images directory')
    parser.add_argument('--labels_dir', required=True, help='Path to labels directory')
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
    
    # Validate input directories
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory '{args.images_dir}' does not exist")
        return
    
    if not os.path.exists(args.labels_dir):
        print(f"Error: Labels directory '{args.labels_dir}' does not exist")
        return
    
    # Run diagnosis if requested
    if args.diagnose:
        is_clean = diagnose_dataset(args.images_dir, args.labels_dir)
        if not is_clean:
            print("Dataset has issues that may cause conflicts. Fix these before splitting.")
            return
    
    print("Starting stratified dataset split...")
    print(f"Images directory: {args.images_dir}")
    print(f"Labels directory: {args.labels_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios - Train: {args.train_ratio}, Valid: {args.valid_ratio}, Test: {args.test_ratio}")
    print(f"Stratified splitting: {'Disabled' if args.no_stratify else 'Enabled'}")
    
    # Get image-label pairs
    pairs = get_image_label_pairs(args.images_dir, args.labels_dir)
    print(f"Found {len(pairs)} image-label pairs")
    
    if len(pairs) == 0:
        print("Error: No matching image-label pairs found")
        return
    
    # Analyze class distribution
    class_data = analyze_class_distribution(pairs)
    
    # Create output directory structure
    create_directory_structure(args.output_dir)
    
    # Split the dataset
    if args.no_stratify:
        # Use original random split for comparison
        random.shuffle(pairs)
        total_samples = len(pairs)
        train_size = int(total_samples * args.train_ratio)
        valid_size = int(total_samples * args.valid_ratio)
        
        train_pairs = pairs[:train_size]
        valid_pairs = pairs[train_size:train_size + valid_size]
        test_pairs = pairs[train_size + valid_size:]
        print("Using random (non-stratified) split")
    else:
        # Use stratified split
        train_pairs, valid_pairs, test_pairs = stratified_split_single_label(
            class_data['class_to_pairs'], 
            args.train_ratio, 
            args.valid_ratio, 
            args.test_ratio
        )
        print("Using stratified split")
    
    print(f"Split sizes - Train: {len(train_pairs)}, Valid: {len(valid_pairs)}, Test: {len(test_pairs)}")
    
    # Validate stratification
    validate_stratification(class_data, train_pairs, valid_pairs, test_pairs)
    
    # Copy files to respective directories
    train_success, train_images, train_labels = copy_files(train_pairs, args.output_dir, 'train')
    valid_success, valid_images, valid_labels = copy_files(valid_pairs, args.output_dir, 'valid')
    test_success, test_images, test_labels = copy_files(test_pairs, args.output_dir, 'test')
    
    print("\n" + "="*50)
    print("FINAL SUMMARY:")
    print("="*50)
    print(f"Original dataset: {len(pairs)} image-label pairs")
    print(f"Expected splits - Train: {len(train_pairs)}, Valid: {len(valid_pairs)}, Test: {len(test_pairs)}")
    print(f"Successful copies - Train: {train_success}, Valid: {valid_success}, Test: {test_success}")
    print(f"Actual file counts:")
    print(f"  Train: {train_images} images, {train_labels} labels")
    print(f"  Valid: {valid_images} images, {valid_labels} labels") 
    print(f"  Test: {test_images} images, {test_labels} labels")
    
    total_expected = len(train_pairs) + len(valid_pairs) + len(test_pairs)
    total_actual = train_images + valid_images + test_images
    
    if total_actual != total_expected:
        print(f"\nWARNING: File count mismatch!")
        print(f"Expected total: {total_expected}, Actual total: {total_actual}")
        print(f"Difference: {total_expected - total_actual}")
    else:
        print(f"\n✓ All files copied successfully!")
    
    print("\nDataset split completed!")
    print(f"Output structure:")
    print(f"  {args.output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── images/ ({train_images} files)")
    print(f"    │   └── labels/ ({train_labels} files)")
    print(f"    ├── valid/")
    print(f"    │   ├── images/ ({valid_images} files)")
    print(f"    │   └── labels/ ({valid_labels} files)")
    print(f"    └── test/")
    print(f"        ├── images/ ({test_images} files)")
    print(f"        └── labels/ ({test_labels} files)")

if __name__ == "__main__":
    main()

### Usage ###
# python ./scripts/split_detection_dataset.py --images_dir D:/Dropbox/data/carabID/imgs/combined_augmented/train/images --labels_dir D:/Dropbox/data/carabID/imgs/combined_augmented/train/labels --output_dir D:/Dropbox/data/carabID/imgs/detection_set
