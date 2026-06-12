import os
import cv2
import numpy as np
import sys
import argparse

################################################################################
######                      Command Line Arguments                        ######
################################################################################

def parse_arguments():
    """Parse command line arguments for input and output directories."""
    parser = argparse.ArgumentParser(
        description="Preprocess images for YOLO classification with selective modifications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_images.py --input input_folder --output output_folder
  python preprocess_images.py -i ./yolo/images -o ./yolo/images_processed --grayscale --contrast
  python preprocess_images.py -i ./yolo/images -o ./yolo/images_processed --size 416 416 --all-mods
  python ./scripts/new_preprocess_images.py -i ./imgs/combined -o ./imgs/combined_processed --contrast --resize
        """
    )
    
    parser.add_argument('--input', '-i',
                       required=True,
                       dest='input_dir',
                       help='Input directory containing YOLO classification image structure')
    parser.add_argument('--output', '-o',
                       required=True, 
                       dest='output_dir',
                       help='Output directory for processed images')
    parser.add_argument('--size', '-s', 
                       nargs=2, 
                       type=int, 
                       default=[640, 640],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Target size for resized images (default: 640 640)')
    parser.add_argument('--grayscale', '-g',
                       action='store_true',
                       help='Convert images to grayscale')
    parser.add_argument('--contrast', '-c',
                       action='store_true',
                       help='Apply contrast stretching')
    parser.add_argument('--resize', '-r',
                       action='store_true',
                       help='Resize images to target size')
    parser.add_argument('--all-mods', '-a',
                       action='store_true',
                       help='Apply all modifications (grayscale, contrast, resize)')
    
    return parser.parse_args()

################################################################################
######                      Image Preprocessing Functions                 ######
################################################################################

def contrast_stretch(image):
    """
    Apply contrast stretching by mapping pixel intensities so that the
    2nd percentile becomes 0 and the 98th percentile becomes 255.
    This approach is more robust to outliers compared to using min/max values.
    """
    p2 = np.percentile(image, 2)
    p98 = np.percentile(image, 98)
    
    if p98 - p2 == 0:
        return image
    
    stretched = (image - p2) * (255.0 / (p98 - p2))
    stretched = np.clip(stretched, 0, 255)
    
    return stretched.astype(np.uint8)

def preprocess_image(image, modifications):
    """
    Apply selected preprocessing modifications to the image.
    
    Args:
        image: Input image (BGR)
        modifications: Dictionary with modification flags and parameters
                      {'resize': bool, 'target_size': tuple, 'grayscale': bool, 'contrast': bool}
    
    Returns:
        Processed image
    """
    processed = image.copy()
    
    # Apply resize
    if modifications['resize']:
        processed = cv2.resize(processed, modifications['target_size'])
    
    # Apply grayscale conversion
    if modifications['grayscale']:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast stretching
    if modifications['contrast']:
        processed = contrast_stretch(processed)
    
    # Convert back to BGR if grayscale was applied but we need color output
    if modifications['grayscale'] and len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    return processed

################################################################################
######                      Process YOLO Classification Structure         ######
################################################################################

def is_yolo_classification_structure(input_root):
    """
    Detect if directory follows YOLO classification structure (train/val/test with class subdirs).
    Returns a tuple: (is_yolo_structure, structure_type, description)
    """
    # Check for common YOLO structure: train/val/test with class subdirectories
    yolo_splits = ['train', 'val', 'test']
    found_splits = 0
    split_info = {}
    
    for split in yolo_splits:
        split_path = os.path.join(input_root, split)
        if os.path.isdir(split_path):
            found_splits += 1
            
            # Check for images or class subdirectories in this split
            items = os.listdir(split_path)
            images_in_split = []
            class_dirs = []
            
            for item in items:
                item_path = os.path.join(split_path, item)
                if os.path.isfile(item_path) and item.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    images_in_split.append(item)
                elif os.path.isdir(item_path):
                    subdir_images = [f for f in os.listdir(item_path)
                                    if os.path.isfile(os.path.join(item_path, f))
                                    and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
                    if subdir_images:
                        class_dirs.append((item, len(subdir_images)))
            
            split_info[split] = {'images': len(images_in_split), 'classes': class_dirs}
    
    # Determine structure type
    if found_splits >= 1:
        has_classes = any(len(info['classes']) > 0 for info in split_info.values())
        if has_classes:
            return True, 'yolo_class_structure', split_info
        else:
            return True, 'yolo_flat_structure', split_info
    
    # Check for flat structure with class subdirectories (no train/val/test split)
    items = os.listdir(input_root)
    class_dirs_in_root = []
    images_in_root = []
    
    for item in items:
        item_path = os.path.join(input_root, item)
        if os.path.isfile(item_path) and item.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            images_in_root.append(item)
        elif os.path.isdir(item_path):
            subdir_images = [f for f in os.listdir(item_path)
                            if os.path.isfile(os.path.join(item_path, f))
                            and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
            if subdir_images:
                class_dirs_in_root.append((item, len(subdir_images)))
    
    if len(class_dirs_in_root) > 0:
        return True, 'class_structure', {'root': {'images': len(images_in_root), 'classes': class_dirs_in_root}}
    elif len(images_in_root) > 0:
        return True, 'flat_structure', {'root': {'images': len(images_in_root), 'classes': []}}
    
    return False, 'unknown', {}

def process_yolo_dataset(input_root, output_root, modifications):
    """
    Process all images in a YOLO classification dataset while maintaining structure.
    Handles train/val/test splits with class subdirectories.
    """
    total_processed = 0
    
    # Detect structure
    is_yolo, structure_type, structure_info = is_yolo_classification_structure(input_root)
    
    if not is_yolo:
        print("Error: Input directory does not appear to be a YOLO classification dataset.")
        print("Expected structure: root/[train/val/test]/class_name/images")
        return 0
    
    print(f"Detected structure: {structure_type}")
    
    if structure_type == 'yolo_class_structure':
        print("Structure: train/val/test splits with class subdirectories")
        for split, info in structure_info.items():
            if info['images'] > 0 or info['classes']:
                print(f"  {split}: {info['images']} direct images, {len(info['classes'])} classes")
    elif structure_type == 'class_structure':
        print("Structure: Class subdirectories (no train/val/test split)")
        info = structure_info['root']
        print(f"  {len(info['classes'])} classes, {info['images']} images in root")
    elif structure_type == 'flat_structure':
        print("Structure: Flat directory (all images in root)")
    
    # Process all directories recursively
    for root, dirs, files in os.walk(input_root):
        rel_path = os.path.relpath(root, input_root)
        
        if rel_path == ".":
            current_output_dir = output_root
        else:
            current_output_dir = os.path.join(output_root, rel_path)
        
        os.makedirs(current_output_dir, exist_ok=True)
        
        # Filter for image files
        image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
        
        if image_files:
            if rel_path == ".":
                print(f"\nProcessing root directory")
            else:
                path_display = rel_path.replace(os.sep, ' / ')
                print(f"\nProcessing: {path_display}")
            print(f"  Found {len(image_files)} image(s)")
        
        for filename in image_files:
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"  Warning: {filename} could not be read.")
                continue
            
            processed_image = preprocess_image(image, modifications)
            
            output_path = os.path.join(current_output_dir, filename)
            cv2.imwrite(output_path, processed_image)
            print(f"  Processed: {filename}")
            total_processed += 1
    
    return total_processed

def print_modifications(modifications):
    """Print which modifications will be applied."""
    print("\nApplying modifications:")
    if modifications['resize']:
        print(f"  • Resize: {modifications['target_size'][0]}x{modifications['target_size'][1]}")
    if modifications['grayscale']:
        print("  • Convert to grayscale")
    if modifications['contrast']:
        print("  • Apply contrast stretching")
    if not any([modifications['resize'], modifications['grayscale'], modifications['contrast']]):
        print("  • No modifications selected (copy only)")

# Main execution
if __name__ == "__main__":
    args = parse_arguments()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    target_size = tuple(args.size)
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory.")
        sys.exit(1)
    
    # Determine which modifications to apply
    if args.all_mods:
        modifications = {
            'resize': True,
            'grayscale': True,
            'contrast': True,
            'target_size': target_size
        }
    else:
        modifications = {
            'resize': args.resize,
            'grayscale': args.grayscale,
            'contrast': args.contrast,
            'target_size': target_size
        }
    
    # If no modifications specified, default to resize only
    if not any([modifications['resize'], modifications['grayscale'], modifications['contrast']]):
        print("Note: No modifications selected. Defaulting to resize only.")
        modifications['resize'] = True
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting image preprocessing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print_modifications(modifications)
    
    total_images = process_yolo_dataset(input_dir, output_dir, modifications)
    
    print(f"\n{'='*60}")
    print(f"All done! Processed {total_images} images total.")
    print(f"{'='*60}")
