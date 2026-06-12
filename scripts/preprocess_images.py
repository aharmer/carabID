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
        description="Preprocess images for YOLO training with contrast stretching and resizing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_preprocessing.py --input input_folder --output output_folder
  python image_preprocessing.py -i "D:/data/images" -o "D:/data/processed"
  python image_preprocessing.py --input ./images --output ./processed --size 416 416
        """
    )
    
    parser.add_argument('--input', '-i',
                       required=True,
                       dest='input_dir',
                       help='Input directory containing images or class subdirectories')
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
    
    return parser.parse_args()

################################################################################
######                      Image Preprocessing Functions                 ######
################################################################################

def contrast_stretch(image):
    """
    Apply contrast stretching by mapping the pixel intensities so that the
    2nd percentile becomes 0 and the 98th percentile becomes 255.
    This approach is more robust to outliers compared to using min/max values.
    """
    # Compute the 2nd and 98th percentiles
    p2 = np.percentile(image, 2)
    p98 = np.percentile(image, 98)
    
    # Avoid division by zero if the percentile range is zero
    if p98 - p2 == 0:
        return image
    
    # Apply the linear transformation
    stretched = (image - p2) * (255.0 / (p98 - p2))
    
    # Clip values to ensure they stay within [0, 255] range
    stretched = np.clip(stretched, 0, 255)
    
    return stretched.astype(np.uint8)

def preprocess_image(image, target_size=(640, 640)):
    """
    Resize the image, convert it to grayscale, apply contrast stretching,
    and then convert it back to BGR.
    """
    # Resize the image to the target dimensions
    resized = cv2.resize(image, target_size)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast stretching
    processed = contrast_stretch(gray)
    
    # Convert grayscale back to BGR
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    return processed_bgr


################################################################################
######                      Process Images in All Subdirectories         ######
################################################################################

def detect_directory_structure(input_root):
    """
    Detect if the input directory has a flat structure or subdirectories with classes.
    Returns True if subdirectories contain images (class structure), False if flat.
    """
    # Check if there are any subdirectories with images
    has_subdirs_with_images = False
    images_in_root = False
    
    # Check for images directly in root
    root_files = [f for f in os.listdir(input_root) 
                  if os.path.isfile(os.path.join(input_root, f)) 
                  and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
    if root_files:
        images_in_root = True
    
    # Check for subdirectories with images
    for item in os.listdir(input_root):
        item_path = os.path.join(input_root, item)
        if os.path.isdir(item_path):
            subdir_files = [f for f in os.listdir(item_path) 
                           if os.path.isfile(os.path.join(item_path, f))
                           and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
            if subdir_files:
                has_subdirs_with_images = True
                break
    
    return has_subdirs_with_images, images_in_root

def process_directory_structure(input_root, output_root, target_size=(640, 640)):
    """
    Process all images while maintaining directory structure.
    Handles both flat structure and subdirectory (class) structure.
    """
    total_processed = 0
    
    # Detect the directory structure
    has_subdirs, has_root_images = detect_directory_structure(input_root)
    
    if has_subdirs and has_root_images:
        print("Warning: Found images both in root directory and subdirectories.")
        print("Processing all images while maintaining structure.")
    elif has_subdirs:
        print("Detected class-based directory structure (subdirectories with images).")
    elif has_root_images:
        print("Detected flat directory structure (all images in one directory).")
    else:
        print("No image files found in the input directory.")
        return 0
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(input_root):
        # Calculate the relative path from the input root
        rel_path = os.path.relpath(root, input_root)
        
        # Create corresponding output directory
        if rel_path == ".":
            current_output_dir = output_root
        else:
            current_output_dir = os.path.join(output_root, rel_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(current_output_dir, exist_ok=True)
        
        # Process images in current directory
        image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
        
        if image_files:
            if rel_path == ".":
                print(f"\nProcessing root directory: {root}")
            else:
                print(f"\nProcessing class directory: {os.path.basename(root)}")
            print(f"Found {len(image_files)} image(s)")
        
        for filename in image_files:
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: {filename} could not be read.")
                continue
            
            processed_image = preprocess_image(image, target_size)
            
            # Save the processed image in the corresponding output directory
            output_path = os.path.join(current_output_dir, filename)
            cv2.imwrite(output_path, processed_image)
            print(f"  Processed and saved {filename}")
            total_processed += 1
    
    return total_processed

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    target_size = tuple(args.size)
    
    # Validate input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images maintaining directory structure
    print("Starting image preprocessing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    
    total_images = process_directory_structure(input_dir, output_dir, target_size)
    
    print(f"\nAll done! Processed {total_images} images total.")
    
    # Optional: Print directory structure summary
    if total_images > 0:
        print("\nDirectory structure summary:")
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            rel_path = os.path.relpath(root, output_dir)
            if rel_path == ".":
                print(f"{indent}{os.path.basename(output_dir)}/")
            else:
                print(f"{indent}{os.path.basename(root)}/")
            
            # Count image files
            image_count = len([f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))])
            if image_count > 0:
                sub_indent = ' ' * 2 * (level + 1)
                print(f"{sub_indent}({image_count} images)")


### Usage ###
# python ./scripts/preprocess_images.py --input D:/Dropbox/data/dipteraID/imgs/detect/train/images --output D:/Dropbox/data/dipteraID/imgs/detect/train/images/processed
