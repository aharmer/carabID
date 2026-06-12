import os
import yaml
import shutil
from pathlib import Path
from PIL import Image
import argparse
import numpy as np


def load_class_names(data_yaml_path):
    """Load class names from data.yaml file."""
    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    names = data['names']
    
    # Handle both list and dictionary formats
    if isinstance(names, list):
        # Convert list to dictionary with index as key
        return {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        # Already in dictionary format
        return names
    else:
        raise ValueError(f"Unsupported names format in data.yaml: {type(names)}")


def polygon_to_bbox(polygon_coords):
    """Convert polygon coordinates to bounding box coordinates.
    
    Args:
        polygon_coords: List of normalized coordinates [x1, y1, x2, y2, ..., xn, yn]
    
    Returns:
        tuple: (x_center, y_center, width, height) in normalized coordinates
    """
    # Reshape coordinates into (n, 2) array of (x, y) points
    points = np.array(polygon_coords).reshape(-1, 2)
    
    # Find bounding box
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    
    # Convert to center format
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height


def detect_annotation_format(annotation_lines):
    """Detect whether annotations are in bounding box or polygon format.
    
    Args:
        annotation_lines: List of annotation lines
    
    Returns:
        str: 'bbox' or 'polygon'
    """
    if not annotation_lines:
        return 'bbox'  # Default assumption
    
    # Check the first non-empty line
    for line in annotation_lines:
        line = line.strip()
        if line:
            parts = line.split()
            # Bounding box format: class_id x_center y_center width height (5 values)
            # Polygon format: class_id x1 y1 x2 y2 ... xn yn (>5 values, even number after class_id)
            if len(parts) == 5:
                return 'bbox'
            elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                return 'polygon'
            else:
                # Ambiguous, assume bbox
                return 'bbox'
    
    return 'bbox'


def parse_yolo_annotation(annotation_path):
    """Parse YOLO format annotation file and return list of bounding boxes.
    
    Handles both bounding box and polygon formats.
    """
    boxes = []
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        
        # Detect annotation format
        annotation_format = detect_annotation_format(lines)
        print(f"Detected annotation format: {annotation_format} for {os.path.basename(annotation_path)}")
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                class_id = int(parts[0])
                
                if annotation_format == 'bbox':
                    # Standard bounding box format
                    if len(parts) >= 5:
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        boxes.append((class_id, x_center, y_center, width, height))
                    else:
                        print(f"Warning: Invalid bounding box format in line: {line}")
                        
                elif annotation_format == 'polygon':
                    # Polygon format - convert to bounding box
                    if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:  # At least 3 points (6 coords) + class_id
                        polygon_coords = [float(x) for x in parts[1:]]
                        x_center, y_center, width, height = polygon_to_bbox(polygon_coords)
                        boxes.append((class_id, x_center, y_center, width, height))
                    else:
                        print(f"Warning: Invalid polygon format in line: {line}")
                        
    return boxes


def yolo_to_pixel_coords(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO normalized coordinates to pixel coordinates."""
    # Convert to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Calculate bounding box corners
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    return x1, y1, x2, y2


def resize_image(image, target_size, resample_method=Image.LANCZOS):
    """Resize image to target size by stretching (no aspect ratio preservation).
    
    Args:
        image: PIL Image object
        target_size: tuple (width, height) for target dimensions
        resample_method: PIL resampling method for resizing
    
    Returns:
        PIL Image: Resized image
    """
    return image.resize(target_size, resample_method)


def create_classification_structure(output_dir, class_names):
    """Create directory structure for classification dataset with single train directory."""
    # Create only train directory structure
    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    
    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)


def process_dataset(input_dir, output_dir, data_yaml_path, min_crop_size=32, resize_to=None):
    """Convert detection dataset to classification dataset."""
    # Load class names
    class_names = load_class_names(data_yaml_path)
    print(f"Loaded {len(class_names)} classes: {list(class_names.values())}")
    
    if resize_to:
        print(f"Images will be resized to {resize_to[0]}x{resize_to[1]} by stretching")
    
    # Define splits to process from input
    splits = ['train', 'valid', 'test']
    
    # Create output directory structure (only train directory)
    create_classification_structure(output_dir, class_names.values())
    
    # Output directory is always train
    output_train_dir = os.path.join(output_dir, 'train')
    
    total_crop_count = 0
    
    # Process each split but put everything in train directory
    for split in splits:
        split_input_dir = os.path.join(input_dir, split)
        
        if not os.path.exists(split_input_dir):
            print(f"Warning: {split_input_dir} does not exist, skipping...")
            continue
            
        images_dir = os.path.join(split_input_dir, 'images')
        labels_dir = os.path.join(split_input_dir, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"Warning: {images_dir} does not exist, skipping...")
            continue
            
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(images_dir).glob(f'*{ext}'))
            # image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
        
        print(f"Processing {len(image_files)} images from {split} split...")
        
        split_crop_count = 0
        for img_path in image_files:
            # Get corresponding annotation file
            annotation_path = os.path.join(labels_dir, img_path.stem + '.txt')
            
            # Parse annotations
            boxes = parse_yolo_annotation(annotation_path)
            
            if not boxes:
                print(f"No annotations found for {img_path.name}, skipping...")
                continue
            
            # Load image
            try:
                image = Image.open(img_path)
                img_width, img_height = image.size
            except Exception as e:
                print(f"Error loading image {img_path.name}: {e}")
                continue
            
            # Process each bounding box
            for i, (class_id, x_center, y_center, width, height) in enumerate(boxes):
                if class_id not in class_names:
                    print(f"Warning: Class ID {class_id} not found in class names, skipping...")
                    continue
                
                # Convert to pixel coordinates
                x1, y1, x2, y2 = yolo_to_pixel_coords(
                    x_center, y_center, width, height, img_width, img_height
                )
                
                # Check if crop is large enough
                crop_width = x2 - x1
                crop_height = y2 - y1
                
                if crop_width < min_crop_size or crop_height < min_crop_size:
                    print(f"Crop too small ({crop_width}x{crop_height}) for {img_path.name}, skipping...")
                    continue
                
                # Crop image
                try:
                    cropped_image = image.crop((x1, y1, x2, y2))
                    
                    # Resize image if specified
                    if resize_to:
                        cropped_image = resize_image(cropped_image, resize_to)
                    
                    # Generate output filename with split prefix to avoid name conflicts
                    class_name = class_names[class_id]
                    output_filename = f"{split}_{img_path.stem}_{i}.{img_path.suffix[1:]}"
                    output_path = os.path.join(output_train_dir, class_name, output_filename)
                    
                    # Save cropped (and potentially resized) image
                    cropped_image.save(output_path)
                    split_crop_count += 1
                    total_crop_count += 1
                    
                except Exception as e:
                    print(f"Error cropping/resizing image {img_path.name}: {e}")
                    continue
        
        print(f"Completed {split} split: {split_crop_count} crops saved")
    
    print(f"Dataset conversion completed! Total crops saved: {total_crop_count}")
    print(f"All images saved to: {output_train_dir}")
    if resize_to:
        print(f"All images resized to: {resize_to[0]}x{resize_to[1]}")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO detection dataset to classification dataset (single train directory)')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing train/valid/test folders')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for classification dataset')
    parser.add_argument('--data_yaml', type=str, required=True,
                       help='Path to data.yaml file containing class names')
    parser.add_argument('--min_crop_size', type=int, default=32,
                       help='Minimum size for cropped images (default: 32)')
    parser.add_argument('--resize_to', type=int, nargs=2, default=None,
                       help='Resize images to specified dimensions (width height), e.g., --resize_to 640 640')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Validate data.yaml file
    if not os.path.exists(args.data_yaml):
        print(f"Error: data.yaml file {args.data_yaml} does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert resize_to to tuple if provided
    resize_to = tuple(args.resize_to) if args.resize_to else None
    
    # Process dataset
    process_dataset(args.input_dir, args.output_dir, args.data_yaml, args.min_crop_size, resize_to)


if __name__ == "__main__":
    main()


# Usage examples:
# Without resizing (original functionality):
# python convert_detection_to_classification.py --input_dir D:/Dropbox/data/tephritID/imgs/detection_set --output_dir D:/Dropbox/data/tephritID/imgs/classification_set --data_yaml D:/Dropbox/data/tephritID/imgs/detection_set/data.yaml --min_crop_size 32

# With resizing to 640x640:
# python ./scripts/convert_detection_to_classification.py --input_dir D:/Dropbox/data/carabID/imgs/detection_set --output_dir D:/Dropbox/data/carabID/imgs/c1 --data_yaml D:/Dropbox/data/carabID/imgs/detection_set/data.yaml --resize_to 640 640


