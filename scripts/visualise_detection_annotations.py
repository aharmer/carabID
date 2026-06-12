import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import argparse
from matplotlib.patches import Rectangle, Polygon

def detect_annotation_type(label_path):
    """
    Detect whether annotations are bounding boxes or polygons.
    
    Args:
        label_path (str): Path to the label file
        
    Returns:
        str: 'bbox' for bounding boxes, 'polygon' for polygons, 'unknown' if cannot determine
    """
    if not os.path.exists(label_path):
        return 'unknown'
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 2:
                continue
            
            # Skip class ID, count coordinate values
            coord_count = len(parts) - 1
            
            if coord_count == 4:
                return 'bbox'  # class + 4 values = bounding box
            elif coord_count >= 6 and coord_count % 2 == 0:
                return 'polygon'  # class + even number of coordinates >= 6 = polygon
        
        return 'unknown'
    
    except Exception as e:
        print(f"Error detecting annotation type in {label_path}: {e}")
        return 'unknown'

def read_bbox_annotations(label_path):
    """
    Read YOLO bounding box annotations from a text file.
    
    Args:
        label_path (str): Path to the label file
        
    Returns:
        list: List of annotations, each containing [class, x_center, y_center, width, height]
    """
    annotations = []
    
    if not os.path.exists(label_path):
        return annotations
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 5:  # class + 4 bbox values
                # Convert to float, keeping class as int
                annotation = [int(parts[0])] + [float(x) for x in parts[1:5]]
                annotations.append(annotation)
    
    except Exception as e:
        print(f"Error reading bbox annotations from {label_path}: {e}")
    
    return annotations

def read_polygon_annotations(label_path):
    """
    Read YOLO polygon annotations from a text file.
    
    Args:
        label_path (str): Path to the label file
        
    Returns:
        list: List of annotations, each containing [class, [x1, y1, x2, y2, ...]]
    """
    annotations = []
    
    if not os.path.exists(label_path):
        return annotations
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 7:  # class + at least 6 coordinates (3 points minimum)
                class_id = int(parts[0])
                coordinates = [float(x) for x in parts[1:]]
                
                # Ensure even number of coordinates
                if len(coordinates) % 2 == 0:
                    annotations.append([class_id, coordinates])
    
    except Exception as e:
        print(f"Error reading polygon annotations from {label_path}: {e}")
    
    return annotations

def yolo_to_corners(x_center, y_center, width, height, img_width, img_height):
    """
    Convert YOLO format (center + width/height) to corner coordinates.
    
    Args:
        x_center (float): Normalized x center coordinate
        y_center (float): Normalized y center coordinate  
        width (float): Normalized width
        height (float): Normalized height
        img_width (int): Image width in pixels
        img_height (int): Image height in pixels
        
    Returns:
        tuple: (x_min, y_min, x_max, y_max) in pixel coordinates
    """
    # Convert normalized coordinates to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Calculate corner coordinates
    x_min = int(x_center_px - width_px / 2)
    y_min = int(y_center_px - height_px / 2)
    x_max = int(x_center_px + width_px / 2)
    y_max = int(y_center_px + height_px / 2)
    
    return x_min, y_min, x_max, y_max

def yolo_polygon_to_pixels(coordinates, img_width, img_height):
    """
    Convert YOLO normalized polygon coordinates to pixel coordinates.
    
    Args:
        coordinates (list): List of normalized coordinates [x1, y1, x2, y2, ...]
        img_width (int): Image width in pixels
        img_height (int): Image height in pixels
        
    Returns:
        list: List of pixel coordinates [(x1, y1), (x2, y2), ...]
    """
    pixel_coords = []
    for i in range(0, len(coordinates), 2):
        x_norm = coordinates[i]
        y_norm = coordinates[i + 1]
        x_pixel = int(x_norm * img_width)
        y_pixel = int(y_norm * img_height)
        pixel_coords.append((x_pixel, y_pixel))
    
    return pixel_coords

def plot_image_with_annotations(image_path, label_path, class_names=None, figsize=(12, 8)):
    """
    Plot an image with its annotations (bounding boxes or polygons).
    
    Args:
        image_path (str): Path to the image file
        label_path (str): Path to the corresponding label file
        class_names (dict): Dictionary mapping class IDs to names
        figsize (tuple): Figure size for matplotlib
    """
    # Read image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Read image using OpenCV (BGR) and convert to RGB for matplotlib
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Failed to load image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_height, img_width = img_rgb.shape[:2]
    
    # Detect annotation type
    annotation_type = detect_annotation_type(label_path)
    
    if annotation_type == 'unknown':
        print(f"Could not determine annotation type for {label_path}")
        return
    
    # Read annotations based on type
    if annotation_type == 'bbox':
        annotations = read_bbox_annotations(label_path)
        print(f"Detected bounding box annotations: {len(annotations)} boxes")
    else:  # polygon
        annotations = read_polygon_annotations(label_path)
        print(f"Detected polygon annotations: {len(annotations)} polygons")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_rgb)
    ax.set_title(f"Image: {os.path.basename(image_path)} ({annotation_type.upper()})")
    
    # Define colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 
              'brown', 'pink', 'gray', 'olive', 'navy', 'lime', 'teal', 'silver']
    
    # Draw annotations
    for i, annotation in enumerate(annotations):
        class_id = annotation[0]
        color = colors[class_id % len(colors)]
        
        if annotation_type == 'bbox':
            # Handle bounding box
            x_center, y_center, width, height = annotation[1:5]
            
            # Convert YOLO format to corner coordinates
            x_min, y_min, x_max, y_max = yolo_to_corners(
                x_center, y_center, width, height, img_width, img_height
            )
            
            # Create rectangle patch
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=2, edgecolor=color, facecolor=color, alpha=0.2)
            ax.add_patch(rect)
            
            # Draw rectangle outline
            ax.plot([x_min, x_max, x_max, x_min, x_min], 
                   [y_min, y_min, y_max, y_max, y_min], 
                   color=color, linewidth=2)
            
            # Label position
            label_x = x_min
            label_y = y_min - 5 if y_min > 20 else y_max + 15
            
        else:  # polygon
            # Handle polygon
            coordinates = annotation[1]
            pixel_coords = yolo_polygon_to_pixels(coordinates, img_width, img_height)
            
            # Create polygon patch
            polygon = Polygon(pixel_coords, linewidth=2, edgecolor=color, 
                            facecolor=color, alpha=0.2)
            ax.add_patch(polygon)
            
            # Draw polygon outline
            x_coords = [coord[0] for coord in pixel_coords] + [pixel_coords[0][0]]
            y_coords = [coord[1] for coord in pixel_coords] + [pixel_coords[0][1]]
            ax.plot(x_coords, y_coords, color=color, linewidth=2)
            
            # Label position (use centroid of polygon)
            centroid_x = sum(coord[0] for coord in pixel_coords) / len(pixel_coords)
            centroid_y = sum(coord[1] for coord in pixel_coords) / len(pixel_coords)
            label_x = centroid_x
            label_y = centroid_y - 10
        
        # Add class label
        if class_names and class_id in class_names:
            label_text = f"{class_names[class_id]} ({class_id})"
        else:
            label_text = f"Class {class_id}"
        
        ax.text(label_x, label_y, label_text, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                fontsize=10, ha='center', va='center', color='white', weight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print annotation details
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Dimensions: {img_width} x {img_height}")
    print(f"Annotation type: {annotation_type.upper()}")
    print(f"Total annotations: {len(annotations)}")
    
    if annotations:
        print("Classes found:", [ann[0] for ann in annotations])
        for i, ann in enumerate(annotations):
            class_id = ann[0]
            if annotation_type == 'bbox':
                x_center, y_center, width, height = ann[1:5]
                x_min, y_min, x_max, y_max = yolo_to_corners(x_center, y_center, width, height, img_width, img_height)
                print(f"  Box {i+1}: Class {class_id}, Center({x_center:.3f}, {y_center:.3f}), "
                      f"Size({width:.3f}, {height:.3f}), Pixels({x_min}, {y_min}, {x_max}, {y_max})")
            else:  # polygon
                coordinates = ann[1]
                num_points = len(coordinates) // 2
                print(f"  Polygon {i+1}: Class {class_id}, Points: {num_points}")

def visualize_dataset_samples(dataset_path, num_samples=5, class_names=None):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset_path (str): Path to dataset directory containing 'images' and 'labels' folders
        num_samples (int): Number of random samples to visualize
        class_names (dict): Dictionary mapping class IDs to names
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        return
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
        image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images in dataset")
    
    # Select random samples
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    for image_file in sample_files:
        # Find corresponding label file
        label_file = labels_dir / (image_file.stem + '.txt')
        
        print(f"\n{'='*50}")
        plot_image_with_annotations(str(image_file), str(label_file), class_names)

def visualize_single_image(image_path, label_path, class_names=None):
    """
    Visualize a single image with its annotations.
    
    Args:
        image_path (str): Path to the image file
        label_path (str): Path to the label file
        class_names (dict): Dictionary mapping class IDs to names
    """
    print(f"Visualizing: {image_path}")
    plot_image_with_annotations(image_path, label_path, class_names)

def analyze_dataset_statistics(dataset_path, class_names=None):
    """
    Analyze and display dataset statistics.
    
    Args:
        dataset_path (str): Path to dataset directory containing 'images' and 'labels' folders
        class_names (dict): Dictionary mapping class IDs to names
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        print("Dataset directories not found!")
        return
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
        image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
    
    print(f"Dataset Statistics")
    print("=" * 30)
    print(f"Total images: {len(image_files)}")
    
    # Analyze annotations
    class_counts = {}
    total_annotations = 0
    images_with_labels = 0
    annotation_types = {'bbox': 0, 'polygon': 0, 'unknown': 0}
    
    for image_file in image_files:
        label_file = labels_dir / (image_file.stem + '.txt')
        
        # Detect annotation type
        ann_type = detect_annotation_type(str(label_file))
        annotation_types[ann_type] += 1
        
        if ann_type == 'bbox':
            annotations = read_bbox_annotations(str(label_file))
        elif ann_type == 'polygon':
            annotations = read_polygon_annotations(str(label_file))
        else:
            annotations = []
        
        if annotations:
            images_with_labels += 1
            total_annotations += len(annotations)
            
            for ann in annotations:
                class_id = ann[0]
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                class_counts[class_id] += 1
    
    print(f"Images with labels: {images_with_labels}")
    print(f"Images without labels: {len(image_files) - images_with_labels}")
    print(f"Total annotations: {total_annotations}")
    print(f"Average annotations per image: {total_annotations / len(image_files):.2f}")
    
    print(f"\nAnnotation types:")
    for ann_type, count in annotation_types.items():
        if count > 0:
            print(f"  {ann_type.upper()}: {count} images")
    
    print(f"\nClass distribution:")
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"
        percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
        print(f"  {class_name}: {count} annotations ({percentage:.1f}%)")

def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(description='YOLO Annotation Visualizer - Supports both bounding boxes and polygons')
    parser.add_argument('--mode', choices=['dataset', 'single', 'stats'], default='dataset',
                       help='Visualization mode: dataset (random samples), single (specific image), or stats (dataset statistics)')
    parser.add_argument('--dataset-path', type=str, required=False,
                       help='Path to dataset directory containing images and labels folders')
    parser.add_argument('--image-path', type=str, required=False,
                       help='Path to specific image file (for single mode)')
    parser.add_argument('--label-path', type=str, required=False,
                       help='Path to specific label file (for single mode)')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of random samples to visualize (default: 5)')
    parser.add_argument('--class-names', type=str, nargs='+', required=False,
                       help='Class names in order (e.g., --class-names person car truck)')
    
    args = parser.parse_args()
    
    # Create class names dictionary if provided
    class_names = None
    if args.class_names:
        class_names = {i: name for i, name in enumerate(args.class_names)}
    
    print("YOLO Annotation Visualizer (BBox + Polygon Support)")
    print("=" * 50)
    
    if args.mode == 'dataset':
        if not args.dataset_path:
            print("Error: --dataset-path is required for dataset mode")
            return
        print(f"Visualizing {args.num_samples} random samples from: {args.dataset_path}")
        visualize_dataset_samples(args.dataset_path, args.num_samples, class_names)
        
    elif args.mode == 'single':
        if not args.image_path or not args.label_path:
            print("Error: --image-path and --label-path are required for single mode")
            return
        print(f"Visualizing single image: {args.image_path}")
        visualize_single_image(args.image_path, args.label_path, class_names)
        
    elif args.mode == 'stats':
        if not args.dataset_path:
            print("Error: --dataset-path is required for stats mode")
            return
        print(f"Analyzing dataset statistics: {args.dataset_path}")
        analyze_dataset_statistics(args.dataset_path, class_names)

if __name__ == "__main__":
    main()

# Usage examples:
"""
# Visualize 10 random samples from dataset
python ./scripts/visualise_detection_annotations.py --mode dataset --dataset-path D:/Dropbox/data/carabID/imgs/combined_augmented/train --num-samples 10

# Visualize specific image
python visualise_detection_annotations.py --mode single --image-path /path/to/image.jpg --label-path /path/to/label.txt

# Show dataset statistics
python visualise_detection_annotations.py --mode stats --dataset-path /path/to/dataset

"""
