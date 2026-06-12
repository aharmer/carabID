import os
import cv2
import numpy as np
import random
from pathlib import Path
import argparse

class YOLOAugmenter:
    def __init__(self, images_dir, labels_dir, output_images_dir, output_labels_dir, 
                 enable_flip=True, enable_rotation=True, rotation_range=15,
                 convert_to_bbox=False, multiplication_factor=2.0):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_images_dir = Path(output_images_dir)
        self.output_labels_dir = Path(output_labels_dir)
        
        # Augmentation settings
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation
        self.rotation_range = rotation_range
        self.convert_to_bbox = convert_to_bbox
        self.multiplication_factor = multiplication_factor
        
        # Create output directories if they don't exist
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    def is_bbox_format(self, coords):
        """Determine if annotation is in bbox format (4 values) or polygon format (>4 values)"""
        return len(coords) == 4
    
    def parse_annotation(self, annotation_line):
        """Parse annotation line and return class_id, coords, and format type"""
        parts = annotation_line.strip().split()
        class_id = int(parts[0])
        values = [float(x) for x in parts[1:]]
        
        if len(values) == 4:
            # Bounding box format: class_id x_center y_center width height
            return class_id, np.array(values), 'bbox'
        else:
            # Polygon format: class_id x1 y1 x2 y2 x3 y3 ...
            coords = np.array(values).reshape(-1, 2)
            return class_id, coords, 'polygon'
    
    def parse_polygon_annotation(self, annotation_line):
        """Parse a single polygon annotation line (legacy method for compatibility)"""
        parts = annotation_line.strip().split()
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        # Reshape to coordinate pairs (x, y)
        coords = np.array(coords).reshape(-1, 2)
        return class_id, coords
    
    def bbox_to_polygon(self, bbox_coords):
        """Convert bbox format (x_center, y_center, width, height) to polygon corner coordinates"""
        x_center, y_center, width, height = bbox_coords
        
        # Calculate corners
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2
        
        # Return as polygon corners (4 points: top-left, top-right, bottom-right, bottom-left)
        polygon = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])
        
        return polygon
    
    def polygon_to_bbox(self, coords):
        """Convert polygon coordinates to bounding box format"""
        # Find min/max coordinates
        x_min = np.min(coords[:, 0])
        x_max = np.max(coords[:, 0])
        y_min = np.min(coords[:, 1])
        y_max = np.max(coords[:, 1])
        
        # Calculate center and dimensions (already normalized)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        return x_center, y_center, width, height
    
    def format_bbox_annotation(self, class_id, x_center, y_center, width, height):
        """Format bounding box annotation to string"""
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def format_polygon_annotation(self, class_id, coords):
        """Format polygon annotation to string"""
        coords_flat = coords.flatten()
        coords_str = " ".join([f"{coord:.6f}" for coord in coords_flat])
        return f"{class_id} {coords_str}"
    
    def denormalize_coords(self, coords, img_height, img_width):
        """Convert normalized coordinates to pixel coordinates"""
        coords_pixel = coords.copy()
        coords_pixel[:, 0] *= img_width
        coords_pixel[:, 1] *= img_height
        return coords_pixel
    
    def normalize_coords(self, coords, img_height, img_width):
        """Convert pixel coordinates to normalized coordinates"""
        coords_norm = coords.copy()
        coords_norm[:, 0] /= img_width
        coords_norm[:, 1] /= img_height
        return coords_norm
    
    def flip_horizontal(self, image, coords_list):
        """Flip image horizontally and update polygon coordinates"""
        # Flip image
        flipped_image = cv2.flip(image, 1)
        img_height, img_width = image.shape[:2]
        
        flipped_coords_list = []
        for coords in coords_list:
            # Denormalize coordinates
            coords_pixel = self.denormalize_coords(coords, img_height, img_width)
            
            # Flip x-coordinates
            coords_pixel[:, 0] = img_width - coords_pixel[:, 0]
            
            # Normalize back
            coords_norm = self.normalize_coords(coords_pixel, img_height, img_width)
            flipped_coords_list.append(coords_norm)
        
        return flipped_image, flipped_coords_list
    
    def rotate_image_and_coords(self, image, coords_list, angle):
        """Rotate image and update polygon coordinates"""
        img_height, img_width = image.shape[:2]
        center = (img_width / 2, img_height / 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int((img_height * sin_angle) + (img_width * cos_angle))
        new_height = int((img_height * cos_angle) + (img_width * sin_angle))
        
        # Adjust rotation matrix for translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Rotate image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        
        # Transform coordinates
        rotated_coords_list = []
        for coords in coords_list:
            # Denormalize coordinates
            coords_pixel = self.denormalize_coords(coords, img_height, img_width)
            
            # Add homogeneous coordinate (1) for matrix multiplication
            coords_homogeneous = np.column_stack([coords_pixel, np.ones(coords_pixel.shape[0])])
            
            # Apply rotation transformation
            rotated_coords_pixel = np.dot(rotation_matrix, coords_homogeneous.T).T
            
            # Normalize coordinates with new image dimensions
            coords_norm = self.normalize_coords(rotated_coords_pixel, new_height, new_width)
            
            # Clamp coordinates to [0, 1] range
            coords_norm = np.clip(coords_norm, 0, 1)
            
            rotated_coords_list.append(coords_norm)
        
        return rotated_image, rotated_coords_list
    
    def save_annotations(self, annotations, output_path):
        """Save annotations in the specified format (polygon or bbox)"""
        with open(output_path, 'w') as f:
            for class_id, coords in annotations:
                if self.convert_to_bbox:
                    x_center, y_center, width, height = self.polygon_to_bbox(coords)
                    annotation_line = self.format_bbox_annotation(class_id, x_center, y_center, width, height)
                else:
                    annotation_line = self.format_polygon_annotation(class_id, coords)
                f.write(annotation_line + '\n')
    
    def generate_augmentation_plan(self, num_images):
        """Generate a random augmentation plan based on multiplication factor"""
        # Calculate total number of images needed (including originals)
        total_images_needed = int(num_images * self.multiplication_factor)
        
        # Number of augmentations needed
        augmentations_needed = total_images_needed - num_images
        
        # Create list of available augmentation types
        augmentation_types = []
        if self.enable_flip:
            augmentation_types.append('flip')
        if self.enable_rotation:
            augmentation_types.append('rotation')
        
        if not augmentation_types:
            print("Warning: No augmentation types enabled. Only original images will be saved.")
            return []
        
        # Generate random augmentation plan
        augmentation_plan = []
        for _ in range(augmentations_needed):
            aug_type = random.choice(augmentation_types)
            
            if aug_type == 'flip':
                augmentation_plan.append(('flip', None))
            elif aug_type == 'rotation':
                # Random rotation angle within range
                angle = random.uniform(-self.rotation_range, self.rotation_range)
                augmentation_plan.append(('rotation', angle))
        
        return augmentation_plan
    
    def apply_augmentation(self, image, annotations, aug_type, aug_param):
        """Apply a specific augmentation to image and annotations"""
        if aug_type == 'flip':
            if annotations:
                aug_image, aug_coords = self.flip_horizontal(image, [coords for _, coords in annotations])
                aug_annotations = list(zip([class_id for class_id, _ in annotations], aug_coords))
            else:
                aug_image = cv2.flip(image, 1)
                aug_annotations = []
            return aug_image, aug_annotations, "flip"
        
        elif aug_type == 'rotation':
            angle = aug_param
            if annotations:
                aug_image, aug_coords = self.rotate_image_and_coords(image, [coords for _, coords in annotations], angle)
                aug_annotations = list(zip([class_id for class_id, _ in annotations], aug_coords))
            else:
                aug_image, _ = self.rotate_image_and_coords(image, [], angle)
                aug_annotations = []
            return aug_image, aug_annotations, f"rot_{angle:.1f}deg"
        
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    def process_image(self, image_path, label_path, image_augmentation_plan):
        """Process a single image with its annotations using the provided augmentation plan"""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return
        
        # Read annotations - now handles both bbox and polygon formats
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id, coords, format_type = self.parse_annotation(line)
                        
                        # Convert bbox to polygon for internal processing
                        if format_type == 'bbox':
                            coords = self.bbox_to_polygon(coords)
                        
                        annotations.append((class_id, coords))
        
        # Get base filename without extension
        base_name = image_path.stem
        
        # Save original image and annotations in specified format
        cv2.imwrite(str(self.output_images_dir / image_path.name), image)
        if label_path.exists() and annotations:
            self.save_annotations(annotations, self.output_labels_dir / label_path.name)
        elif label_path.exists():
            # Create empty label file if original exists but has no valid annotations
            open(self.output_labels_dir / label_path.name, 'w').close()
        
        # Apply augmentations according to plan
        for i, (aug_type, aug_param) in enumerate(image_augmentation_plan):
            try:
                aug_image, aug_annotations, suffix = self.apply_augmentation(image, annotations, aug_type, aug_param)
                
                # Save augmented image
                aug_image_name = f"{base_name}_{suffix}_{i+1}{image_path.suffix}"
                cv2.imwrite(str(self.output_images_dir / aug_image_name), aug_image)
                
                # Save augmented annotations
                aug_label_name = f"{base_name}_{suffix}_{i+1}.txt"
                self.save_annotations(aug_annotations, self.output_labels_dir / aug_label_name)
                
            except Exception as e:
                print(f"Warning: Failed to apply {aug_type} augmentation to {image_path.name}: {str(e)}")
        
        format_type = "bbox" if self.convert_to_bbox else "polygon"
        print(f"Processed {image_path.name}: 1 original + {len(image_augmentation_plan)} augmentations ({format_type} format)")
    
    def augment_dataset(self):
        """Augment the entire dataset with random augmentation selection"""
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in self.images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        num_images = len(image_files)
        
        print(f"Found {num_images} images to process")
        print(f"Dataset multiplication factor: {self.multiplication_factor}x")
        print(f"Target dataset size: {int(num_images * self.multiplication_factor)} images")
        print(f"Augmentation settings:")
        print(f"  - Input format: Mixed (bbox and/or polygon annotations)")
        format_output = "Bounding box" if self.convert_to_bbox else "Polygon"
        print(f"  - Output format: {format_output} annotations")
        print(f"  - Horizontal flip: {'Enabled' if self.enable_flip else 'Disabled'}")
        print(f"  - Rotation: {'Enabled' if self.enable_rotation else 'Disabled'}")
        if self.enable_rotation:
            print(f"    - Rotation range: ±{self.rotation_range}°")
        print()
        
        # Generate augmentation plan
        augmentation_plan = self.generate_augmentation_plan(num_images)
        
        # Distribute augmentations across images
        augmentations_per_image = len(augmentation_plan) // num_images
        remaining_augmentations = len(augmentation_plan) % num_images
        
        print(f"Augmentation distribution:")
        print(f"  - Base augmentations per image: {augmentations_per_image}")
        print(f"  - Images with extra augmentation: {remaining_augmentations}")
        print(f"  - Total augmentations: {len(augmentation_plan)}")
        
        # Shuffle augmentation plan for random distribution
        random.shuffle(augmentation_plan)
        
        augmentation_index = 0
        for i, image_path in enumerate(image_files):
            # Find corresponding label file
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            
            # Determine how many augmentations this image gets
            num_augs_for_image = augmentations_per_image
            if i < remaining_augmentations:
                num_augs_for_image += 1
            
            # Get augmentations for this image
            image_augmentation_plan = augmentation_plan[augmentation_index:augmentation_index + num_augs_for_image]
            augmentation_index += num_augs_for_image
            
            try:
                self.process_image(image_path, label_path, image_augmentation_plan)
            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")
        
        print(f"\nDataset augmentation completed!")
        print(f"Output images saved to: {self.output_images_dir}")
        print(f"Output labels saved to: {self.output_labels_dir}")
        format_msg = "converted to bounding box format" if self.convert_to_bbox else "kept in original format"
        print(f"All annotations {format_msg}")
        
        # Count final results
        final_images = len(list(self.output_images_dir.glob('*')))
        final_labels = len(list(self.output_labels_dir.glob('*.txt')))
        print(f"Final dataset size: {final_images} images, {final_labels} label files")
        actual_multiplication = final_images / num_images if num_images > 0 else 0
        print(f"Actual multiplication factor: {actual_multiplication:.2f}x")

def main():
    parser = argparse.ArgumentParser(description='Augment YOLO dataset with mixed bbox/polygon annotations')
    parser.add_argument('--images-dir', required=True, help='Directory containing original images')
    parser.add_argument('--labels-dir', required=True, help='Directory containing original label files')
    parser.add_argument('--output-images-dir', required=True, help='Directory to save augmented images')
    parser.add_argument('--output-labels-dir', required=True, help='Directory to save augmented labels')
    
    # Augmentation control arguments
    parser.add_argument('--no-flip', action='store_true', help='Disable horizontal flipping')
    parser.add_argument('--no-rotation', action='store_true', help='Disable rotation augmentation')
    parser.add_argument('--rotation-range', type=float, default=15.0, help='Maximum rotation angle in degrees (default: 15.0)')
    
    # New multiplication factor argument
    parser.add_argument('--multiplication-factor', type=float, default=2.0, 
                       help='Dataset size multiplication factor (default: 2.0, e.g., 2.0 = double the dataset size)')
    
    # Format conversion argument
    parser.add_argument('--convert-to-bbox', action='store_true', 
                       help='Convert all annotations to bounding box format (default: keep original format)')
    
    args = parser.parse_args()
    
    # Validation
    if args.multiplication_factor < 1.0:
        print("Error: multiplication-factor must be >= 1.0")
        return
    
    if not args.no_flip and not args.no_rotation:
        pass  # At least one augmentation type is enabled
    elif args.no_flip and args.no_rotation:
        if args.multiplication_factor > 1.0:
            print("Warning: No augmentation types enabled, but multiplication factor > 1.0. Only original images will be saved.")
    
    # Create augmenter instance
    augmenter = YOLOAugmenter(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_images_dir=args.output_images_dir,
        output_labels_dir=args.output_labels_dir,
        enable_flip=not args.no_flip,
        enable_rotation=not args.no_rotation,
        rotation_range=args.rotation_range,
        convert_to_bbox=args.convert_to_bbox,
        multiplication_factor=args.multiplication_factor
    )
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Perform augmentation
    augmenter.augment_dataset()

if __name__ == "__main__":
    main()

# Example usage:
# Double dataset size with random mix of flips and rotations:
# python augment_detection_dataset.py --images-dir ./original/images --labels-dir ./original/labels --output-images-dir ./augmented/images --output-labels-dir ./augmented/labels --multiplication-factor 2.0

# Triple dataset size with only rotations:
# python augment_detection_dataset.py --images-dir ./original/images --labels-dir ./original/labels --output-images-dir ./augmented/images --output-labels-dir ./augmented/labels --no-flip --multiplication-factor 3.0 --rotation-range 30

# Convert mixed format dataset to bbox format:
# python ./scripts/augment_detection_dataset.py --images-dir ./imgs/combined_processed/train/images --labels-dir ./imgs/combined_processed/train/labels --output-images-dir ./imgs/combined_augmented/train/images --output-labels-dir ./imgs/combined_augmented/train/labels --multiplication-factor 1.0 --no-flip --no-rotation --convert-to-bbox
