import os
import cv2
import numpy as np
import random
from pathlib import Path
import argparse

class YOLOClassificationAugmenter:
    def __init__(self, input_dir, output_dir, multiplication_factor=2.0,
                 enable_flip=True, enable_rotation=True, rotation_range=15,
                 enable_blur=True, enable_noise=True,
                 rotation_method='crop_inscribed'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.multiplication_factor = multiplication_factor
        
        # Augmentation settings
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation
        self.rotation_range = rotation_range
        self.enable_blur = enable_blur
        self.enable_noise = enable_noise
        self.rotation_method = rotation_method
        
        # Create list of available augmentation types
        self.augmentation_types = []
        if self.enable_flip:
            self.augmentation_types.append('flip')
        if self.enable_rotation:
            self.augmentation_types.append('rotation')
        if self.enable_blur:
            self.augmentation_types.append('motion_blur')
        if self.enable_noise:
            self.augmentation_types.extend(['gaussian_noise', 'uniform_noise'])
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def flip_horizontal(self, image):
        """Flip image horizontally"""
        return cv2.flip(image, 1)
    
    def rotate_image(self, image, angle, method='crop_inscribed'):
        """Rotate image by specified angle without black edges"""
        img_height, img_width = image.shape[:2]
        center = (img_width / 2, img_height / 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        if method == 'crop_inscribed':
            # Method 1: Crop to largest inscribed rectangle (maintains aspect ratio)
            angle_rad = np.radians(abs(angle))
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # Calculate inscribed rectangle dimensions
            inscribed_width = int((img_width * cos_a + img_height * sin_a) * cos_a)
            inscribed_height = int((img_height * cos_a + img_width * sin_a) * cos_a)
            
            # Rotate with original dimensions
            rotated_image = cv2.warpAffine(image, rotation_matrix, (img_width, img_height))
            
            # Crop to inscribed rectangle
            x_offset = (img_width - inscribed_width) // 2
            y_offset = (img_height - inscribed_height) // 2
            
            # Ensure we don't go out of bounds
            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)
            x_end = min(img_width, x_offset + inscribed_width)
            y_end = min(img_height, y_offset + inscribed_height)
            
            cropped_image = rotated_image[y_offset:y_end, x_offset:x_end]
            return cropped_image
            
        elif method == 'crop_tight':
            # Method 2: Expand canvas, rotate, then crop tightly
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_width = int((img_height * sin_angle) + (img_width * cos_angle))
            new_height = int((img_height * cos_angle) + (img_width * sin_angle))
            
            # Adjust rotation matrix for translation
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Rotate with expanded dimensions
            rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
            
            # Find tight crop boundaries by detecting non-black pixels
            gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find bounding box of largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cropped_image = rotated_image[y:y+h, x:x+w]
            else:
                cropped_image = rotated_image
                
            return cropped_image
            
        elif method == 'zoom_fill':
            # Method 3: Scale up rotated image to fill original dimensions
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            
            # Calculate scale factor to eliminate black edges
            scale_factor = max(
                img_width / (img_width * cos_angle + img_height * sin_angle),
                img_height / (img_height * cos_angle + img_width * sin_angle)
            )
            
            # Apply scaling to rotation matrix
            rotation_matrix[0, 0] *= scale_factor
            rotation_matrix[0, 1] *= scale_factor
            rotation_matrix[1, 0] *= scale_factor
            rotation_matrix[1, 1] *= scale_factor
            
            # Rotate and scale
            rotated_image = cv2.warpAffine(image, rotation_matrix, (img_width, img_height))
            return rotated_image
            
        elif method == 'edge_extend':
            # Method 4: Extend edge pixels (better than reflection)
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_width = int((img_height * sin_angle) + (img_width * cos_angle))
            new_height = int((img_height * cos_angle) + (img_width * sin_angle))
            
            # Adjust rotation matrix for translation
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Rotate with edge replication
            rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                         borderMode=cv2.BORDER_REPLICATE)
            
            # Crop to remove most of the extended edges
            margin = min(new_width, new_height) // 20  # Remove 5% margin
            cropped_image = rotated_image[margin:new_height-margin, margin:new_width-margin]
            
            return cropped_image
        
        else:
            raise ValueError(f"Unknown rotation method: {method}")
    
    def add_gaussian_blur(self, image, kernel_size=None, sigma=None):
        """Add Gaussian blur to image"""
        if kernel_size is None:
            kernel_size = random.choice([3, 5, 7, 9])
        if sigma is None:
            sigma = random.uniform(0.5, 2.0)
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred
    
    def add_motion_blur(self, image, kernel_size=None, direction=None):
        """Add motion blur to image"""
        if kernel_size is None:
            kernel_size = random.choice([5, 7, 9, 11])
        if direction is None:
            direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        
        if direction == 'horizontal':
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        elif direction == 'vertical':
            kernel[:, int((kernel_size-1)/2)] = np.ones(kernel_size)
        else:  # diagonal
            np.fill_diagonal(kernel, 1)
        
        kernel = kernel / kernel_size
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
    
    def add_gaussian_noise(self, image, mean=0, std=None):
        """Add Gaussian noise to image"""
        if std is None:
            std = random.uniform(5, 25)
        
        noise = np.random.normal(mean, std, image.shape).astype(np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy_image
    
    def add_salt_pepper_noise(self, image, salt_prob=None, pepper_prob=None):
        """Add salt and pepper noise to image"""
        if salt_prob is None:
            salt_prob = random.uniform(0.001, 0.01)
        if pepper_prob is None:
            pepper_prob = random.uniform(0.001, 0.01)
        
        noisy_image = image.copy()
        
        # Generate random noise mask
        noise = np.random.random(image.shape[:2])
        
        # Add salt noise (white pixels)
        noisy_image[noise < salt_prob] = 255
        
        # Add pepper noise (black pixels)
        noisy_image[noise > 1 - pepper_prob] = 0
        
        return noisy_image
    
    def add_uniform_noise(self, image, noise_range=None):
        """Add uniform noise to image"""
        if noise_range is None:
            noise_range = random.uniform(10, 30)
        
        noise = np.random.uniform(-noise_range, noise_range, image.shape).astype(np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy_image
    
    def apply_random_augmentation(self, image, augmentation_type):
        """Apply a specific type of augmentation to an image"""
        if augmentation_type == 'flip':
            return self.flip_horizontal(image), "flip"
        
        elif augmentation_type == 'rotation':
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            rotated = self.rotate_image(image, angle, self.rotation_method)
            return rotated, f"rot_{angle:.1f}deg"
        
        elif augmentation_type == 'motion_blur':
            blurred = self.add_motion_blur(image)
            return blurred, "mblur"
        
        elif augmentation_type == 'gaussian_noise':
            noisy = self.add_gaussian_noise(image)
            return noisy, "gnoise"
        
        elif augmentation_type == 'uniform_noise':
            noisy = self.add_uniform_noise(image)
            return noisy, "unoise"
        
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    def process_image(self, image_path, class_output_dir, target_augmentations):
        """Process a single image with random augmentations"""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return 0
        
        # Get base filename without extension
        base_name = image_path.stem
        extension = image_path.suffix
        
        # Save original image
        original_output_path = class_output_dir / image_path.name
        cv2.imwrite(str(original_output_path), image)
        images_created = 1
        
        # Generate random augmentations
        if target_augmentations > 0 and self.augmentation_types:
            # Randomly select augmentation types for this image
            selected_augmentations = random.choices(
                self.augmentation_types, 
                k=min(target_augmentations, len(self.augmentation_types) * 2)  # Allow some repetition
            )
            
            # Apply selected augmentations
            for i, aug_type in enumerate(selected_augmentations):
                try:
                    augmented_image, suffix = self.apply_random_augmentation(image, aug_type)
                    
                    # Create unique filename
                    aug_image_name = f"{base_name}_{suffix}_{i+1}{extension}"
                    aug_output_path = class_output_dir / aug_image_name
                    
                    cv2.imwrite(str(aug_output_path), augmented_image)
                    images_created += 1
                    
                    # Stop if we've reached the target
                    if images_created >= target_augmentations + 1:  # +1 for original
                        break
                        
                except Exception as e:
                    print(f"Warning: Failed to apply {aug_type} augmentation to {image_path.name}: {str(e)}")
        
        return images_created
    
    def process_class_directory(self, class_dir):
        """Process all images in a class directory"""
        class_name = class_dir.name
        class_output_dir = self.output_dir / class_name
        
        # Create output class directory
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in class_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in class directory: {class_name}")
            return 0, 0
        
        original_count = len(image_files)
        total_target_images = int(original_count * self.multiplication_factor)
        target_augmentations_per_image = max(0, int((total_target_images - original_count) / original_count))
        
        print(f"Processing class '{class_name}': {original_count} images")
        print(f"  Target total images: {total_target_images}")
        print(f"  Target augmentations per image: {target_augmentations_per_image}")
        
        total_images = 0
        
        for image_path in image_files:
            try:
                images_created = self.process_image(image_path, class_output_dir, target_augmentations_per_image)
                total_images += images_created
            except Exception as e:
                print(f"Error processing {image_path.name} in class {class_name}: {str(e)}")
        
        print(f"  Created {total_images} total images for class '{class_name}' (factor: {total_images/original_count:.1f}x)")
        return original_count, total_images
    
    def augment_dataset(self):
        """Augment the entire classification dataset"""
        if not self.input_dir.exists():
            print(f"Error: Input directory {self.input_dir} does not exist")
            return
        
        # Get all class directories
        class_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            print(f"Error: No class directories found in {self.input_dir}")
            return
        
        if not self.augmentation_types:
            print("Warning: No augmentation types enabled. Only copying original images.")
        
        print(f"Found {len(class_dirs)} class directories")
        print(f"Target multiplication factor: {self.multiplication_factor}x")
        print(f"Augmentation settings:")
        print(f"  - Available augmentation types: {', '.join(self.augmentation_types) if self.augmentation_types else 'None'}")
        print(f"  - Rotation method: {self.rotation_method}")
        if self.enable_rotation:
            print(f"  - Rotation range: ±{self.rotation_range}°")
        print()
        
        total_original_images = 0
        total_augmented_images = 0
        
        for class_dir in sorted(class_dirs):
            original_count, augmented_count = self.process_class_directory(class_dir)
            total_original_images += original_count
            total_augmented_images += augmented_count
        
        print(f"\nDataset augmentation completed!")
        print(f"Original images: {total_original_images}")
        print(f"Total images after augmentation: {total_augmented_images}")
        print(f"Actual multiplication factor: {total_augmented_images / total_original_images:.1f}x")
        print(f"Target was: {self.multiplication_factor}x")
        print(f"Output saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Augment YOLO classification dataset with random augmentations based on multiplication factor')
    parser.add_argument('--input-dir', required=True, help='Directory containing class subdirectories with images')
    parser.add_argument('--output-dir', required=True, help='Directory to save augmented dataset')
    
    # New multiplication factor argument
    parser.add_argument('--multiplication-factor', type=float, default=2.0, 
                       help='Target multiplication factor for dataset size (default: 2.0)')
    
    # Augmentation type control arguments
    parser.add_argument('--no-flip', action='store_true', help='Disable horizontal flipping')
    parser.add_argument('--no-rotation', action='store_true', help='Disable rotation augmentation')
    parser.add_argument('--rotation-range', type=float, default=15.0, help='Maximum rotation angle in degrees (default: 15.0)')
    parser.add_argument('--no-blur', action='store_true', help='Disable blur augmentation')
    parser.add_argument('--no-noise', action='store_true', help='Disable noise augmentation')
    parser.add_argument('--rotation-method', choices=['crop_inscribed', 'crop_tight', 'zoom_fill', 'edge_extend'], 
                       default='crop_inscribed', help='Method to handle rotation black edges (default: crop_inscribed)')
    
    args = parser.parse_args()
    
    # Validate multiplication factor
    if args.multiplication_factor < 1.0:
        print("Error: Multiplication factor must be >= 1.0")
        return
    
    # Create augmenter instance
    augmenter = YOLOClassificationAugmenter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        multiplication_factor=args.multiplication_factor,
        enable_flip=not args.no_flip,
        enable_rotation=not args.no_rotation,
        rotation_range=args.rotation_range,
        enable_blur=not args.no_blur,
        enable_noise=not args.no_noise,
        rotation_method=args.rotation_method
    )
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Perform augmentation
    augmenter.augment_dataset()

if __name__ == "__main__":
    main()

# Example usage:
# Basic usage - double the dataset size with random augmentations:
# python augment_classification_dataset.py --input-dir ./original_dataset --output-dir ./augmented_dataset --multiplication-factor 2.0

# Triple the dataset size with only rotation and blur:
# python augment_classification_dataset.py --input-dir ./data/train --output-dir ./data/train_augmented --multiplication-factor 3.0 --no-flip --no-noise

# Conservative augmentation - 50% increase with only flips and mild rotations:
# python augment_classification_dataset.py --input-dir ./data/train --output-dir ./data/train_augmented --multiplication-factor 1.5 --no-blur --no-noise --rotation-range 10

# Aggressive augmentation - 5x dataset size:
# python augment_classification_dataset.py --input-dir ./data/train --output-dir ./data/train_augmented --multiplication-factor 5.0

# Custom settings for different rotation handling:
# python ./scripts/augment_classification_dataset.py --input-dir D:/Dropbox/data/carabID/imgs/c1/train --output-dir D:/Dropbox/data/carabID/imgs/c1_augmented --multiplication-factor 3 --no-rotation --no-noise

# Expected output files (example with multiplication factor 2.0):
# augmented_dataset/
# ├── class1/
# │   ├── image1.jpg                    # Original
# │   ├── image1_flip_1.jpg             # Random augmentation 1
# │   ├── image2.jpg                    # Original
# │   ├── image2_rot_12.3deg_1.jpg      # Random augmentation 1
# │   ├── image3.jpg                    # Original
# │   ├── image3_gblur_1.jpg            # Random augmentation 1
# │   └── ...
# └── ...


