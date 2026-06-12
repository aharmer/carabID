import os
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import yaml
import argparse

class DatasetBalancer:
    """
    A class to handle dataset balancing for classification tasks
    """
    
    def __init__(self, dataset_path, other_classes=None):
        """
        Initialize the dataset balancer
        
        Args:
            dataset_path: Path to your dataset directory
            other_classes: List of class names that are considered "other" classes
        """
        self.dataset_path = Path(dataset_path)
        self.other_classes = other_classes or []
        self.class_counts = {}
        self.class_names = []
        
    def analyze_dataset(self):
        """
        Analyze the current dataset distribution
        """
        print("Analyzing dataset distribution...")
        
        # Get class names from dataset.yaml if it exists
        yaml_path = self.dataset_path / "dataset.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                self.class_names = data.get('names', [])
        
        # Count images in each class folder
        train_path = self.dataset_path / "train"
        if train_path.exists():
            for class_folder in train_path.iterdir():
                if class_folder.is_dir():
                    class_name = class_folder.name
                    image_count = len([f for f in class_folder.iterdir() 
                                     if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                    self.class_counts[class_name] = image_count
        
        # Print analysis
        total_images = sum(self.class_counts.values())
        print(f"\nDataset Analysis:")
        print(f"Total images: {total_images}")
        print(f"Total classes: {len(self.class_counts)}")
        
        # Separate other classes from species classes
        other_count = sum(self.class_counts.get(cls, 0) for cls in self.other_classes)
        species_count = total_images - other_count
        
        print(f"\nClass Distribution:")
        print(f"'Other' classes: {other_count} images ({other_count/total_images*100:.1f}%)")
        print(f"Species classes: {species_count} images ({species_count/total_images*100:.1f}%)")
        
        # Show top 10 classes by count
        sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 classes by image count:")
        for class_name, count in sorted_classes[:10]:
            print(f"  {class_name}: {count} images")
        
        return self.class_counts
    
    def undersample_dataset(self, target_ratio=2.0, output_path=None, seed=42):
        """
        Undersample the majority classes (other classes) to balance the dataset
        
        Args:
            target_ratio: Target ratio of other classes to largest species class
            output_path: Path to save the balanced dataset
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        if output_path is None:
            output_path = self.dataset_path.parent / f"{self.dataset_path.name}_balanced"
        
        output_path = Path(output_path)
        
        # Calculate target counts
        species_counts = {k: v for k, v in self.class_counts.items() if k not in self.other_classes}
        max_species_count = max(species_counts.values()) if species_counts else 1000
        target_other_count = int(max_species_count * target_ratio)
        
        print(f"\nUndersampling Strategy:")
        print(f"Largest species class: {max_species_count} images")
        print(f"Target count per 'other' class: {target_other_count} images")
        
        # Create output directory structure
        for split in ['train', 'val', 'test']:
            split_path = output_path / split
            split_path.mkdir(parents=True, exist_ok=True)
        
        # Process each split
        for split in ['train', 'valid', 'test']:
            input_split_path = self.dataset_path / split
            output_split_path = output_path / split
            
            if not input_split_path.exists():
                continue
                
            print(f"\nProcessing {split} split...")
            
            for class_folder in input_split_path.iterdir():
                if not class_folder.is_dir():
                    continue
                    
                class_name = class_folder.name
                output_class_path = output_split_path / class_name
                output_class_path.mkdir(exist_ok=True)
                
                # Get all images in this class
                images = [f for f in class_folder.iterdir() 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                
                # Determine how many images to keep
                if class_name in self.other_classes:
                    # Undersample other classes
                    keep_count = min(len(images), target_other_count)
                    selected_images = random.sample(images, keep_count)
                else:
                    # Keep all species images
                    selected_images = images
                
                # Copy selected images
                for img in selected_images:
                    shutil.copy2(img, output_class_path / img.name)
                
                print(f"  {class_name}: {len(images)} -> {len(selected_images)} images")
        
        # Copy dataset.yaml if it exists
        yaml_path = self.dataset_path / "data.yaml"
        if yaml_path.exists():
            shutil.copy2(yaml_path, output_path / "dataset.yaml")
        
        print(f"\nBalanced dataset saved to: {output_path}")
        return output_path
    
    def calculate_class_weights(self, method='inverse_freq'):
        """
        Calculate class weights for loss function
        
        Args:
            method: Method to calculate weights ('inverse_freq', 'sqrt_inverse_freq', 'balanced')
        """
        total_samples = sum(self.class_counts.values())
        num_classes = len(self.class_counts)
        
        if method == 'inverse_freq':
            # Weight inversely proportional to class frequency
            weights = {cls: total_samples / (num_classes * count) 
                      for cls, count in self.class_counts.items()}
        
        elif method == 'sqrt_inverse_freq':
            # Square root of inverse frequency (less aggressive)
            weights = {cls: np.sqrt(total_samples / (num_classes * count)) 
                      for cls, count in self.class_counts.items()}
        
        elif method == 'balanced':
            # Sklearn-style balanced weights
            weights = {cls: total_samples / (num_classes * count) 
                      for cls, count in self.class_counts.items()}
        
        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {cls: w / weight_sum * num_classes for cls, w in weights.items()}
        
        print(f"\nClass Weights ({method}):")
        for cls, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cls}: {weight:.4f}")
        
        return weights


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Balance a classification dataset by undersampling majority classes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python balance_classification_dataset.py --dataset-path /path/to/dataset --other-classes diptera hymenoptera
  
  python balance_classification_dataset.py -d /path/to/dataset -o diptera hymenoptera -r 1.5 --output-path /path/to/output
  
  python balance_classification_dataset.py --dataset-path /path/to/dataset --analyze-only
        """
    )
    
    parser.add_argument(
        '--dataset-path', '-d',
        type=str,
        required=True,
        help='Path to the dataset directory containing train/val/test splits'
    )
    
    parser.add_argument(
        '--other-classes', '-o',
        type=str,
        nargs='*',
        default=[],
        help='Names of classes considered as "other" classes (space-separated)'
    )
    
    parser.add_argument(
        '--target-ratio', '-r',
        type=float,
        default=2.0,
        help='Target ratio of other classes to largest species class (default: 2.0)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Path to save the balanced dataset (default: dataset_path_balanced)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--weight-method',
        choices=['inverse_freq', 'sqrt_inverse_freq', 'balanced'],
        default='inverse_freq',
        help='Method to calculate class weights (default: inverse_freq)'
    )
    
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze the dataset without creating a balanced version'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run the dataset balancer
    """
    args = parse_arguments()
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist!")
        return
    
    # Initialize balancer
    print(f"Dataset path: {args.dataset_path}")
    print(f"Other classes: {args.other_classes}")
    print(f"Target ratio: {args.target_ratio}")
    if args.output_path:
        print(f"Output path: {args.output_path}")
    
    balancer = DatasetBalancer(args.dataset_path, args.other_classes)
    
    # Analyze current dataset
    class_counts = balancer.analyze_dataset()
    
    if not class_counts:
        print("Error: No classes found in the dataset!")
        return
    
    # Calculate class weights
    weights = balancer.calculate_class_weights(method=args.weight_method)
    
    # Create balanced dataset if not analyze-only
    if not args.analyze_only:
        balanced_dataset_path = balancer.undersample_dataset(
            target_ratio=args.target_ratio,
            output_path=args.output_path,
            seed=args.seed
        )
        print(f"\nBalancing complete! Balanced dataset saved to: {balanced_dataset_path}")
    else:
        print("\nAnalysis complete. No balanced dataset created (--analyze-only flag used).")


if __name__ == "__main__":
    main()


# Usage
# python ./scripts/balance_classification_dataset.py --dataset-path D:/Dropbox/data/dipteraID/imgs/c1 --analyze-only
# python ./scripts/balance_classification_dataset.py --dataset-path D:/Dropbox/data/dipteraID/imgs/c1 --other-classes hymenoptera --target-ratio 1
