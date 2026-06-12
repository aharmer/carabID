import torch
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import yaml
from collections import defaultdict

# Load the Ultralytics YOLO model
from ultralytics import YOLO

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate YOLO object detection model performance')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained YOLO detection model (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML file or dataset directory containing train/val/test subdirectories')
    
    # Optional arguments
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which split to evaluate (default: test)')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for model input (default: 640)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output during validation')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save prediction results in YOLO format')
    parser.add_argument('--save-conf', action='store_true',
                        help='Save prediction confidences in saved txt files')
    
    return parser.parse_args()

def create_data_yaml(data_path, split='test'):
    """Create a temporary data.yaml file for YOLO validation if directory is provided."""
    if data_path.endswith('.yaml') or data_path.endswith('.yml'):
        return data_path
    
    # If it's a directory, create a temporary yaml file
    data_dir = Path(data_path)
    train_path = data_dir / 'train'
    valid_path = data_dir / 'valid'  # Detection uses 'valid' directory
    test_path = data_dir / 'test'
    
    # Create temporary yaml content
    yaml_content = {
        'path': str(data_dir.absolute()),
    }
    
    # Add paths for existing splits - yaml uses 'val' key but points to 'valid' directory
    if train_path.exists():
        yaml_content['train'] = str(train_path / 'images')
    if valid_path.exists():
        yaml_content['val'] = str(valid_path / 'images')  # yaml key is 'val' but directory is 'valid'
    if test_path.exists():
        yaml_content['test'] = str(test_path / 'images')
    
    # Try to infer class names from a labels file or use generic names
    class_names = infer_class_names(data_dir, split)
    if class_names:
        yaml_content['nc'] = len(class_names)
        yaml_content['names'] = class_names
    
    # Save temporary yaml file
    temp_yaml_path = data_dir / 'temp_data.yaml'
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    return str(temp_yaml_path)

def infer_class_names(data_dir, split):
    """Try to infer class names from the dataset."""
    split_dir = data_dir / split / 'labels'
    if not split_dir.exists():
        return None
    
    class_ids = set()
    # Read a few label files to get class IDs
    label_files = list(split_dir.glob('*.txt'))[:100]  # Sample first 100 files
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
        except (ValueError, IndexError):
            continue
    
    if class_ids:
        max_class_id = max(class_ids)
        return [f'class_{i}' for i in range(max_class_id + 1)]
    
    return None


def count_images_per_split(data_yaml_path):
    """Count images in each split."""
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        split_counts = {}
        
        print(f"Data config keys: {list(data_config.keys())}")
        
        # Handle different ways paths might be specified in the YAML
        for split in ['train', 'val', 'test']:
            split_counts[split] = 0
            
            # Try to get the path for this split
            images_path = None
            
            if split in data_config and data_config[split]:
                # Path is directly specified in YAML
                specified_path = Path(data_config[split])
                
                print(f"Debug: YAML {split} path = {specified_path}")
                
                # Check if this path directly contains images
                if specified_path.exists() and specified_path.is_dir():
                    # First, check if it's directly an images directory
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
                    direct_images = [f for f in specified_path.iterdir() 
                                   if f.is_file() and f.suffix.lower() in image_extensions]
                    
                    if direct_images:
                        # Images are directly in this directory
                        images_path = specified_path
                    else:
                        # Check if there's an 'images' subdirectory
                        images_subdir = specified_path / 'images'
                        if images_subdir.exists() and images_subdir.is_dir():
                            images_path = images_subdir
                        else:
                            # List contents to help debug
                            print(f"Debug: Contents of {specified_path}:")
                            for item in specified_path.iterdir():
                                print(f"  {'Dir' if item.is_dir() else 'File'}: {item.name}")
            
            if images_path and images_path.exists() and images_path.is_dir():
                print(f"Looking for {split} images in: {images_path}")
                
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
                count = 0
                for file in images_path.iterdir():
                    if file.is_file() and file.suffix.lower() in image_extensions:
                        count += 1
                split_counts[split] = count
                print(f"Found {count} images in {split} split")
            else:
                print(f"Images directory not found for {split} split. Tried: {images_path}")
        
        return split_counts
    except Exception as e:
        print(f"Warning: Could not count images - {e}")
        import traceback
        traceback.print_exc()
        return {'train': 0, 'val': 0, 'test': 0}

def run_yolo_validation(model, data_yaml_path, split, conf_thres, iou_thres, img_size, verbose, save_txt, save_conf):
    """Run YOLO's built-in validation."""
    print(f"\nRunning YOLO validation on {split} split...")
    
    # Set up validation parameters
    val_params = {
        'data': data_yaml_path,
        'split': split,
        'conf': conf_thres,
        'iou': iou_thres,
        'imgsz': img_size,
        'verbose': verbose,
        'save_txt': save_txt,
        'save_conf': save_conf,
    }
    
    # Run validation
    results = model.val(**val_params)
    
    return results

def extract_detailed_metrics(results):
    """Extract detailed metrics from YOLO validation results."""
    metrics = {}
    
    if hasattr(results, 'box'):
        box_metrics = results.box
        
        # Overall metrics
        metrics['mAP_50'] = float(box_metrics.map50) if box_metrics.map50 is not None else 0.0
        metrics['mAP_50_95'] = float(box_metrics.map) if box_metrics.map is not None else 0.0
        metrics['precision'] = float(box_metrics.mp) if box_metrics.mp is not None else 0.0
        metrics['recall'] = float(box_metrics.mr) if box_metrics.mr is not None else 0.0
        
        # Per-class metrics
        if hasattr(box_metrics, 'ap_class_index') and box_metrics.ap_class_index is not None:
            class_indices = box_metrics.ap_class_index
            class_names = results.names if hasattr(results, 'names') else {}
            
            # Per-class AP at IoU=0.50
            if hasattr(box_metrics, 'ap50') and box_metrics.ap50 is not None:
                metrics['per_class_ap50'] = {}
                for i, class_idx in enumerate(class_indices):
                    class_name = class_names.get(int(class_idx), f'class_{int(class_idx)}')
                    metrics['per_class_ap50'][class_name] = float(box_metrics.ap50[i])
            
            # Per-class AP at IoU=0.50:0.95
            if hasattr(box_metrics, 'ap') and box_metrics.ap is not None:
                metrics['per_class_ap50_95'] = {}
                for i, class_idx in enumerate(class_indices):
                    class_name = class_names.get(int(class_idx), f'class_{int(class_idx)}')
                    # box_metrics.ap is shape [num_classes, num_iou_thresholds]
                    # We want the mean across all IoU thresholds
                    metrics['per_class_ap50_95'][class_name] = float(np.mean(box_metrics.ap[i]))
    
    return metrics

def save_results_to_csv(metrics, image_counts, output_dir, split_name):
    """Save detailed results to CSV files."""
    
    # Save overall metrics
    overall_metrics = {
        'Metric': ['mAP@0.50', 'mAP@0.50:0.95', 'Precision', 'Recall'],
        'Value': [
            metrics.get('mAP_50', 0.0),
            metrics.get('mAP_50_95', 0.0),
            metrics.get('precision', 0.0),
            metrics.get('recall', 0.0)
        ]
    }
    
    overall_df = pd.DataFrame(overall_metrics)
    overall_path = output_dir / f'{split_name}_overall_metrics.csv'
    overall_df.to_csv(overall_path, index=False)
    print(f"Overall metrics saved to: {overall_path}")
    
    # Save per-class metrics
    if 'per_class_ap50' in metrics or 'per_class_ap50_95' in metrics:
        per_class_data = []
        
        all_classes = set()
        if 'per_class_ap50' in metrics:
            all_classes.update(metrics['per_class_ap50'].keys())
        if 'per_class_ap50_95' in metrics:
            all_classes.update(metrics['per_class_ap50_95'].keys())
        
        for class_name in sorted(all_classes):
            per_class_data.append({
                'Class': class_name,
                'Train_Images': image_counts.get('train', 0),
                'Val_Images': image_counts.get('val', 0),
                'Test_Images': image_counts.get('test', 0),
                'AP@0.50': metrics.get('per_class_ap50', {}).get(class_name, 0.0),
                'AP@0.50:0.95': metrics.get('per_class_ap50_95', {}).get(class_name, 0.0)
            })
        
        per_class_df = pd.DataFrame(per_class_data)
        per_class_path = output_dir / f'{split_name}_per_class_metrics.csv'
        per_class_df.to_csv(per_class_path, index=False)
        print(f"Per-class metrics saved to: {per_class_path}")
        
        # Debug: Print what we're saving
        print(f"\nDebug - Image counts: {image_counts}")
        print(f"Debug - Classes found: {sorted(all_classes)}")
    else:
        print("Warning: No per-class metrics found to save")

def print_results_summary(metrics, split_name):
    """Print a formatted summary of results."""
    print(f"\n{'='*60}")
    print(f"OBJECT DETECTION PERFORMANCE SUMMARY ({split_name.upper()} SET)")
    print(f"{'='*60}")
    
    print(f"Overall mAP@0.50     : {metrics.get('mAP_50', 0.0):.4f}")
    print(f"Overall mAP@0.50:0.95: {metrics.get('mAP_50_95', 0.0):.4f}")
    print(f"Overall Precision    : {metrics.get('precision', 0.0):.4f}")
    print(f"Overall Recall       : {metrics.get('recall', 0.0):.4f}")
    
    # Per-class results
    if 'per_class_ap50' in metrics:
        print(f"\n{'='*80}")
        print(f"PER-CLASS AVERAGE PRECISION ({split_name.upper()} SET)")
        print(f"{'='*80}")
        print(f"{'Class':<25} {'AP@0.50':<12} {'AP@0.50:0.95':<15}")
        print(f"{'-'*80}")
        
        for class_name in sorted(metrics['per_class_ap50'].keys()):
            ap50 = metrics['per_class_ap50'].get(class_name, 0.0)
            ap50_95 = metrics.get('per_class_ap50_95', {}).get(class_name, 0.0)
            print(f"{class_name:<25} {ap50:<12.4f} {ap50_95:<15.4f}")

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory based on model path
    model_path = Path(args.model)
    model_parent_dir = model_path.parent.parent  # Go up from weights/ to model directory
    output_dir = model_parent_dir / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading detection model from: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Evaluating on: {args.split} split")
    print(f"Results will be saved to: {output_dir}")
    print(f"Confidence threshold: {args.conf_thres}")
    print(f"IoU threshold: {args.iou_thres}")
    print(f"Image size: {args.img_size}")
    
    # Load the model
    try:
        model = YOLO(args.model)
        print("Detection model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create or validate data.yaml file
    try:
        data_yaml_path = create_data_yaml(args.data, args.split)
        print(f"Using data configuration: {data_yaml_path}")
    except Exception as e:
        print(f"Error processing data configuration: {e}")
        return
    
    # Count images
    print("\nCounting dataset statistics...")
    image_counts = count_images_per_split(data_yaml_path)
    
    print(f"Dataset statistics:")
    print(f"  Train images: {image_counts['train']}")
    print(f"  Val images: {image_counts['val']}")
    print(f"  Test images: {image_counts['test']}")
    
    # Run YOLO validation
    try:
        results = run_yolo_validation(
            model, data_yaml_path, args.split, args.conf_thres, 
            args.iou_thres, args.img_size, args.verbose, args.save_txt, args.save_conf
        )
    except Exception as e:
        print(f"Error during validation: {e}")
        return
    
    # Extract detailed metrics
    metrics = extract_detailed_metrics(results)
    
    # Print results summary
    print_results_summary(metrics, args.split)
    
    # Save results to CSV
    save_results_to_csv(metrics, image_counts, output_dir, args.split)
    
    # Clean up temporary yaml file if created
    temp_yaml = Path(args.data) / 'temp_data.yaml'
    if temp_yaml.exists() and not args.data.endswith(('.yaml', '.yml')):
        temp_yaml.unlink()
    
    print(f"\nEvaluation complete! All results saved to: {output_dir}")

if __name__ == "__main__":
    main()

# Usage
# python ./scripts/assess_detection_model.py --model D:/Dropbox/data/carabID/runs/detect/model_11n_ep30_autobatch/weights/best.pt --data D:/Dropbox/data/carabID/imgs/detection_set/data.yaml --split test


