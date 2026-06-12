import torch
from torchvision import datasets, transforms
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import os
import argparse
from pathlib import Path

# Load the Ultralytics YOLO model
from ultralytics import YOLO

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Assess YOLO classification model performance')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO classification model (.pt file)')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to dataset root directory containing train/val/test subdirectories')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='test',
                        help='Dataset split to evaluate on (default: test)')
    parser.add_argument('--output-dir', type=str, 
                        help='Directory to save results (optional - defaults to model_directory/results)')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for evaluation (default: 32)')
    parser.add_argument('--img-size', type=int, default=640, 
                        help='Image size for model input (default: 640)')
    
    return parser.parse_args()

def count_images_per_class(dataset_path, class_names):
    """Count the number of images per class in a dataset directory."""
    class_counts = {}
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.exists(class_dir):
            # Count image files (common extensions)
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            count = 0
            for file in os.listdir(class_dir):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    count += 1
            class_counts[class_name] = count
        else:
            class_counts[class_name] = 0
    
    return class_counts

def get_output_directory(model_path, custom_output_dir=None):
    """
    Determine output directory based on model path or custom directory.
    Creates a 'results' folder within the model directory.
    Expected model path structure: runs/classify/model_name/weights/best.pt
    """
    if custom_output_dir:
        return custom_output_dir
    
    # Convert to Path object for easier manipulation
    model_path = Path(model_path)
    
    # Check if the model path follows the expected structure
    path_parts = model_path.parts
    
    # Look for 'runs' directory in the path
    try:
        runs_index = path_parts.index('runs')
        
        # Expected structure: runs/classify/model_name/weights/best.pt
        if len(path_parts) > runs_index + 2:
            # Reconstruct path up to model_name directory and add 'results'
            model_dir = Path(*path_parts[:runs_index + 3])  # runs/classify/model_name
            results_dir = model_dir / 'results'
            return str(results_dir)
        else:
            # Fallback to model file's directory + results
            return str(model_path.parent / 'results')
            
    except (ValueError, IndexError):
        # 'runs' not found in path, use model file's directory + results
        return str(model_path.parent / 'results')

def get_classification_paths(data_root, eval_split='test'):
    """Get paths to train, val, and test directories from the dataset root."""
    train_path = os.path.join(data_root, 'train')
    val_path = os.path.join(data_root, 'val')
    test_path = os.path.join(data_root, 'test')
    
    # Check which paths exist
    paths = {
        'train': train_path if os.path.exists(train_path) else None,
        'val': val_path if os.path.exists(val_path) else None,
        'test': test_path if os.path.exists(test_path) else None
    }
    
    # Check if the requested evaluation split exists
    eval_path = os.path.join(data_root, eval_split)
    if not os.path.exists(eval_path):
        available_splits = [split for split, path in paths.items() if path is not None]
        raise ValueError(f"Evaluation split '{eval_split}' not found at {eval_path}. Available splits: {available_splits}")
    
    paths['eval'] = eval_path
    paths['eval_split_name'] = eval_split
    
    return paths
    """Count the number of images per class in a dataset directory."""
    class_counts = {}
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.exists(class_dir):
            # Count image files (common extensions)
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            count = 0
            for file in os.listdir(class_dir):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    count += 1
            class_counts[class_name] = count
        else:
            class_counts[class_name] = 0
    
    return class_counts



def evaluate_classification_model(model, dataset, output_dir, eval_split, img_size):
    """Evaluate classification model using model.predict() method."""
    print(f"Evaluating classification model on '{eval_split}' split...")
    
    all_preds = []
    all_labels = []
    all_confidences = []
    all_image_paths = []
    
    # Get image paths for proper prediction
    image_paths = []
    for img_path, label in dataset.samples:
        image_paths.append((img_path, label))
    
    print(f"Processing {len(image_paths)} images from {eval_split} split...")
    
    # Process images in batches
    batch_size = 32
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = [path for path, _ in batch_paths]
        batch_labels = [label for _, label in batch_paths]
        
        # Use YOLO's predict method with image paths
        results = model.predict(batch_imgs, imgsz=img_size, verbose=False)
        
        for j, result in enumerate(results):
            if result.probs is not None:
                pred_class = result.probs.top1
                confidence = result.probs.top1conf.item() if hasattr(result.probs, 'top1conf') else 0.0
                all_preds.append(pred_class)
                all_confidences.append(confidence)
            else:
                all_preds.append(0)
                all_confidences.append(0.0)
            
            all_labels.append(batch_labels[j])
            all_image_paths.append(batch_imgs[j])
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'image_path': all_image_paths,
        'true_label': all_labels, 
        'predicted_label': all_preds,
        'confidence': all_confidences
    })
    results_df.to_csv(os.path.join(output_dir, f'classification_results_{eval_split}.csv'), index=False)
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"Overall Accuracy on {eval_split}: {overall_accuracy:.4f}")
    print(f"Overall Precision on {eval_split}: {precision:.4f}")
    print(f"Overall Recall on {eval_split}: {recall:.4f}")
    print(f"Overall F1 Score on {eval_split}: {f1:.4f}")
    
    # Calculate class-wise metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_wise_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    return {
        'overall_accuracy': overall_accuracy,
        'overall_precision': precision,
        'overall_recall': recall,
        'overall_f1': f1,
        'confusion_matrix': conf_matrix,
        'class_wise_accuracy': class_wise_accuracy,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_confidences': all_confidences
    }

def save_classification_class_metrics(dataset, results, data_paths, output_dir, eval_split):
    """Save detailed class-wise metrics for classification."""
    all_labels = results['all_labels']
    all_preds = results['all_preds']
    class_wise_accuracy = results['class_wise_accuracy']
    
    # Count images per class in available splits
    train_class_counts = {}
    val_class_counts = {}
    test_class_counts = {}
    eval_class_counts = count_images_per_class(data_paths['eval'], dataset.classes)
    
    if data_paths['train']:
        train_class_counts = count_images_per_class(data_paths['train'], dataset.classes)
    if data_paths['val']:
        val_class_counts = count_images_per_class(data_paths['val'], dataset.classes)
    if data_paths['test']:
        test_class_counts = count_images_per_class(data_paths['test'], dataset.classes)
    
    # Collect metrics for each class
    metrics_list = []
    for idx, class_name in enumerate(dataset.classes):
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=[idx], average='macro', zero_division=0
        )
        metrics_list.append({
            'Class Name': class_name,
            'Class Index': idx,
            'Train Images': train_class_counts.get(class_name, 'N/A'),
            'Val Images': val_class_counts.get(class_name, 'N/A'),
            'Test Images': test_class_counts.get(class_name, 'N/A'),
            f'{eval_split.title()} Images': eval_class_counts.get(class_name, 0),
            'Accuracy': class_wise_accuracy[idx] if idx < len(class_wise_accuracy) else 0,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(output_dir, f'classification_class_metrics_{eval_split}.csv'), index=False)
    
    print(f"\nClass-wise metrics saved to: {os.path.join(output_dir, f'classification_class_metrics_{eval_split}.csv')}")

def main():
    args = parse_arguments()
    
    # Determine output directory (now creates 'results' subfolder)
    output_dir = get_output_directory(args.model, args.output_dir)
    print(f"Results will be saved to: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading classification model from: {args.model}")
    model = YOLO(args.model)
    
    # Classification task - get paths from dataset root
    print(f"Running classification evaluation on '{args.split}' split...")
    
    # Get paths to train/val/test directories
    data_paths = get_classification_paths(args.data, args.split)
    print(f"Using {args.split} data from: {data_paths['eval']}")
    if data_paths['train']:
        print(f"Train data found at: {data_paths['train']}")
    if data_paths['val']:
        print(f"Validation data found at: {data_paths['val']}")
    if data_paths['test']:
        print(f"Test data found at: {data_paths['test']}")
    
    # Define the transformation for loading dataset structure
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    
    # Load the evaluation dataset to get structure and labels
    eval_dataset = datasets.ImageFolder(data_paths['eval'], transform=transform)
    
    # Evaluate the model using predict method (respects your existing splits)
    results = evaluate_classification_model(model, eval_dataset, output_dir, args.split, args.img_size)
    
    # Save class-wise metrics
    save_classification_class_metrics(eval_dataset, results, data_paths, output_dir, args.split)
    
    # Save overall metrics
    overall_metrics = {
        'evaluation_split': args.split,
        'overall_accuracy': results['overall_accuracy'],
        'overall_precision': results['overall_precision'],
        'overall_recall': results['overall_recall'],
        'overall_f1_score': results['overall_f1']
    }
    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(os.path.join(output_dir, f'classification_overall_metrics_{args.split}.csv'), index=False)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()



# Usage
# python ./scripts/assess_classification_model.py --model D:/Dropbox/data/carabID/runs/classify/carabid_cv_11ncls_ep30_autobatch_do02_lr001_fold5/weights/best.pt --data D:/Dropbox/data/carabID/imgs/cv_classification_set/fold_5 --split test

