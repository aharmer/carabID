"""
YOLO Classification Model Cross-Validation Assessment Script

This script evaluates all k-fold models on a common test set and aggregates results.
It can evaluate on either a held-out test set or use each fold's validation set.

Usage:
    python assess_classification_cv.py --models_dir path/to/cv/results --data path/to/dataset --split test
    
Example:
    python ./scripts/kfold_assess_classification_models.py --models_dir ./runs/classify --name carabid_cv_11ncls_ep30_autobatch_do02_lr001 --data ./imgs/cv_classification_set --split test --save-plots
"""

import torch
from torchvision import datasets, transforms
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from ultralytics import YOLO


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Assess YOLO classification models from k-fold CV')
    
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory containing fold model subdirectories (e.g., runs/classify)')
    parser.add_argument('--name', type=str, required=True,
                        help='Base name of CV run (e.g., carabid_cv)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CV dataset root directory')
    parser.add_argument('--split', type=str, choices=['test', 'val'], default='test',
                        help='Dataset split to evaluate on (test for held-out, val for each fold\'s validation)')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save results (default: models_dir/name_cv_assessment)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for model input (default: 640)')
    parser.add_argument('--weight-type', type=str, choices=['best', 'last'], default='best',
                        help='Which model weights to use (default: best)')
    parser.add_argument('--folds', type=str, default='all',
                        help='Which folds to evaluate (e.g., "1,3,5" or "all")')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save confusion matrix plots for each fold')
    
    return parser.parse_args()


def detect_fold_models(models_dir: Path, name: str, weight_type: str) -> List[Dict]:
    """
    Detect all fold model directories and their weights.
    
    Args:
        models_dir: Directory containing model subdirectories
        name: Base name of the CV run
        weight_type: 'best' or 'last'
    
    Returns:
        List of dictionaries with fold info and model paths
    """
    fold_models = []
    
    # Look for directories matching the pattern: name_fold*
    for dir_path in sorted(models_dir.iterdir()):
        if dir_path.is_dir() and dir_path.name.startswith(f"{name}_fold"):
            try:
                # Extract fold number
                fold_num = int(dir_path.name.replace(f"{name}_fold", ""))
                
                # Look for weights
                weights_dir = dir_path / "weights"
                if not weights_dir.exists():
                    print(f"Warning: No weights directory found in {dir_path.name}")
                    continue
                
                weight_file = weights_dir / f"{weight_type}.pt"
                if not weight_file.exists():
                    print(f"Warning: No {weight_type}.pt found in {dir_path.name}/weights")
                    continue
                
                fold_models.append({
                    'fold': fold_num,
                    'model_path': str(weight_file),
                    'model_dir': str(dir_path)
                })
                
            except ValueError:
                continue
    
    if not fold_models:
        print(f"Error: No fold models found matching pattern '{name}_fold*' in {models_dir}")
        return []
    
    # Sort by fold number
    fold_models.sort(key=lambda x: x['fold'])
    
    return fold_models


def get_evaluation_dataset(data_root: Path, split: str, fold_num: int = None) -> Tuple[Path, datasets.ImageFolder]:
    """
    Get the evaluation dataset path and ImageFolder dataset.
    
    Args:
        data_root: Root directory of CV dataset
        split: 'test' or 'val'
        fold_num: Fold number (only used if split='val')
    
    Returns:
        Tuple of (dataset_path, ImageFolder dataset)
    """
    if split == 'test':
        # Use common test set (should be same across all folds)
        eval_path = data_root / "fold_1" / "test"
        
        if not eval_path.exists():
            # Try without fold structure (single test set)
            eval_path = data_root / "test"
            
        if not eval_path.exists():
            raise ValueError(f"Test set not found at {eval_path}")
            
    else:  # split == 'val'
        if fold_num is None:
            raise ValueError("fold_num must be specified when split='val'")
        
        eval_path = data_root / f"fold_{fold_num}" / "val"
        
        if not eval_path.exists():
            raise ValueError(f"Validation set not found at {eval_path}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(eval_path, transform=transform)
    
    return eval_path, dataset


def evaluate_single_fold(model_path: str, dataset: datasets.ImageFolder, 
                         fold_num: int, img_size: int, batch_size: int) -> Dict:
    """
    Evaluate a single fold model on the dataset.
    
    Args:
        model_path: Path to model weights
        dataset: ImageFolder dataset
        fold_num: Fold number
        img_size: Image size for model input
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING FOLD {fold_num}")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    all_preds = []
    all_labels = []
    all_confidences = []
    all_image_paths = []
    
    # Get image paths
    image_paths = [(img_path, label) for img_path, label in dataset.samples]
    
    print(f"Processing {len(image_paths)} images...")
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = [path for path, _ in batch_paths]
        batch_labels = [label for _, label in batch_paths]
        
        # Predict
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
    
    # Calculate metrics
    overall_accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracy
    class_wise_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    print(f"Fold {fold_num} Results:")
    print(f"  Overall Accuracy: {overall_accuracy:.4f}")
    print(f"  Macro Precision:  {precision:.4f}")
    print(f"  Macro Recall:     {recall:.4f}")
    print(f"  Macro F1:         {f1:.4f}")
    
    return {
        'fold': fold_num,
        'model_path': model_path,
        'overall_accuracy': overall_accuracy,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'confusion_matrix': conf_matrix,
        'class_wise_accuracy': class_wise_accuracy,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'predictions': all_preds,
        'labels': all_labels,
        'confidences': all_confidences,
        'image_paths': all_image_paths
    }


def aggregate_fold_results(fold_results: List[Dict], class_names: List[str]) -> Dict:
    """
    Aggregate results across all folds.
    
    Args:
        fold_results: List of result dictionaries from each fold
        class_names: List of class names
    
    Returns:
        Dictionary with aggregated statistics
    """
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS ACROSS FOLDS")
    print(f"{'='*70}")
    
    n_folds = len(fold_results)
    n_classes = len(class_names)
    
    # Collect overall metrics
    accuracies = [r['overall_accuracy'] for r in fold_results]
    precisions = [r['macro_precision'] for r in fold_results]
    recalls = [r['macro_recall'] for r in fold_results]
    f1s = [r['macro_f1'] for r in fold_results]
    
    # Collect per-class metrics
    per_class_acc = np.array([r['class_wise_accuracy'] for r in fold_results])
    per_class_prec = np.array([r['per_class_precision'] for r in fold_results])
    per_class_rec = np.array([r['per_class_recall'] for r in fold_results])
    per_class_f1_scores = np.array([r['per_class_f1'] for r in fold_results])
    
    # Calculate aggregated confusion matrix (sum across folds)
    aggregated_cm = sum(r['confusion_matrix'] for r in fold_results)
    
    aggregated = {
        'n_folds': n_folds,
        'overall_accuracy_mean': np.mean(accuracies),
        'overall_accuracy_std': np.std(accuracies),
        'overall_accuracy_min': np.min(accuracies),
        'overall_accuracy_max': np.max(accuracies),
        'macro_precision_mean': np.mean(precisions),
        'macro_precision_std': np.std(precisions),
        'macro_recall_mean': np.mean(recalls),
        'macro_recall_std': np.std(recalls),
        'macro_f1_mean': np.mean(f1s),
        'macro_f1_std': np.std(f1s),
        'per_class_accuracy_mean': per_class_acc.mean(axis=0),
        'per_class_accuracy_std': per_class_acc.std(axis=0),
        'per_class_precision_mean': per_class_prec.mean(axis=0),
        'per_class_precision_std': per_class_prec.std(axis=0),
        'per_class_recall_mean': per_class_rec.mean(axis=0),
        'per_class_recall_std': per_class_rec.std(axis=0),
        'per_class_f1_mean': per_class_f1_scores.mean(axis=0),
        'per_class_f1_std': per_class_f1_scores.std(axis=0),
        'aggregated_confusion_matrix': aggregated_cm
    }
    
    return aggregated


def save_fold_results(fold_results: List[Dict], class_names: List[str], 
                     output_dir: Path, eval_split: str):
    """Save individual fold results to CSV files."""
    
    for result in fold_results:
        fold_num = result['fold']
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'image_path': result['image_paths'],
            'true_label_idx': result['labels'],
            'true_label_name': [class_names[idx] for idx in result['labels']],
            'predicted_label_idx': result['predictions'],
            'predicted_label_name': [class_names[idx] for idx in result['predictions']],
            'confidence': result['confidences'],
            'correct': [l == p for l, p in zip(result['labels'], result['predictions'])]
        })
        predictions_df.to_csv(
            output_dir / f'fold_{fold_num}_predictions_{eval_split}.csv', 
            index=False
        )
        
        # Save per-class metrics
        class_metrics_df = pd.DataFrame({
            'class_name': class_names,
            'class_index': range(len(class_names)),
            'accuracy': result['class_wise_accuracy'],
            'precision': result['per_class_precision'],
            'recall': result['per_class_recall'],
            'f1_score': result['per_class_f1']
        })
        class_metrics_df.to_csv(
            output_dir / f'fold_{fold_num}_class_metrics_{eval_split}.csv',
            index=False
        )
        
        # Save confusion matrix
        cm_df = pd.DataFrame(
            result['confusion_matrix'],
            index=[f'True_{cn}' for cn in class_names],
            columns=[f'Pred_{cn}' for cn in class_names]
        )
        cm_df.to_csv(output_dir / f'fold_{fold_num}_confusion_matrix_{eval_split}.csv')


def save_aggregated_results(aggregated: Dict, class_names: List[str], 
                           output_dir: Path, eval_split: str):
    """Save aggregated results across all folds."""
    
    # Overall metrics summary
    overall_summary = pd.DataFrame([{
        'metric': 'Overall Accuracy',
        'mean': aggregated['overall_accuracy_mean'],
        'std': aggregated['overall_accuracy_std'],
        'min': aggregated['overall_accuracy_min'],
        'max': aggregated['overall_accuracy_max']
    }, {
        'metric': 'Macro Precision',
        'mean': aggregated['macro_precision_mean'],
        'std': aggregated['macro_precision_std'],
        'min': aggregated['macro_precision_mean'] - aggregated['macro_precision_std'],
        'max': aggregated['macro_precision_mean'] + aggregated['macro_precision_std']
    }, {
        'metric': 'Macro Recall',
        'mean': aggregated['macro_recall_mean'],
        'std': aggregated['macro_recall_std'],
        'min': aggregated['macro_recall_mean'] - aggregated['macro_recall_std'],
        'max': aggregated['macro_recall_mean'] + aggregated['macro_recall_std']
    }, {
        'metric': 'Macro F1',
        'mean': aggregated['macro_f1_mean'],
        'std': aggregated['macro_f1_std'],
        'min': aggregated['macro_f1_mean'] - aggregated['macro_f1_std'],
        'max': aggregated['macro_f1_mean'] + aggregated['macro_f1_std']
    }])
    overall_summary.to_csv(
        output_dir / f'cv_overall_summary_{eval_split}.csv',
        index=False
    )
    
    # Per-class aggregated metrics
    per_class_summary = pd.DataFrame({
        'class_name': class_names,
        'class_index': range(len(class_names)),
        'accuracy_mean': aggregated['per_class_accuracy_mean'],
        'accuracy_std': aggregated['per_class_accuracy_std'],
        'precision_mean': aggregated['per_class_precision_mean'],
        'precision_std': aggregated['per_class_precision_std'],
        'recall_mean': aggregated['per_class_recall_mean'],
        'recall_std': aggregated['per_class_recall_std'],
        'f1_mean': aggregated['per_class_f1_mean'],
        'f1_std': aggregated['per_class_f1_std']
    })
    per_class_summary.to_csv(
        output_dir / f'cv_per_class_summary_{eval_split}.csv',
        index=False
    )
    
    # Aggregated confusion matrix
    cm_df = pd.DataFrame(
        aggregated['aggregated_confusion_matrix'],
        index=[f'True_{cn}' for cn in class_names],
        columns=[f'Pred_{cn}' for cn in class_names]
    )
    cm_df.to_csv(output_dir / f'cv_aggregated_confusion_matrix_{eval_split}.csv')
    
    print(f"\n✓ Saved aggregated results to {output_dir}")


def save_comparison_table(fold_results: List[Dict], output_dir: Path, eval_split: str):
    """Create a comparison table showing all folds side by side."""
    
    comparison_data = []
    for result in fold_results:
        comparison_data.append({
            'fold': result['fold'],
            'accuracy': result['overall_accuracy'],
            'precision': result['macro_precision'],
            'recall': result['macro_recall'],
            'f1_score': result['macro_f1']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add mean and std rows
    mean_row = {
        'fold': 'MEAN',
        'accuracy': comparison_df['accuracy'].mean(),
        'precision': comparison_df['precision'].mean(),
        'recall': comparison_df['recall'].mean(),
        'f1_score': comparison_df['f1_score'].mean()
    }
    std_row = {
        'fold': 'STD',
        'accuracy': comparison_df['accuracy'].std(),
        'precision': comparison_df['precision'].std(),
        'recall': comparison_df['recall'].std(),
        'f1_score': comparison_df['f1_score'].std()
    }
    
    comparison_df = pd.concat([
        comparison_df,
        pd.DataFrame([mean_row, std_row])
    ], ignore_index=True)
    
    comparison_df.to_csv(
        output_dir / f'cv_fold_comparison_{eval_split}.csv',
        index=False
    )
    
    print(f"✓ Saved fold comparison to {output_dir / f'cv_fold_comparison_{eval_split}.csv'}")


def plot_confusion_matrices(fold_results: List[Dict], aggregated: Dict,
                           class_names: List[str], output_dir: Path, 
                           eval_split: str):
    """Plot confusion matrices for each fold and aggregated."""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot individual folds
        for result in fold_results:
            plt.figure(figsize=(10, 8))
            
            # Normalize confusion matrix
            cm_normalized = result['confusion_matrix'].astype('float') / result['confusion_matrix'].sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.title(f'Fold {result["fold"]} - Confusion Matrix (Normalized)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_dir / f'fold_{result["fold"]}_confusion_matrix_{eval_split}.png', dpi=300)
            plt.close()
        
        # Plot aggregated confusion matrix
        plt.figure(figsize=(12, 10))
        cm_normalized = aggregated['aggregated_confusion_matrix'].astype('float') / aggregated['aggregated_confusion_matrix'].sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Aggregated Confusion Matrix - All {aggregated["n_folds"]} Folds (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / f'cv_aggregated_confusion_matrix_{eval_split}.png', dpi=300)
        plt.close()
        
        print(f"✓ Saved confusion matrix plots to {output_dir}")
        
    except ImportError:
        print("Warning: matplotlib/seaborn not available, skipping plots")


def create_text_summary(fold_results: List[Dict], aggregated: Dict,
                       class_names: List[str], output_dir: Path,
                       eval_split: str, args):
    """Create a human-readable text summary."""
    
    summary_path = output_dir / f'cv_assessment_summary_{eval_split}.txt'
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CROSS-VALIDATION MODEL ASSESSMENT SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Evaluation split: {eval_split}\n")
        f.write(f"Number of folds: {len(fold_results)}\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write(f"Classes: {', '.join(class_names)}\n\n")
        
        f.write("="*70 + "\n")
        f.write("OVERALL METRICS (Mean ± Std)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Overall Accuracy:  {aggregated['overall_accuracy_mean']:.4f} ± {aggregated['overall_accuracy_std']:.4f}\n")
        f.write(f"Macro Precision:   {aggregated['macro_precision_mean']:.4f} ± {aggregated['macro_precision_std']:.4f}\n")
        f.write(f"Macro Recall:      {aggregated['macro_recall_mean']:.4f} ± {aggregated['macro_recall_std']:.4f}\n")
        f.write(f"Macro F1 Score:    {aggregated['macro_f1_mean']:.4f} ± {aggregated['macro_f1_std']:.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("PER-FOLD RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for result in fold_results:
            f.write(f"Fold {result['fold']}:\n")
            f.write(f"  Model: {result['model_path']}\n")
            f.write(f"  Overall Accuracy: {result['overall_accuracy']:.4f}\n")
            f.write(f"  Macro Precision:  {result['macro_precision']:.4f}\n")
            f.write(f"  Macro Recall:     {result['macro_recall']:.4f}\n")
            f.write(f"  Macro F1:         {result['macro_f1']:.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("PER-CLASS METRICS (Mean ± Std)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'Class':<30} {'Accuracy':<20} {'Precision':<20} {'Recall':<20} {'F1':<20}\n")
        f.write("-"*110 + "\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<30} ")
            f.write(f"{aggregated['per_class_accuracy_mean'][i]:.3f}±{aggregated['per_class_accuracy_std'][i]:.3f}       ")
            f.write(f"{aggregated['per_class_precision_mean'][i]:.3f}±{aggregated['per_class_precision_std'][i]:.3f}       ")
            f.write(f"{aggregated['per_class_recall_mean'][i]:.3f}±{aggregated['per_class_recall_std'][i]:.3f}       ")
            f.write(f"{aggregated['per_class_f1_mean'][i]:.3f}±{aggregated['per_class_f1_std'][i]:.3f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*70 + "\n\n")
        
        f.write("The metrics reported above represent the mean ± standard deviation across all folds.\n")
        f.write("Lower standard deviation indicates more consistent performance across folds.\n")
        f.write("High variance might indicate:\n")
        f.write("  - Different fold difficulties\n")
        f.write("  - Small class sizes leading to sampling variability\n")
        f.write("  - Model instability during training\n\n")
        
        # Identify best and worst performing classes
        best_class_idx = np.argmax(aggregated['per_class_f1_mean'])
        worst_class_idx = np.argmin(aggregated['per_class_f1_mean'])
        
        f.write(f"Best performing class:  {class_names[best_class_idx]} (F1: {aggregated['per_class_f1_mean'][best_class_idx]:.4f})\n")
        f.write(f"Worst performing class: {class_names[worst_class_idx]} (F1: {aggregated['per_class_f1_mean'][worst_class_idx]:.4f})\n")
    
    print(f"✓ Saved text summary to {summary_path}")


def main():
    args = parse_arguments()
    
    # Setup paths
    models_dir = Path(args.models_dir)
    data_root = Path(args.data)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = models_dir / f"{args.name}_cv_assessment"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("YOLO CLASSIFICATION CROSS-VALIDATION ASSESSMENT")
    print("="*70)
    print(f"Models directory: {models_dir}")
    print(f"CV run name: {args.name}")
    print(f"Dataset: {data_root}")
    print(f"Evaluation split: {args.split}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    # Detect fold models
    fold_models = detect_fold_models(models_dir, args.name, args.weight_type)
    
    if not fold_models:
        print("Error: No fold models found!")
        return
    
    print(f"\nDetected {len(fold_models)} fold models:")
    for fm in fold_models:
        print(f"  Fold {fm['fold']}: {fm['model_path']}")
    
    # Filter folds if specified
    if args.folds.lower() != 'all':
        try:
            selected_folds = [int(f.strip()) for f in args.folds.split(',')]
            fold_models = [fm for fm in fold_models if fm['fold'] in selected_folds]
            print(f"\nEvaluating selected folds: {[fm['fold'] for fm in fold_models]}")
        except Exception as e:
            print(f"Error parsing --folds argument: {e}")
            return
    
    # Get evaluation dataset
    # For test split, use common test set; for val split, we'll load per-fold
    if args.split == 'test':
        eval_path, eval_dataset = get_evaluation_dataset(data_root, args.split)
        print(f"\nUsing common test set: {eval_path}")
        print(f"Number of test images: {len(eval_dataset)}")
        print(f"Number of classes: {len(eval_dataset.classes)}")
        print(f"Classes: {eval_dataset.classes}")
    
    # Evaluate each fold
    fold_results = []
    
    for fm in fold_models:
        if args.split == 'val':
            # Load fold-specific validation set
            eval_path, eval_dataset = get_evaluation_dataset(
                data_root, args.split, fm['fold']
            )
            print(f"\nUsing fold {fm['fold']} validation set: {eval_path}")
        
        result = evaluate_single_fold(
            fm['model_path'],
            eval_dataset,
            fm['fold'],
            args.img_size,
            args.batch_size
        )
        fold_results.append(result)
    
    # Aggregate results
    aggregated = aggregate_fold_results(fold_results, eval_dataset.classes)
    
    # Save all results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    save_fold_results(fold_results, eval_dataset.classes, output_dir, args.split)
    save_aggregated_results(aggregated, eval_dataset.classes, output_dir, args.split)
    save_comparison_table(fold_results, output_dir, args.split)
    
    if args.save_plots:
        plot_confusion_matrices(fold_results, aggregated, eval_dataset.classes, 
                              output_dir, args.split)
    
    create_text_summary(fold_results, aggregated, eval_dataset.classes, 
                       output_dir, args.split, args)
    
    # Print final summary to console
    print(f"\n{'='*70}")
    print("FINAL CROSS-VALIDATION ASSESSMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Evaluated {len(fold_results)} folds on {args.split} set")
    print(f"Total images evaluated: {len(eval_dataset)}")
    print(f"\nOverall Performance (Mean ± Std):")
    print(f"  Accuracy:  {aggregated['overall_accuracy_mean']:.4f} ± {aggregated['overall_accuracy_std']:.4f}")
    print(f"  Precision: {aggregated['macro_precision_mean']:.4f} ± {aggregated['macro_precision_std']:.4f}")
    print(f"  Recall:    {aggregated['macro_recall_mean']:.4f} ± {aggregated['macro_recall_std']:.4f}")
    print(f"  F1 Score:  {aggregated['macro_f1_mean']:.4f} ± {aggregated['macro_f1_std']:.4f}")
    
    print(f"\nPer-Fold Performance Range:")
    print(f"  Accuracy: [{aggregated['overall_accuracy_min']:.4f}, {aggregated['overall_accuracy_max']:.4f}]")
    
    print(f"\n{'='*70}")
    print("✓ CROSS-VALIDATION ASSESSMENT COMPLETED!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - cv_assessment_summary_{args.split}.txt (human-readable summary)")
    print(f"  - cv_overall_summary_{args.split}.csv (aggregated metrics)")
    print(f"  - cv_per_class_summary_{args.split}.csv (per-class aggregated metrics)")
    print(f"  - cv_fold_comparison_{args.split}.csv (side-by-side fold comparison)")
    print(f"  - cv_aggregated_confusion_matrix_{args.split}.csv (summed confusion matrix)")
    print(f"  - fold_N_predictions_{args.split}.csv (detailed predictions for each fold)")
    print(f"  - fold_N_class_metrics_{args.split}.csv (per-class metrics for each fold)")
    
    if args.save_plots:
        print(f"  - fold_N_confusion_matrix_{args.split}.png (confusion matrix plots)")
        print(f"  - cv_aggregated_confusion_matrix_{args.split}.png (aggregated plot)")
    
    print("\n💡 Recommendations:")
    
    # Calculate coefficient of variation
    cv_accuracy = (aggregated['overall_accuracy_std'] / aggregated['overall_accuracy_mean']) * 100
    
    if cv_accuracy < 2:
        print("  ✓ Very consistent performance across folds (CV < 2%)")
        print("    → Model is stable and robust")
    elif cv_accuracy < 5:
        print("  ✓ Good consistency across folds (CV < 5%)")
        print("    → Model performance is reliable")
    else:
        print("  ⚠️  High variance across folds (CV ≥ 5%)")
        print("    → Consider:")
        print("       • Increasing training data")
        print("       • More training epochs")
        print("       • Different data augmentation")
        print("       • Checking for data quality issues")
    
    # Identify problematic classes
    low_performing_classes = []
    for i, class_name in enumerate(eval_dataset.classes):
        if aggregated['per_class_f1_mean'][i] < 0.7:
            low_performing_classes.append((class_name, aggregated['per_class_f1_mean'][i]))
    
    if low_performing_classes:
        print(f"\n  ⚠️  Classes with F1 < 0.7:")
        for class_name, f1 in low_performing_classes:
            print(f"     • {class_name}: {f1:.4f}")
        print("    → Review these classes for data quality or add more training samples")


if __name__ == "__main__":
    main()


# Usage Examples:
#
# Evaluate all folds on a common test set:
# python ./scripts/kfold_assess_classification_models.py --models_dir runs/classify --name carabid_cv --data D:/Dropbox/data/carabID/imgs/cv_split --split test
#
# Evaluate on each fold's validation set:
# python assess_classification_cv.py --models_dir runs/classify --name carabid_cv --data D:/Dropbox/data/carabID/imgs/cv_split --split val
#
# Evaluate specific folds with plots:
# python assess_classification_cv.py --models_dir runs/classify --name carabid_cv --data D:/Dropbox/data/carabID/imgs/cv_split --split test --folds 1,2,3 --save-plots
#
# Use last.pt instead of best.pt:
# python assess_classification_cv.py --models_dir runs/classify --name carabid_cv --data D:/Dropbox/data/carabID/imgs/cv_split --split test --weight-type last
