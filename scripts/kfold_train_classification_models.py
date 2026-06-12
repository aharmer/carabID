"""
YOLO Classification Model Cross-Validation Training Script

This script trains YOLO classification models on all k-folds for cross-validation.
It automatically detects folds, trains a model on each, and aggregates results.

Usage:
    python train_classification_cv.py --data path/to/cv/dataset --epochs 30 --name my_cv_model
    
Example:
    python ./scripts/kfold_train_classification_models.py --data D:/Dropbox/data/carabID/imgs/cv_classification_set --epochs 30 --dropout 0.2 --lr0 0.001 --name carabid_cv_11ncls_ep30_autobatch_do02_lr001
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
from ultralytics import YOLO
import pandas as pd
from datetime import datetime


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLO classification models on all k-folds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the CV dataset directory containing fold_1, fold_2, etc."
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs per fold"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="yolo_cv",
        help="Base name for the training runs (will append _fold1, _fold2, etc.)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-cls.pt",
        help="Model to use for training (use -cls models for classification)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 for auto batch sizing)"
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate for training"
    )
    
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate"
    )
    
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor"
    )
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"],
        help="Optimizer to use for training"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run on (e.g., 0, 1, cpu)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output during training"
    )
    
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Epochs to wait for no improvement before early stopping"
    )
    
    parser.add_argument(
        "--save_best_only",
        action="store_true",
        help="Only save the best model checkpoint for each fold"
    )
    
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Which folds to train (e.g., '1,3,5' or 'all')"
    )
    
    return parser.parse_args()


def detect_folds(data_path: Path) -> List[Path]:
    """
    Detect all fold directories in the dataset path.
    
    Args:
        data_path: Path to the CV dataset directory
    
    Returns:
        List of fold directory paths
    """
    fold_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    
    if not fold_dirs:
        print(f"Error: No fold directories found in '{data_path}'")
        print("Expected directory structure:")
        print("cv_dataset/")
        print("├── fold_1/")
        print("│   ├── train/")
        print("│   └── val/")
        print("├── fold_2/")
        print("│   ├── train/")
        print("│   └── val/")
        print("└── ...")
        sys.exit(1)
    
    return fold_dirs


def validate_fold_structure(fold_dir: Path) -> Tuple[bool, str]:
    """
    Validate that a fold directory has the proper structure.
    
    Args:
        fold_dir: Path to fold directory
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    train_dir = fold_dir / "train"
    val_dir = fold_dir / "val"
    
    if not train_dir.exists():
        return False, f"Missing 'train' directory in {fold_dir.name}"
    
    if not val_dir.exists():
        return False, f"Missing 'val' directory in {fold_dir.name}"
    
    # Check if directories have class subdirectories
    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    val_classes = [d for d in val_dir.iterdir() if d.is_dir()]
    
    if not train_classes:
        return False, f"No class directories found in {fold_dir.name}/train"
    
    if not val_classes:
        return False, f"No class directories found in {fold_dir.name}/val"
    
    return True, ""


def train_fold(fold_dir: Path, fold_num: int, args, model_path: str) -> Dict:
    """
    Train a model on a single fold.
    
    Args:
        fold_dir: Path to fold directory
        fold_num: Fold number
        args: Command-line arguments
        model_path: Path to model weights
    
    Returns:
        Dictionary containing training results and metrics
    """
    print("\n" + "="*70)
    print(f"TRAINING FOLD {fold_num}")
    print("="*70)
    print(f"Fold directory: {fold_dir}")
    print(f"Model: {model_path}")
    print(f"Epochs: {args.epochs}")
    print("="*70 + "\n")
    
    try:
        # Load the model (fresh weights for each fold)
        model = YOLO(model_path)
        
        # Prepare training arguments
        fold_name = f"{args.name}_fold{fold_num}"
        train_args = {
            'data': str(fold_dir),
            'epochs': args.epochs,
            'imgsz': args.imgsz,
            'batch': args.batch,
            'verbose': args.verbose,
            'name': fold_name,
            'pretrained': args.pretrained,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'lrf': args.lrf,
            'patience': args.patience,
        }
        
        # Add dropout if specified
        if args.dropout > 0:
            train_args['dropout'] = args.dropout
        
        # Add device if specified
        if args.device:
            train_args['device'] = args.device
        
        # Add save_period if save_best_only
        if args.save_best_only:
            train_args['save_period'] = -1  # Only save best and last
        
        # Train the model
        print(f"Starting training for fold {fold_num}...")
        results = model.train(**train_args)
        
        # Extract key metrics from results
        metrics = extract_metrics(results, fold_num)
        
        print(f"\n✓ Fold {fold_num} training completed!")
        print(f"  Best metrics:")
        for key, value in metrics.items():
            if key.startswith('best_'):
                print(f"    {key}: {value:.4f}")
        
        return {
            'fold': fold_num,
            'status': 'success',
            'metrics': metrics,
            'save_dir': str(Path('runs/classify') / fold_name)
        }
        
    except Exception as e:
        print(f"\n✗ Error training fold {fold_num}: {str(e)}")
        return {
            'fold': fold_num,
            'status': 'failed',
            'error': str(e),
            'metrics': {}
        }


def extract_metrics(results, fold_num: int) -> Dict:
    """
    Extract key metrics from training results.
    
    Args:
        results: YOLO training results object
        fold_num: Fold number
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'fold': fold_num,
    }
    
    try:
        # Try to get metrics from results object
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            
            # Common classification metrics
            metric_keys = [
                'metrics/accuracy_top1',
                'metrics/accuracy_top5',
                'train/loss',
                'val/loss',
            ]
            
            for key in metric_keys:
                if key in results_dict:
                    clean_key = key.replace('metrics/', '').replace('/', '_')
                    metrics[f'final_{clean_key}'] = float(results_dict[key])
        
        # Try to get best metrics from validator
        if hasattr(results, 'best'):
            metrics['best_accuracy_top1'] = float(results.best)
        
        # Try to read from CSV results file
        try:
            save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else None
            if save_dir and (save_dir / 'results.csv').exists():
                df = pd.read_csv(save_dir / 'results.csv')
                df.columns = df.columns.str.strip()
                
                # Get final epoch metrics
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    
                    # Map common column names
                    col_mapping = {
                        'metrics/accuracy_top1': 'final_accuracy_top1',
                        'metrics/accuracy_top5': 'final_accuracy_top5',
                        'val/loss': 'final_val_loss',
                        'train/loss': 'final_train_loss',
                    }
                    
                    for csv_col, metric_name in col_mapping.items():
                        if csv_col in df.columns:
                            metrics[metric_name] = float(last_row[csv_col])
                
                # Get best metrics (minimum loss, maximum accuracy)
                if 'metrics/accuracy_top1' in df.columns:
                    metrics['best_accuracy_top1'] = float(df['metrics/accuracy_top1'].max())
                
                if 'metrics/accuracy_top5' in df.columns:
                    metrics['best_accuracy_top5'] = float(df['metrics/accuracy_top5'].max())
                
                if 'val/loss' in df.columns:
                    metrics['best_val_loss'] = float(df['val/loss'].min())
        
        except Exception as e:
            print(f"  Warning: Could not read detailed metrics from CSV: {e}")
    
    except Exception as e:
        print(f"  Warning: Could not extract all metrics: {e}")
    
    return metrics


def aggregate_results(fold_results: List[Dict]) -> Dict:
    """
    Aggregate results across all folds.
    
    Args:
        fold_results: List of fold result dictionaries
    
    Returns:
        Dictionary with aggregated statistics
    """
    successful_folds = [r for r in fold_results if r['status'] == 'success']
    
    if not successful_folds:
        return {
            'status': 'failed',
            'message': 'No folds completed successfully'
        }
    
    # Collect metrics from all folds
    all_metrics = {}
    for fold_result in successful_folds:
        for metric_name, value in fold_result['metrics'].items():
            if metric_name != 'fold' and isinstance(value, (int, float)):
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
    
    # Calculate statistics
    aggregated = {
        'n_folds_total': len(fold_results),
        'n_folds_successful': len(successful_folds),
        'n_folds_failed': len(fold_results) - len(successful_folds),
    }
    
    for metric_name, values in all_metrics.items():
        if values:
            aggregated[f'{metric_name}_mean'] = sum(values) / len(values)
            aggregated[f'{metric_name}_std'] = (sum((x - aggregated[f'{metric_name}_mean'])**2 for x in values) / len(values))**0.5
            aggregated[f'{metric_name}_min'] = min(values)
            aggregated[f'{metric_name}_max'] = max(values)
    
    return aggregated


def save_cv_summary(fold_results: List[Dict], aggregated: Dict, args, output_dir: Path):
    """
    Save cross-validation summary to files.
    
    Args:
        fold_results: List of fold result dictionaries
        aggregated: Aggregated statistics
        args: Command-line arguments
        output_dir: Directory to save summary files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'fold_results': fold_results,
        'aggregated': aggregated
    }
    
    json_path = output_dir / 'cv_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved detailed summary to: {json_path}")
    
    # Create readable text summary
    text_path = output_dir / 'cv_summary.txt'
    with open(text_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CROSS-VALIDATION TRAINING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.data}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Epochs per fold: {args.epochs}\n")
        f.write(f"Total folds: {aggregated['n_folds_total']}\n")
        f.write(f"Successful folds: {aggregated['n_folds_successful']}\n")
        f.write(f"Failed folds: {aggregated['n_folds_failed']}\n\n")
        
        f.write("="*70 + "\n")
        f.write("PER-FOLD RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for result in fold_results:
            f.write(f"Fold {result['fold']}:\n")
            f.write(f"  Status: {result['status']}\n")
            
            if result['status'] == 'success':
                f.write(f"  Save directory: {result['save_dir']}\n")
                f.write("  Metrics:\n")
                for key, value in result['metrics'].items():
                    if key != 'fold' and isinstance(value, (int, float)):
                        f.write(f"    {key}: {value:.4f}\n")
            else:
                f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
            
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("AGGREGATED STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        # Group metrics by type
        metric_groups = {}
        for key in aggregated.keys():
            if key.endswith('_mean'):
                base_metric = key[:-5]
                metric_groups[base_metric] = {}
        
        for base_metric in metric_groups.keys():
            for suffix in ['mean', 'std', 'min', 'max']:
                key = f'{base_metric}_{suffix}'
                if key in aggregated:
                    metric_groups[base_metric][suffix] = aggregated[key]
        
        for metric_name, stats in sorted(metric_groups.items()):
            f.write(f"{metric_name}:\n")
            f.write(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n\n")
    
    print(f"✓ Saved text summary to: {text_path}")
    
    # Create CSV for easy analysis
    try:
        rows = []
        for result in fold_results:
            if result['status'] == 'success':
                row = {'fold': result['fold'], 'status': result['status']}
                row.update(result['metrics'])
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = output_dir / 'cv_results.csv'
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved CSV results to: {csv_path}")
    except Exception as e:
        print(f"Warning: Could not save CSV: {e}")


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Validate and resolve data path
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset directory '{args.data}' does not exist!")
        sys.exit(1)
    
    data_path = data_path.resolve()
    
    # Detect fold directories
    print("="*70)
    print("YOLO CLASSIFICATION CROSS-VALIDATION TRAINING")
    print("="*70)
    print(f"Dataset: {data_path}")
    print(f"Model: {args.model}")
    print(f"Epochs per fold: {args.epochs}")
    
    fold_dirs = detect_folds(data_path)
    print(f"Detected {len(fold_dirs)} folds: {[d.name for d in fold_dirs]}")
    
    # Filter folds if specified
    if args.folds.lower() != 'all':
        try:
            selected_folds = [int(f.strip()) for f in args.folds.split(',')]
            fold_dirs = [d for d in fold_dirs if int(d.name.split('_')[1]) in selected_folds]
            print(f"Training selected folds: {[d.name for d in fold_dirs]}")
        except Exception as e:
            print(f"Error parsing --folds argument: {e}")
            sys.exit(1)
    
    if not fold_dirs:
        print("Error: No folds to train!")
        sys.exit(1)
    
    # Validate fold structures
    print("\nValidating fold structures...")
    for fold_dir in fold_dirs:
        is_valid, error_msg = validate_fold_structure(fold_dir)
        if not is_valid:
            print(f"✗ {error_msg}")
            sys.exit(1)
        print(f"✓ {fold_dir.name} structure is valid")
    
    print("\nConfiguration:")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Initial LR: {args.lr0}")
    print(f"  Final LR factor: {args.lrf}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Device: {args.device if args.device else 'auto'}")
    print(f"  Patience: {args.patience}")
    print("="*70)
    
    # Confirm before starting
    response = input(f"\nProceed with training {len(fold_dirs)} folds? (y/n): ").lower()
    if response != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    # Train each fold
    fold_results = []
    start_time = datetime.now()
    
    for i, fold_dir in enumerate(fold_dirs, 1):
        fold_num = int(fold_dir.name.split('_')[1])
        
        result = train_fold(fold_dir, fold_num, args, args.model)
        fold_results.append(result)
        
        print(f"\nProgress: {i}/{len(fold_dirs)} folds completed")
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)
    
    aggregated = aggregate_results(fold_results)
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Successful folds: {aggregated['n_folds_successful']}/{aggregated['n_folds_total']}")
    print(f"Failed folds: {aggregated['n_folds_failed']}/{aggregated['n_folds_total']}")
    
    print("\nKey Metrics (Mean ± Std):")
    metric_priority = ['best_accuracy_top1', 'best_accuracy_top5', 'best_val_loss', 
                      'final_accuracy_top1', 'final_val_loss']
    
    for metric in metric_priority:
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        if mean_key in aggregated and std_key in aggregated:
            print(f"  {metric}: {aggregated[mean_key]:.4f} ± {aggregated[std_key]:.4f}")
    
    # Save summary files
    summary_dir = Path('runs/classify') / f"{args.name}_cv_summary"
    save_cv_summary(fold_results, aggregated, args, summary_dir)
    
    print("\n" + "="*70)
    print("✓ CROSS-VALIDATION TRAINING COMPLETED!")
    print("="*70)
    print(f"\nResults saved to: {summary_dir}")
    print("\nIndividual fold models saved to:")
    for result in fold_results:
        if result['status'] == 'success':
            print(f"  - {result['save_dir']}")
    
    print("\n💡 Next steps:")
    print("  1. Review cv_summary.txt for detailed results")
    print("  2. Compare fold performance in cv_results.csv")
    print("  3. Use the best performing fold model, or ensemble all models")
    print("  4. Evaluate on test set if you have one")


if __name__ == "__main__":
    main()
