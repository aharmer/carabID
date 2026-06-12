"""
YOLO Classification Model Training Script

This script trains a YOLO classification model using the Ultralytics YOLO library.
It allows you to specify the dataset path, number of epochs, model name, dropout, etc
as command-line arguments.

Usage:
    python train_classification_model.py --data path/to/dataset --epochs 15 --name my_model
    
Example:
    python ./scripts/train_classification_model.py --data D:/Dropbox/data/carabID/imgs/final_classification_set --epochs 30 --dropout 0.2 --lr0 0.001 --name final_carabid_model_11ncls_ep30_autobatch_do02_lr001
    
    python ./scripts/train_classification_model.py --data D:/Dropbox/data/tephritID/imgs/classification_set --epochs 30 --dropout 0.2 --name model-11n-cls_ep30_autobatch_do02
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a YOLO classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the classification dataset directory"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="yolo_classification",
        help="Name for the training run"
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
    
    return parser.parse_args()


def validate_data_path(data_path):
    """Validate that the dataset directory exists and has proper structure."""
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Error: Dataset directory '{data_path}' does not exist!")
        sys.exit(1)
    
    if not data_dir.is_dir():
        print(f"Error: '{data_path}' is not a directory!")
        sys.exit(1)
    
    # Check for train/val subdirectories (typical structure)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists():
        print(f"Warning: No 'train' subdirectory found in '{data_path}'")
        print("Make sure your dataset follows the expected structure:")
        print("dataset/")
        print("├── train/")
        print("│   ├── class1/")
        print("│   └── class2/")
        print("└── val/")
        print("    ├── class1/")
        print("    └── class2/")
    
    return str(data_dir.resolve())


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Validate data path
    data_path = validate_data_path(args.data)
    
    print("="*60)
    print("YOLO Classification Model Training")
    print("="*60)
    print(f"Dataset: {data_path}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Name: {args.name}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Dropout: {args.dropout}")
    print(f"Initial LR: {args.lr0}")
    print(f"Final LR factor: {args.lrf}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Verbose: {args.verbose}")
    print(f"Pretrained: {args.pretrained}")
    print("="*60)
    
    try:
        # Load the model
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)
        
        # Prepare training arguments
        train_args = {
            'data': data_path,
            'epochs': args.epochs,
            'imgsz': args.imgsz,
            'batch': args.batch,
            'verbose': args.verbose,
            'name': args.name,
            'pretrained': args.pretrained,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'lrf': args.lrf
        }
        
        # Add dropout if specified
        if args.dropout > 0:
            train_args['dropout'] = args.dropout
        
        # Add device if specified
        if args.device:
            train_args['device'] = args.device
        
        # Train the model
        print("Starting training...")
        results = model.train(**train_args)
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Results saved to: runs/classify/{args.name}")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nCommon issues:")
        print("1. Make sure you're using a classification model (e.g., yolo11n-cls.pt)")
        print("2. Check that your dataset has the correct structure")
        print("3. Verify that the ultralytics library is installed and up to date")
        sys.exit(1)


if __name__ == "__main__":
    main()
