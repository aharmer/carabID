"""
YOLO Detection Model Training Script

This script trains a YOLO detection model using the Ultralytics YOLO library.
It allows you to specify the data.yaml path, number of epochs, model name, etc
as command-line arguments.

Usage:
    python train_detection_model.py --data path/to/data.yaml --epochs 30 --name my_model
    
Example:
    python ./scripts/train_detection_model.py --data D:/Dropbox/data/carabID/imgs/detection_set/data.yaml --epochs 30 --name model_11n_ep30_autobatch --single-cls
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a YOLO detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the data.yaml file"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="yolo_training",
        help="Name for the training run"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Model to use for training"
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
        "--single-cls",
        action="store_true",
        default=True,
        help="Train as single-class dataset"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output during training"
    )
    
    return parser.parse_args()


def validate_data_path(data_path):
    """Validate that the data.yaml file exists."""
    data_file = Path(data_path)
    if not data_file.exists():
        print(f"Error: Data file '{data_path}' does not exist!")
        sys.exit(1)
    
    if not data_file.suffix.lower() in ['.yaml', '.yml']:
        print(f"Warning: Data file '{data_path}' doesn't have a .yaml/.yml extension")
    
    return str(data_file.resolve())


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Validate data path
    data_path = validate_data_path(args.data)
    
    print("="*50)
    print("YOLO Detection Model Training")
    print("="*50)
    print(f"Data file: {data_path}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Name: {args.name}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Single class: {args.single_cls}")
    print(f"Verbose: {args.verbose}")
    print("="*50)
    
    try:
        # Load the model
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)
        
        # Train the model
        print("Starting training...")
        results = model.train(
            data=data_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            single_cls=args.single_cls,
            batch=args.batch,
            verbose=args.verbose,
            name=args.name
        )
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Results saved to: runs/detect/{args.name}")
        print("="*50)
        
        return results
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
