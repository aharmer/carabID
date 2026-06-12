import os
import csv
from pathlib import Path
from collections import defaultdict

def count_images_per_class(dataset_path, output_csv='class_distribution.csv'):
    """
    Count the number of images in each class of a classification dataset.
    
    Args:
        dataset_path: Path to the dataset root directory
        output_csv: Output CSV filename
    
    Returns:
        Dictionary with class names and image counts
    """
    dataset_path = Path(dataset_path)
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # Dictionary to store counts
    class_counts = defaultdict(int)
    
    # Walk through the dataset directory
    # Assumes structure: dataset_path/class_name/images
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            
            # Count images in this class directory
            image_count = sum(
                1 for f in class_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in image_extensions
            )
            
            if image_count > 0:
                class_counts[class_name] = image_count
    
    # Sort by class name for consistent output
    sorted_counts = dict(sorted(class_counts.items()))
    
    # Calculate total
    total_images = sum(sorted_counts.values())
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class', 'Image Count', 'Percentage'])
        
        for class_name, count in sorted_counts.items():
            percentage = (count / total_images * 100) if total_images > 0 else 0
            writer.writerow([class_name, count, f'{percentage:.2f}%'])
        
        # Add total row
        writer.writerow(['TOTAL', total_images, '100.00%'])
    
    print(f"Analysis complete! Results saved to '{output_csv}'")
    print(f"\nSummary:")
    print(f"Total classes: {len(sorted_counts)}")
    print(f"Total images: {total_images}")
    print(f"\nClass distribution:")
    for class_name, count in sorted_counts.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.2f}%)")
    
    return sorted_counts

# Example usage
if __name__ == "__main__":
    # Update this path to your dataset location
    # Expected structure: dataset_path/class1/, dataset_path/class2/, etc.
    dataset_path = "D:/Dropbox/data/carabID/imgs/c1/train"
    
    # Optional: specify custom output filename
    output_file = "D:/Dropbox/data/carabID/runs/classify/class_distribution_original.csv"
    
    try:
        counts = count_images_per_class(dataset_path, output_file)
    except FileNotFoundError:
        print(f"Error: Dataset path '{dataset_path}' not found.")
        print("Please update the dataset_path variable with your actual dataset location.")
    except Exception as e:
        print(f"An error occurred: {e}")    