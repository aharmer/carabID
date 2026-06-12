import os
import shutil
import yaml
from pathlib import Path
import argparse
from collections import OrderedDict

def load_yaml_file(yaml_path):
    """Load and parse a YAML file"""
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml_file(data, yaml_path):
    """Save data to a YAML file"""
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

def extract_class_names(data):
    """Extract class names from YAML data, handling both list and dict formats"""
    names = data.get('names', [])
    
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    else:
        return names

def create_unified_class_mapping(classes1, classes2):
    """
    Create a unified class list and mapping dictionaries for both datasets
    Returns:
        - unified_classes: List of all unique class names
        - mapping1: Dict mapping old indices to new indices for dataset 1
        - mapping2: Dict mapping old indices to new indices for dataset 2
    """
    # Create ordered set of unique class names (preserving order from dataset1 first)
    unified_classes = []
    seen_classes = set()
    
    # Add classes from dataset 1 first
    for class_name in classes1:
        if class_name not in seen_classes:
            unified_classes.append(class_name)
            seen_classes.add(class_name)
    
    # Add new classes from dataset 2
    for class_name in classes2:
        if class_name not in seen_classes:
            unified_classes.append(class_name)
            seen_classes.add(class_name)
    
    # Create mapping dictionaries
    mapping1 = {}
    mapping2 = {}
    
    # Create mapping for dataset 1
    for old_idx, class_name in enumerate(classes1):
        new_idx = unified_classes.index(class_name)
        mapping1[old_idx] = new_idx
    
    # Create mapping for dataset 2
    for old_idx, class_name in enumerate(classes2):
        new_idx = unified_classes.index(class_name)
        mapping2[old_idx] = new_idx
    
    return unified_classes, mapping1, mapping2

def update_label_file_with_mapping(label_path, class_mapping):
    """Update class indices in a label file using a mapping dictionary"""
    updated_lines = []
    
    with open(label_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) >= 9:  # class_index + 8 coordinates
                    # Update class index using mapping
                    old_class = int(parts[0])
                    if old_class in class_mapping:
                        new_class = class_mapping[old_class]
                        parts[0] = str(new_class)
                        updated_lines.append(' '.join(parts))
                    else:
                        print(f"Warning: Class index {old_class} not found in mapping for file {label_path}")
                        updated_lines.append(line)  # Keep original if mapping not found
                else:
                    # Keep malformed lines as is
                    updated_lines.append(line)
    
    # Write updated content back to file
    with open(label_path, 'w') as file:
        file.write('\n'.join(updated_lines))
        if updated_lines:  # Add final newline if file is not empty
            file.write('\n')

def copy_dataset_files_with_mapping(src_dataset_path, dest_dataset_path, class_mapping, dataset_name):
    """Copy dataset files and update labels using class mapping"""
    src_train_path = os.path.join(src_dataset_path, 'train')
    dest_train_path = os.path.join(dest_dataset_path, 'train')
    
    # Create destination directories
    os.makedirs(os.path.join(dest_train_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_train_path, 'labels'), exist_ok=True)
    
    # Copy images
    src_images_path = os.path.join(src_train_path, 'images')
    dest_images_path = os.path.join(dest_train_path, 'images')
    
    if os.path.exists(src_images_path):
        for filename in os.listdir(src_images_path):
            src_file = os.path.join(src_images_path, filename)
            dest_file = os.path.join(dest_images_path, filename)
            
            # Handle potential filename conflicts
            counter = 1
            base_name, ext = os.path.splitext(filename)
            original_filename = filename
            while os.path.exists(dest_file):
                new_filename = f"{base_name}_{dataset_name}_{counter}{ext}"
                dest_file = os.path.join(dest_images_path, new_filename)
                filename = new_filename
                counter += 1
            
            shutil.copy2(src_file, dest_file)
            if filename != original_filename:
                print(f"Copied image: {original_filename} -> {filename}")
            else:
                print(f"Copied image: {filename}")
    
    # Copy and update labels
    src_labels_path = os.path.join(src_train_path, 'labels')
    dest_labels_path = os.path.join(dest_train_path, 'labels')
    
    if os.path.exists(src_labels_path):
        for filename in os.listdir(src_labels_path):
            if filename.endswith('.txt'):
                src_file = os.path.join(src_labels_path, filename)
                dest_file = os.path.join(dest_labels_path, filename)
                
                # Handle potential filename conflicts
                counter = 1
                base_name, ext = os.path.splitext(filename)
                original_filename = filename
                while os.path.exists(dest_file):
                    new_filename = f"{base_name}_{dataset_name}_{counter}{ext}"
                    dest_file = os.path.join(dest_labels_path, new_filename)
                    filename = new_filename
                    counter += 1
                
                # Copy the file first
                shutil.copy2(src_file, dest_file)
                
                # Update class indices using mapping
                update_label_file_with_mapping(dest_file, class_mapping)
                
                if filename != original_filename:
                    print(f"Copied and updated label: {original_filename} -> {filename}")
                else:
                    print(f"Copied and updated label: {filename}")

def merge_datasets_with_class_deduplication(dataset1_path, dataset2_path, output_path):
    """Main function to merge two YOLO OBB datasets with class deduplication"""
    
    # Validate input paths
    if not os.path.exists(dataset1_path):
        raise FileNotFoundError(f"Dataset 1 path not found: {dataset1_path}")
    if not os.path.exists(dataset2_path):
        raise FileNotFoundError(f"Dataset 2 path not found: {dataset2_path}")
    
    # Load data.yaml files
    yaml1_path = os.path.join(dataset1_path, 'data.yaml')
    yaml2_path = os.path.join(dataset2_path, 'data.yaml')
    
    if not os.path.exists(yaml1_path):
        raise FileNotFoundError(f"data.yaml not found in dataset 1: {yaml1_path}")
    if not os.path.exists(yaml2_path):
        raise FileNotFoundError(f"data.yaml not found in dataset 2: {yaml2_path}")
    
    data1 = load_yaml_file(yaml1_path)
    data2 = load_yaml_file(yaml2_path)
    
    # Extract class information
    classes1 = extract_class_names(data1)
    classes2 = extract_class_names(data2)
    
    print(f"Dataset 1 classes ({len(classes1)}): {classes1}")
    print(f"Dataset 2 classes ({len(classes2)}): {classes2}")
    
    # Create unified class mapping
    unified_classes, mapping1, mapping2 = create_unified_class_mapping(classes1, classes2)
    
    print(f"\nUnified classes ({len(unified_classes)}): {unified_classes}")
    print(f"Dataset 1 class mapping: {mapping1}")
    print(f"Dataset 2 class mapping: {mapping2}")
    
    # Find common and unique classes
    common_classes = set(classes1) & set(classes2)
    unique_to_1 = set(classes1) - set(classes2)
    unique_to_2 = set(classes2) - set(classes1)
    
    print(f"\nClass analysis:")
    print(f"Common classes ({len(common_classes)}): {sorted(common_classes)}")
    print(f"Unique to dataset 1 ({len(unique_to_1)}): {sorted(unique_to_1)}")
    print(f"Unique to dataset 2 ({len(unique_to_2)}): {sorted(unique_to_2)}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Copy dataset 1 files with mapping
    print(f"\nCopying Dataset 1 with class remapping...")
    copy_dataset_files_with_mapping(dataset1_path, output_path, mapping1, "ds1")
    
    # Copy dataset 2 files with mapping
    print(f"\nCopying Dataset 2 with class remapping...")
    copy_dataset_files_with_mapping(dataset2_path, output_path, mapping2, "ds2")
    
    # Create merged data.yaml
    merged_data = {
        'path': output_path,
        'train': 'train/images',
        'val': 'train/images',  # You might want to create a separate validation set
        'nc': len(unified_classes),
        'names': unified_classes
    }
    
    # Save merged data.yaml
    merged_yaml_path = os.path.join(output_path, 'data.yaml')
    save_yaml_file(merged_data, merged_yaml_path)
    
    print(f"\nMerged dataset created successfully!")
    print(f"Output path: {output_path}")
    print(f"Total unique classes: {len(unified_classes)}")
    print(f"New data.yaml saved to: {merged_yaml_path}")
    
    return {
        'unified_classes': unified_classes,
        'mapping1': mapping1,
        'mapping2': mapping2,
        'common_classes': common_classes,
        'unique_to_1': unique_to_1,
        'unique_to_2': unique_to_2
    }

def main():
    parser = argparse.ArgumentParser(description='Merge two YOLO datasets with class deduplication')
    parser.add_argument('dataset1', help='Path to first dataset directory')
    parser.add_argument('dataset2', help='Path to second dataset directory')
    parser.add_argument('output', help='Path to output merged dataset directory')
    
    args = parser.parse_args()
    
    try:
        result = merge_datasets_with_class_deduplication(args.dataset1, args.dataset2, args.output)
        print(f"\nMerge completed successfully!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
    

# Usage
# python ./scripts/combine_datasets.py D:/Dropbox/data/carabID/imgs/carabids_genus_v3-6 D:/Dropbox/data/carabID/imgs/carabidae_extra-1 D:/Dropbox/data/carabID/imgs/combined
