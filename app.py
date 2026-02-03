# Import required libraries
import warnings
import logging
import os
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings if present
logging.getLogger('ultralytics').setLevel(logging.ERROR)

import PIL
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from ultralytics import YOLO

def contrast_stretch(input_image):
    """
    Apply contrast stretching by mapping pixel intensities so that the
    2nd percentile becomes 0 and the 98th percentile becomes 255.
    This approach is more robust to outliers compared to using min/max values.
    """
    p2 = np.percentile(input_image, 2)
    p98 = np.percentile(input_image, 98)
    
    if p98 - p2 == 0:
        return input_image
    
    stretched = (input_image - p2) * (255.0 / (p98 - p2))
    stretched = np.clip(stretched, 0, 255)
    
    return stretched.astype(np.uint8)


def format_genus_name(class_name):
    """Format the predicted class name for display."""
    parts = class_name.split("_")
    parts[0] = parts[0].capitalize()
    return " ".join(parts)


def get_classification_predictions(results, top_k=3, conf_threshold=0.25):
    """
    Extract top K predictions from YOLO classification results.
    Returns list of tuples: (genus_name, confidence_percentage)
    """
    predictions = []
    
    # Check if results contain classification probabilities
    if hasattr(results[0], 'probs') and results[0].probs is not None:
        probs = results[0].probs.data.cpu().numpy()
        for cls_idx, prob in enumerate(probs):
            if prob >= conf_threshold:
                cls_name = results[0].names[cls_idx]
                formatted_name = format_genus_name(cls_name)
                confidence = round(float(prob) * 100, 1)
                predictions.append((formatted_name, confidence))
    
    # Sort by confidence (descending) and return top K
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_k]


def crop_beetle_parts(image, detection_results):
    """
    Crop the detected beetle parts (thorax/elytra) from the image.
    Returns the largest detected region (assuming it's the beetle).
    """
    if not detection_results[0].boxes or len(detection_results[0].boxes) == 0:
        return None
    
    # Get the box with highest confidence (or largest area)
    boxes = detection_results[0].boxes
    
    # Find the largest box by area
    max_area = 0
    best_box = None
    
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_box = (x1, y1, x2, y2)
    
    if best_box is None:
        return None
    
    # Crop the image
    x1, y1, x2, y2 = best_box
    cropped = image.crop((x1, y1, x2, y2))
    
    return cropped


# Load models and data
detection_model_path = './static/detection.pt'  # Detection model (thorax/elytra)
classification_model_path = './static/classification.pt'  # Classification model
df = pd.read_csv('./static/class_counts.csv')

if os.path.exists(file_path):
    size = os.path.getsize(file_path)
    st.write(f"File exists! Size: {size} bytes")
    if size < 2000:
        st.error("⚠️ The file is too small! This is likely a Git LFS pointer, not the actual model.")
else:
    st.error("File definitely not found at this path.")
    
try:
    detection_model = YOLO(detection_model_path)
    # Update this path if your classification model is named differently
    # If it's the same model doing both tasks, you can use: classification_model = detection_model
    classification_model = YOLO(classification_model_path)  
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {detection_model_path}")
    st.error(ex)


# Page configuration
st.set_page_config(
    page_title="CarabID",
    page_icon="./static/carabid_icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create tabs
tab1, tab2 = st.tabs(["App", "About"])

with tab1:
    # Sidebar
    with st.sidebar:
        st.image('./static/carabid_icon.png')   

        st.header("Identify a ground beetle")
        
        # File uploader
        source_imgs = st.file_uploader(
            "Upload images...", 
            type=("jpg", "jpeg", "png", 'bmp', 'webp'),
            accept_multiple_files=True
        )

        # Confidence threshold slider
        detection_confidence = float(st.slider(
            "Detection confidence threshold", 
            25, 100, 50
        )) / 100
        
        classification_confidence = float(st.slider(
            "Classification confidence threshold", 
            1, 100, 25
        )) / 100
        
        # Number of predictions to show
        top_k = st.slider(
            "Number of top predictions to show",
            1, 5, 3
        )

    # Main page
    st.title("CarabID")
    st.caption('Upload photos of ground beetles.')
    st.caption('Then click the :blue[Identify] button and check the results.')

    # Identify button
    if st.sidebar.button('Identify'):
        if not source_imgs:
            st.error("Please upload at least one image first!")
        else:
            with st.spinner('Analyzing images...'):
                
                # GRID CONFIGURATION
                # This sets how many columns (tiles) per row
                COLS_PER_ROW = 4 
                cols = None
                
                for i, source_img in enumerate(source_imgs):
                    
                    # Create a new row of columns every 4 images
                    if i % COLS_PER_ROW == 0:
                        cols = st.columns(COLS_PER_ROW)
                    
                    # Select the specific column for this image
                    current_col = cols[i % COLS_PER_ROW]
                    
                    with current_col:
                        # Add a visual card-like container
                        with st.container(border=True):
                            st.caption(f"{source_img.name}")
                            
                            try:
                                # 1. Load and Preprocess
                                uploaded_image = PIL.Image.open(source_img)
                                if uploaded_image.mode != 'RGB':
                                    uploaded_image = uploaded_image.convert('RGB')
                                
                                prepped_image = uploaded_image.resize((640, 640))
                                prepped_image = contrast_stretch(prepped_image)

                                # 2. Stage 1: Detection
                                detection_results = detection_model.predict(
                                    prepped_image, 
                                    conf=detection_confidence, 
                                    verbose=False
                                )
                                
                                # Check if any beetle parts were detected
                                if not detection_results[0].boxes or len(detection_results[0].boxes) == 0:
                                    st.image(uploaded_image, use_container_width=True)
                                    st.warning("No beetle detected")
                                else:
                                    # Plot detection results
                                    detection_plotted = detection_results[0].plot()[:, :, ::1]
                                    
                                    # Stage 2: Crop and Classify
                                    cropped_beetle = crop_beetle_parts(uploaded_image, detection_results)
                                    
                                    if cropped_beetle is None:
                                        st.image(detection_plotted, use_container_width=True)
                                        st.error("Crop failed")
                                    else:
                                        cropped_resized = cropped_beetle.resize((640, 640))
                                        
                                        classification_results = classification_model.predict(
                                            cropped_resized,
                                            conf=0.01,
                                            verbose=False
                                        )
                                        
                                        top_predictions = get_classification_predictions(
                                            classification_results, 
                                            top_k=top_k, 
                                            conf_threshold=classification_confidence
                                        )
                                        
                                        # Display Image (Tiled size)
                                        st.image(detection_plotted, use_container_width=True)
                                        
                                        if top_predictions:
                                            # Highlight top result
                                            top_genus, top_conf = top_predictions[0]
                                            st.markdown(f"**{top_genus}**")
                                            st.progress(top_conf / 100)
                                            st.caption(f"{top_conf}% confidence")
                                            
                                            # Details expander for other guesses
                                            with st.expander("Details"):
                                                for idx, (genus, conf) in enumerate(top_predictions):
                                                    st.text(f"{idx+1}. {genus} ({conf}%)")
                                        else:
                                            st.warning("Low confidence")

                            except Exception as ex:
                                st.error(f"Error: {ex}")

with tab2:
    st.header("About the project")
    st.markdown("This tool was developed using YOLOv11 models from the Ultralytics Python library Ver. 8.3.166 (Jocher et al., 2023). We trained a two-stage detection and classification pipeline. The first stage uses a detection model (YOLOv11n) with the task of detecting the beetle (thorax and elytra only) in images, with all individuals treated as one class. The second stage uses a classification model (YOLOv11n-cls) with the task of classifying carabids to genus. Full details can be found in the associated publication: Gong, Harmer and Ward, in review.\n")
    st.markdown("Images should be prepared as per the example below. The thorax and elytra should be visible in the dorsal plain, the orientation of the head is not important. Any camera can be used, but try to fill the frame with the beetle.")
    st.image('./static/example_image.jpg', width = 350)
    st.markdown("\nThe classification model was trained on the genera below. Image counts are for the original dataset before augmentation.\n")
    st.dataframe(df, height=600)

