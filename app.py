# Import required libraries
import PIL
from PIL import Image, ImageOps
import numpy as np
import time
import keyboard
import psutil
import os
import streamlit as st
from ultralytics import YOLO


def constrast_stretch(inputImage):
    img = inputImage
    outputImage = Image.new('L',img.size)

    width, height = img.size

    minIntensity = np.percentile(img, 2)
    maxIntensity = np.percentile(img, 98)

    for x in range(width):
     for y in range(height):
        intensity = img.getpixel((x,y))
        minIntensity = min(minIntensity, intensity)
        maxIntensity = max(maxIntensity, intensity)

    for x in range(width):
        for y in range(height):
            intensity = img.getpixel((x,y))
            newIntensity = 255 * ((intensity - minIntensity) / (maxIntensity - minIntensity))
            outputImage.putpixel((x,y), int(newIntensity))

    return outputImage

# Replace the relative path to model file
model_path = './static/best.pt'

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


# Setting page layout
st.set_page_config(
    page_title = "CarabID",  # Setting page title
    page_icon = "./static/beetle_icon.png",     # Setting page icon
    layout = "wide",      # Setting layout to wide
    initial_sidebar_state = "expanded",    # Expanding sidebar by default   
)

tab1, tab2 = st.tabs(["About", "App"])

with tab1:
    st.image('./static/beetle_icon.png')
    st.header("About the project")
    st.markdown("The identification model was trained on the following taxa:")
    st.markdown("*Actenonyx*, *Agonocheila*, *Allocinopus*, *Amarophilus*, *Amarotypus*, *Amaroxenus*, *Anisodactylus*, *Anomotarus*, *Aulacopodus*, *Bembidion*, *Cerabilia*, *Clivina*, *Ctenognathus*, *Demetrida*, *Dicrochile*, *Diglymma*, *Dromius*, *Duvaliomimus*, *Egadroma*, *Euthenarus*, *Gaioxenus*, *Gnathaphanus*, *Gourlayia*, *Hakaharpalus*, *Haplanister*, *Harpalus*, *Holcaspis*, *Hypharpax*, *Kenodactylus*, *Kettlotrechus*, *Kiwiplatynus*, *Kiwitachys*, *Kiwitrechus*, *Kupeharpalus*, *Kupeplatynus*, *Kupetrechus*, *Laemostenus*, *Lecanomerus*, *Loxomerus*, *Maoriharpalus*, *Maoripamborus*, *Maoritrechus*, *Maungazolus*, *Meclothrax*, *Mecodema*, *Megadromus*, *Meonochilus*, *Metaglymma*, *Molopsida*, *Neanops*, *Neocicindela*, *Neoferonia*, *Nesamblyops*, *Notagonum*, *Oarotrechus*, *Oopterus*, *Oregus*, *Parabaris*, *Paratachys*, *Pelodiaetus*, *Pentagonica*, *Pericompsus*, *Perigona*, *Philophlorus*, *Pholeodytes*, *Physolaesthus*, *Platynus*, *Plocamostethus*, *Prosopogmus*, *Prosphodrus*, *Psegmatopterus*, *Pseudoopterus*, *Rhytisternus*, *Scototrechus*, *Selenochilus*, *Syllectus*, *Synteratus*, *Tarastethus*, *Trichopsida*, *Trigonothops*, *Triplosarus*, *Tuiharpalus*, *Tuiplatynus*, *Zeanillus*, *Zecicindela*, *Zeopoecilus*, *Zolus*.")

with tab2:
    # Creating sidebar
    with st.sidebar:
        st.header("Find a ground beetle")     # Adding header to sidebar
        # Adding file uploader to sidebar for selecting images
        source_img = st.file_uploader(
            "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        # Model Options
        confidence = float(st.slider(
            "Set minimum prediction confidence", 25, 100, 98)) / 100


    # Creating main page heading 
    st.image('./static/beetle_icon.png')
    st.title("CarabID")
    st.caption('Upload a photo of a ground beetle.')
    st.caption('Then click the :blue[Identify] button and check the result.')

    # Creating two columns on the main page
    col1, col2 = st.columns(2)

    # Adding image to the first column if image is uploaded
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        prepped_image = uploaded_image.resize((640,640))
        prepped_image = ImageOps.grayscale(prepped_image)
        prepped_image = constrast_stretch(prepped_image)

    with st.sidebar:
        if source_img:
            # Display uploaded image
            st.image(uploaded_image,
                     caption="Uploaded Image"
                     )

    if st.sidebar.button('Identify'):
        try:    
            res = model.predict(prepped_image, conf=confidence)
            
            if res[0].boxes:
                res_plotted = res[0].plot()[:, :, ::-1]
                
                # Retrieve all predicted classes and their confidence scores
                predictions = []
                for box in res[0].boxes:
                    cls_idx = int(box.cls.cpu().item())  # Extract class index as a scalar
                    cls_name = res[0].names[cls_idx]  # Get class name
                    cls_name_parts = cls_name.split("_")
                    cls_name_parts[0] = cls_name_parts[0].capitalize()
                    res_class = " ".join(cls_name_parts)
                    res_conf = str(round(float(box.conf.cpu().item()) * 100))  # Confidence in percentage
                    predictions.append((res_class, res_conf))
        except Exception as ex:
            st.write("No image is uploaded yet!")

        with col1:
            try:    
                st.image(res_plotted, caption = 'Predicted species', use_container_width = True)
            except Exception as ex:
                st.write(":red[Could not make a prediction with sufficient confidence.]")
                st.write(":blue[You can try lowering your acceptable confidence level to find the closest match.]")
                st.write(":blue[It is also possible that your specimen is a species that was not used to train the model, and therefore new to New Zealand.]")
        with col2:
            try:
                with st.expander("ID Results", expanded=True):
                    for species, confidence in predictions:
                        st.markdown(f"""Predicted genus: *{species}*""")
                        st.markdown(f"""Confidence: {confidence}%""")
                        # st.write("Confidence: ", confidence, "%")
            except Exception as ex:
                st.write("")

    # exit_app = st.sidebar.button("Quit")
    # if exit_app:
    #     # Give a bit of delay for user experience
    #     time.sleep(2)
    #     # Close streamlit browser tab
    #     keyboard.press_and_release('ctrl+w')
    #     # Terminate streamlit python process
    #     pid = os.getpid()
    #     p = psutil.Process(pid)
    #     p.terminate()
