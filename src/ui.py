from model_inference import main
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
from skimage import io, color

st.set_page_config(layout="centered", page_title="Face Landmark Detector")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stProgress > div > div > div > div {
                background-color: #f783ac !important;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("How printable is your pic? :camera_with_flash:")
st.markdown("### Upload an image to analyze.")

MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

def process_image(upload):
    st.markdown("### Processing...")
    progress_bar = st.progress(0)
    
    image = io.imread(upload)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    
    progress_bar.progress(50)
    
    result = main(image)
    if result is not None:
        landmarks_image, printability_mean, emotion = result
    else:
        landmarks_image = None
        printability_mean = 0
        emotion = None
    
    progress_bar.progress(100)  


    st.markdown(f"### Printability Meter:")
    
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original Image", use_column_width=True)

    if landmarks_image is not None:
        col2.image(landmarks_image, caption=f"Overall emotion: {emotion}", use_column_width=True)
    else:
        col2.write("Couldn't process image. :cry:")
        return

    
    st.markdown(f"### Printability Meter:")
    st.progress(int(printability_mean))
    st.write(f"Printability: {round(printability_mean, 1)}%")  
    
    if printability_mean == 100:
        st.balloons()
        st.markdown(":tada: **Hooray! Your pic is 100% printable!** :tada:")
    elif printability_mean > 50:
        st.markdown(":smile: **Your pic is pretty printable!** :smile:")
    elif printability_mean > 25:
        st.markdown(":neutral_face: **Not very smiley are we.** :neutral_face:")
    elif printability_mean == 0:
        st.markdown(":cry: **Wouldn't recommend printing that...** :cry:")

with st.spinner('Waiting for image upload...'):
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if upload is not None:
    if upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        process_image(upload)
