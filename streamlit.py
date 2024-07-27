# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import imutils


# Loading the Model
model = load_model('ResNetTransferLearning_model.h5', compile=False)

# Set background color and page width
st.set_page_config(page_title="Object Classifier App", page_icon="🌟", layout="wide", initial_sidebar_state="expanded")

# Custom styling
st.markdown(
    """
    <style>
        body {
            background-color: #cfe2f3;
        }
        .stApp {
            max-width: 1000px;
            margin: 0 auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Image upload widget
st.image('/content/drive/MyDrive/DeepLearning_UniversityProject/bg.jpg')

st.markdown("## Object Classifier App with Deep Learning")

object_image = st.file_uploader("Upload an image...", type=['png', 'jpg', 'webp', 'jpeg'])

submit = st.button('Predict')

# Define your custom labels
custom_labels = ["Chair", "Clock", "Desk", "Laptop", "People"]

# On predict button click
if submit:
    if object_image is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(object_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(opencv_image, channels="BGR", use_column_width=True)

        # Preprocess the image for the model
        opencv_image = cv2.resize(opencv_image, (150, 150))  # Resize to match model input size
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Make predictions
        predictions = model.predict(opencv_image)

        # Get the index of the predicted class
        predicted_class_index = np.argmax(predictions[0])

        # Display the predicted class
        st.markdown("### Prediction:")
        st.text(f'The predicted class is: {custom_labels[predicted_class_index]} with accuracy {predictions[0][predicted_class_index]:.2f}')