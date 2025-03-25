
import keras
import numpy as np
from keras.applications import vgg16

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from PIL import Image
import requests
from io import BytesIO
import streamlit as st


vgg_model = vgg16.VGG16(weights='imagenet')

def predict_image(image_path, model):
    original = Image.open(image_path)

    newsize = (224, 224) 
    original = original.resize(newsize) 

    numpy_image = img_to_array(original)

    image_batch = np.expand_dims(numpy_image, axis=0)

    # prepare the image for the VGG model
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = model.predict(processed_image)

    label = decode_predictions(predictions)
    if label[0][0][-1] >= 0.5:
        return(label[0][0]) #just display top 2
    return 

image = Image.open('./1.jpg')

st.header("Perfume Classifier")

st.image(image)

st.write("Classification Model for Perfume")

uploaded_file = st.file_uploader("Choose an image file", type="jpg")

if uploaded_file:
    predicted = predict_image(uploaded_file, vgg_model)
    st.image(uploaded_file)

    if predicted:
        st.write("Classification: " + (predicted[1]))
        st.write("Accuracy: " + (predicted[-1]))
