import sys
import os
sys.path.append(os.path.abspath("."))  # IMPORTANT

import streamlit as st
import tensorflow as tf
import yaml
import numpy as np
from PIL import Image
from src.utils import preprocess_image

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

CLASS_NAMES = config["class_names"]
IMG_SIZE = config["img_size"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(config["model_path"])

model = load_model()

st.title(" Tire Condition Detection (Deep Learning CNN)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)

    tensor = preprocess_image(img, IMG_SIZE)
    pred = model.predict(tensor)[0]

    idx = np.argmax(pred)

    st.subheader(f"Prediction: **{CLASS_NAMES[idx]}**")
    st.write(f"Confidence: **{pred[idx]*100:.2f}%**")

