import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(img, img_size):
    img = img.resize((img_size, img_size))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)
