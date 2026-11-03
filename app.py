import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# optional: lets the app open iPhone .HEIC / .HEIF photos
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

HERE = Path(__file__).parent
IMG_SIZE = (224, 224)

st.set_page_config(page_title='Cat vs Rabbit', page_icon='🐾')


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(HERE / 'cat_rabbit_cnn.keras')
    class_names = json.loads((HERE / 'class_names.json').read_text())
    return model, class_names


def load_image(file):
    """Open any image format and return a clean RGB image."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)                 # fix sideways phone photos
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')                      # transparent PNG/GIF -> white background
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        background.alpha_composite(img)
        img = background
    return img.convert('RGB')


model, class_names = load_model()

st.title('🐱 Cat vs Rabbit 🐰')
st.write('Upload a photo and the CNN will tell you whether it shows a cat or a rabbit.')

# type=None -> accepts any file; non-images are caught below
uploaded = st.file_uploader('Choose an image (JPG, JPEG, PNG, WEBP, BMP, GIF, TIFF, HEIC ...)', type=None)

if uploaded is not None:
    try:
        img = load_image(uploaded)
    except (UnidentifiedImageError, OSError):
        st.error('Sorry, I could not read that file as an image. Please upload a picture file.')
        st.stop()

    st.image(img, caption='Your image', use_container_width=True)

    # same preprocessing as training: resize to 224x224, raw 0-255 floats
    # (the model rescales to 0-1 internally)
    x = np.expand_dims(np.array(img.resize(IMG_SIZE, Image.BILINEAR), dtype='float32'), axis=0)

    p_class1 = float(model.predict(x, verbose=0)[0][0])
    probs = {class_names[0]: 1 - p_class1, class_names[1]: p_class1}
    label = max(probs, key=probs.get)

    st.subheader(f'Prediction: {label.capitalize()}')
    st.write(f'Confidence: **{probs[label]:.1%}**')
    for name, p in probs.items():
        st.progress(p, text=f'{name}: {p:.1%}')
