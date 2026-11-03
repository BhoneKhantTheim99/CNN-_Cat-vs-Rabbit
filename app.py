import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

HERE = Path(__file__).parent
IMG_SIZE = (224, 224)

st.set_page_config(page_title='Cat vs Rabbit', page_icon='🐾')


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(HERE / 'cat_rabbit_cnn.keras')
    class_names = json.loads((HERE / 'class_names.json').read_text())
    return model, class_names


model, class_names = load_model()

st.title('🐱 Cat vs Rabbit 🐰')
st.write('Upload a photo and the CNN will tell you whether it shows a cat or a rabbit.')

uploaded = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Your image', use_container_width=True)

    x = np.expand_dims(np.array(img.resize(IMG_SIZE, Image.BILINEAR), dtype='float32'), axis=0)

    p_class1 = float(model.predict(x, verbose=0)[0][0])
    probs = {class_names[0]: 1 - p_class1, class_names[1]: p_class1}
    label = max(probs, key=probs.get)

    st.subheader(f'Prediction: {label.capitalize()}')
    st.write(f'Confidence: **{probs[label]:.1%}**')
    for name, p in probs.items():
        st.progress(p, text=f'{name}: {p:.1%}')
