import streamlit as st
import numpy as np
import tensorflow as tf

# -----------------------------
# Konfiguration
# -----------------------------
MODEL_PATH = "emotion_model.h5"
IMG_SIZE = 48  # Anpassen falls nötig

# -----------------------------
# Modell laden
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# Bildvorverarbeitung
# -----------------------------
def preprocess_image(uploaded_file):
    img = tf.keras.utils.load_img(
        uploaded_file,
        color_mode="grayscale",
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# -----------------------------
# UI
# -----------------------------
st.title("Glückserkennung 😊")

uploaded_file = st.file_uploader(
    "Lade ein Bild hoch",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Hochgeladenes Bild")

    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)[0][0]

    st.subheader("Ergebnis")

    if prediction > 0.5:
        st.success(f"Glücklich 😊 (Wahrscheinlichkeit: {prediction:.2f})")
    else:
        st.error(f"Nicht glücklich 😐 (Wahrscheinlichkeit: {1 - prediction:.2f})")

