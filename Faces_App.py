import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# --------------------------------------------------
# Konfiguration
# --------------------------------------------------

MODEL_PATH = "emotion_model.h5"  # Pfad zu deinem trainierten Modell
IMG_SIZE = 48  # Anpassen an dein Trainingsformat (z.B. 48 bei FER2013)

# --------------------------------------------------
# Modell laden
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --------------------------------------------------
# Bildvorverarbeitung
# --------------------------------------------------

def preprocess_image(image):
    # PIL → OpenCV Format
    img = np.array(image)

    # In Graustufen umwandeln (falls dein Modell das erwartet)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gesicht skalieren
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalisieren
    img = img / 255.0

    # Dimensionen anpassen (Batch + Channel)
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 1))

    return img

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("😊 Glückserkennung aus Bild")
st.write("Lade ein Bild hoch und die App sagt dir, ob die Person glücklich ist.")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    probability = float(prediction[0][0])

    st.subheader("Ergebnis:")

    if probability > 0.5:
        st.success(f"Die Person ist glücklich 😊 (Wahrscheinlichkeit: {probability:.2f})")
    else:
        st.error(f"Die Person ist nicht glücklich 😐 (Wahrscheinlichkeit: {1 - probability:.2f})")
