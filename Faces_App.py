import streamlit as st
import tensorflow as tf
import numpy as np

# -------------------------------------------------
# Konfiguration
# -------------------------------------------------
MODEL_PATH = "keras_model.h5"

# -------------------------------------------------
# Modell laden (wichtig: compile=False)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# Debug-Anzeige (kannst du später entfernen)
st.write("Model Input Shape:", model.input_shape)

# -------------------------------------------------
# Bildvorverarbeitung (automatisch passend)
# -------------------------------------------------
def preprocess_image(uploaded_file):
    
    # Zielgröße direkt aus dem Modell nehmen
    target_size = model.input_shape[1:3]

    img = tf.keras.utils.load_img(
        uploaded_file,
        target_size=target_size
    )

    img_array = tf.keras.utils.img_to_array(img)

    # Falls Modell Graustufen erwartet (z.B. 48x48x1)
    if model.input_shape[-1] == 1:
        img_array = tf.image.rgb_to_grayscale(img_array)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Gesichtserkennung: Glücklich oder nicht")

uploaded_file = st.file_uploader(
    "Lade ein Bild hoch",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Hochgeladenes Bild")

    processed_image = preprocess_image(uploaded_file)

    prediction = model.predict(processed_image)

    # Falls Sigmoid (Binary)
    if prediction.shape[-1] == 1:
        probability = float(prediction[0][0])

        st.subheader("Ergebnis")

        if probability > 0.5:
            st.success(f"Glücklich 😊 (Wahrscheinlichkeit: {probability:.2f})")
        else:
            st.error(f"Nicht glücklich 😐 (Wahrscheinlichkeit: {1 - probability:.2f})")

    # Falls Softmax (Multi-Class)
    else:
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        st.subheader("Ergebnis")
        st.write(f"Vorhergesagte Klasse: {predicted_class}")
        st.write(f"Konfidenz: {confidence:.2f}")

        if predicted_class == 1:  # anpassen falls andere Label-Position
            st.success("Glücklich 😊")
        else:
            st.error("Nicht glücklich 😐")
