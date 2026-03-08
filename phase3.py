import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Titre de l'application
st.title("🧑‍💻 Application de Détection de Visages")

# ==============================
# 📖 Instructions
# ==============================
st.markdown("""
### 📌 Instructions d'utilisation :

1. Téléchargez une image contenant un ou plusieurs visages.
2. Ajustez les paramètres de détection si nécessaire :
   - **scaleFactor** : contrôle la réduction de taille de l'image.
   - **minNeighbors** : contrôle la qualité de détection.
3. Choisissez la couleur des rectangles.
4. Cliquez sur le bouton de téléchargement pour sauvegarder l’image détectée.
""")

# ==============================
# 📤 Upload Image
# ==============================
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

# ==============================
# 🎛 Paramètres réglables
# ==============================

st.sidebar.header("⚙ Paramètres de détection")

scaleFactor = st.sidebar.slider(
    "Scale Factor",
    min_value=1.01,
    max_value=1.5,
    value=1.1,
    step=0.01
)

minNeighbors = st.sidebar.slider(
    "Min Neighbors",
    min_value=1,
    max_value=10,
    value=5,
    step=1
)

color = st.sidebar.color_picker("Choisir la couleur du rectangle", "#00FF00")

# Conversion couleur HEX → BGR (OpenCV utilise BGR)
color = color.lstrip('#')
r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
rectangle_color = (b, g, r)

# ==============================
# 📷 Détection
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Charger le modèle Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors
    )

    # Dessiner les rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), rectangle_color, 2)

    # Convertir pour affichage Streamlit
    img_result = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    st.image(img_result, caption="Image avec visages détectés", use_column_width=True)

    st.success(f"{len(faces)} visage(s) détecté(s)")

    # ==============================
    # 💾 Sauvegarde image
    # ==============================

    img_pil = Image.fromarray(img_result)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="💾 Télécharger l'image avec détection",
        data=byte_im,
        file_name="image_visages_detectes.png",
        mime="image/png"
    )