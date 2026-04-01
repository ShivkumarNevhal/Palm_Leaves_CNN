import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
import tensorflow as tf

# =========================
# BACKGROUND
# =========================
def set_bg():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQE2XYWetj2su939Dj7ET4zITUCxWOXIHVsFg&s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    .stApp::before {
        content: "";
        position: fixed;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.6);
        z-index: -1;
    }

    html, body, [class*="css"] {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

set_bg()

# =========================
# MODEL CONFIG
# =========================
model_configs = {
    "CNN Model": {
        "file_id": "https://drive.google.com/file/d/1NiIXUCDw0ZrTCLieIPn3GmyGrNVZbH3L/view?usp=drive_link",
        "file_name": "best_model.keras",
        "img_size": 224
    },
    "MobileNet": {
        "file_id": "https://drive.google.com/file/d/1Zkqtusi1QMZqgOrEU-PAtuE-197g25Up/view?usp=sharing",
        "file_name": "MobileNet.keras",
        "img_size": 224
    },
    "ResNet": {
        "file_id": "https://drive.google.com/file/d/1YUEaZnmxJLyC7DDEU7hNwJfhNp1z4p37/view?usp=drive_link",
        "file_name": "Resnet_model1.keras",
        "img_size": 224
    }
}

selected_model_name = st.selectbox("🔽 Select Model", list(model_configs.keys()))
selected_model_config = model_configs[selected_model_name]

# =========================
# LOAD MODEL (FIXED)
# =========================
@st.cache_resource
def load_model_from_drive(file_id, file_name):
    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        if not os.path.exists(file_name):
            with st.spinner(f"⬇️ Downloading {file_name}..."):
                gdown.download(url, file_name, quiet=False, fuzzy=True)

        model = tf.keras.models.load_model(file_name)
        return model

    except Exception as e:
        st.error("❌ Model loading failed. Check Google Drive file sharing or file ID.")
        st.exception(e)
        return None


model = load_model_from_drive(
    selected_model_config["file_id"],
    selected_model_config["file_name"]
)

IMG_SIZE = selected_model_config["img_size"]

# =========================
# CLASS LABELS
# =========================
class_names = ['Healthy', 'Boron', 'Kalium', 'Magnesium', 'Nitrogen']

# =========================
# TITLE
# =========================
st.markdown("""
<h1 style="
    text-align: center;
    font-size: 50px;
    color: #ffffff;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.9);
">
🌴 Smart Palm Leaf Classification System
</h1>

<p style="
    text-align: center;
    font-size: 20px;
    color: #ccffcc;
">
AI-Based Classification of Nutrient Deficiencies in Palm Leaves 🚀
</p>
""", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
st.subheader("📥 Choose Input Method")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "📁 Upload Image",
        type=["jpg", "png", "jpeg", "webp"]
    )

with col2:
    camera_image = st.camera_input("📷 Capture Image")

# Image selection
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

# =========================
# PREDICTION
# =========================
if image is not None and model is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    pred_index = np.argmax(prediction)
    predicted_class = class_names[pred_index]
    confidence = prediction[0][pred_index] * 100

    # Model Used
    st.markdown(f"""
    <div style="background:rgba(0,128,0,0.8); padding:15px; border-radius:10px;">
    ✅ Model Used: {selected_model_name}
    </div>
    """, unsafe_allow_html=True)

    # Prediction
    st.markdown(f"""
    <div style="background:rgba(255,140,0,0.9); padding:15px; border-radius:10px; margin-top:10px;">
    🌿 Prediction: {predicted_class}
    </div>
    """, unsafe_allow_html=True)

    # Confidence
    st.markdown(f"""
    <div style="background:rgba(0,102,204,0.9); padding:15px; border-radius:10px; margin-top:10px;">
    📊 Confidence: {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # Low confidence warning
    if confidence < 60:
        st.warning("⚠️ Low confidence prediction. Try a clearer image.")

    # Probabilities
    st.subheader("📈 Class Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")
