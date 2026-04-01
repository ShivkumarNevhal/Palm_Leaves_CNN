import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Background (same as your code)
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

# Models
model_options = {
    "CNN Model": "models/best_model.keras",
    "MobileNet": "models/mobilenet.keras",
    "ResNet": "models/Resnet_model1.keras"
}

selected_model_name = st.selectbox("🔽 Select Model", list(model_options.keys()))

@st.cache_resource
def load_my_model(path):
    return load_model(path)

model = load_my_model(model_options[selected_model_name])

# Classes
class_names = ['Healthy', 'Boron', 'Kalium', 'Magnesium', 'Nitrogen']

# Title
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
# INPUT SECTION (UPDATED)
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

# Decide image source
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

# =========================
# PREDICTION
# =========================

if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    pred_index = np.argmax(prediction)
    predicted_class = class_names[pred_index]
    confidence = prediction[0][pred_index] * 100

    # Custom Styled Output (Better than st.success)
    st.markdown(f"""
    <div style="background:rgba(0,128,0,0.8); padding:15px; border-radius:10px; margin:10px 0;">
    ✅ Model Used: {selected_model_name}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(255,140,0,0.9); padding:15px; border-radius:10px; margin:10px 0;">
    🌿 Prediction: {predicted_class}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:rgba(0,102,204,0.9); padding:15px; border-radius:10px; margin:10px 0;">
    📊 Confidence: {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # Probabilities
    st.subheader("📈 Class Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")