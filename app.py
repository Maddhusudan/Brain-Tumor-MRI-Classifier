import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import datetime

# App setup
st.set_page_config(page_title="🧠 Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classification App")

st.markdown("""
Upload or capture a brain MRI image and the AI model will auto-predict the tumor type:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**
""")

# Load model
@st.cache_resource
def load_model_once():
    return load_model("model.h5")

model = load_model_once()
class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
explanations = {
    "Glioma": "Tumor in the brain or spinal cord.",
    "Meningioma": "Tumor in membranes surrounding brain/spine.",
    "Pituitary": "Tumor in the hormone-regulating pituitary gland.",
    "No Tumor": "No signs of tumor in the image."
}

# Sidebar input type
st.sidebar.header("📤 Choose Input Method")
input_method = st.sidebar.radio("Select input method:", ["Upload Image", "Use Webcam"])

image = None
source = None

# Upload image
if input_method == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        source = uploaded_file.name

# Webcam auto-capture
elif input_method == "Use Webcam":
    st.markdown("📷 **Automatically captures and classifies after photo is taken**")
    captured = st.camera_input("Take a photo using webcam")
    if captured:
        image = Image.open(captured).convert("RGB")
        source = "Webcam Image"

# If image is available, auto-process
if image:
    st.image(image, caption="🖼 MRI Image", use_container_width=True)

    # Preprocess image
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # Predict
    prediction = model.predict(img_input)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show results
    st.success(f"🧠 Prediction: **{predicted_class}**")
    st.info(f"📈 Confidence: **{confidence:.2f}%**")
    st.markdown(f"📘 Explanation: {explanations[predicted_class]}")

    # Class probabilities
    st.subheader("📊 Class Probabilities")
    st.bar_chart({class_names[i]: float(prediction[i]) for i in range(4)})

    # Downloadable report
    report = f"""Brain Tumor Classification Report
-----------------------------
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: {source}
Prediction: {predicted_class}
Confidence: {confidence:.2f}%
Explanation: {explanations[predicted_class]}
"""
    st.download_button("📥 Download Report", report, file_name="tumor_prediction_report.txt")

# Footer
st.markdown("---")
st.markdown("🔬 Developed by Madhusudan using TensorFlow + Streamlit")
