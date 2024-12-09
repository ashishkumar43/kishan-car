
import streamlit as st
from gtts import gTTS
from io import BytesIO
import cv2
import pytesseract
import re
import time
import base64
from PIL import Image
import numpy as np
from ultralytics import YOLO

import os

if not os.system("pip show gTTS"):  
    import subprocess
    subprocess.run(["python", "newfile.py"]) 
    
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# model = YOLO(r'D:\Minor Project\app_sound\best_license_plate_model_updated.pt')
model = YOLO('./best_license_plate_model_updated.pt')  

def apply_custom_css():
    st.markdown("""
    <style>
    /* General body styling */
    body {
        background: white;
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .stApp {
        background-color: #f4f4f9;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
    }
    .sidebar h2 {
        color: #ffffff;  /* White title */
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
    }
    .sidebar {
        background: linear-gradient(to bottom, #1e90ff, #ff4f58);
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar-radio {
        padding: 10px;
        border-radius: 8px;
        background: white;
        color: black;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #1e90ff;  /* Blue button */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1.2rem;
        transition: background 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #ff4f58;  /* Red hover */
        transform: scale(1.05);
    }
    /* Styling for main content titles */
    .main-title {
        color: #1e90ff;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #555555;
        margin: 20px 0;
        text-align: justify;
        line-height: 1.6;
    }
    .subtitle strong {
        color: #ff6347;
    }
    .file-upload {
        font-size: 1rem;
        font-weight: bold;
        color: #1e90ff;
        text-align: center;
        margin-bottom: 40px;
    }
    /* Styling for images */
    img {
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def generate_audio(message):
    tts = gTTS(message, lang='en')
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    return audio_data.read()

# Play audio
def autoplay_audio(audio_bytes):
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def spell_out_text(text):
    return ' '.join(text)

def predict_and_validate_license_plate(uploaded_image):
    image = Image.open(uploaded_image)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("🔍 Processing the image, please wait..."):
        time.sleep(2)

    results = model.predict(image, device='cpu')

    plate_pattern = r'^(MH|HR|DL)[A-Z0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}[A-Z0-9]{0,4}$'

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0] * 100 
            roi = image[y1:y2, x1:x2]

            text = pytesseract.image_to_string(roi, config='--psm 6').strip().upper()
            spelled_text = spell_out_text(text)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{confidence:.2f}%", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            st.write(f"🔑 **Detected Plate:** {text}")
            autoplay_audio(generate_audio(f"Detected plate is: {spelled_text}"))
            st.write(f"🔊 **Audio played for plate:** {text}")
            time.sleep(7)

            if re.match(plate_pattern, text):
                validation_text = " VALID PLATE"
                color = (0, 255, 0)
                autoplay_audio(generate_audio("This is a valid plate."))
            else:
                validation_text = " FRAUDULENT PLATE"
                color = (255, 0, 0)
                autoplay_audio(generate_audio("This is a fraudulent plate. Please investigate immediately."))
                
            cv2.putText(image, validation_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            time.sleep(10)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="📸 Detected License Plate with Bounding Boxes", use_container_width=True)

apply_custom_css()

st.sidebar.markdown('<div class="sidebar"><h2>Navigation</h2></div>', unsafe_allow_html=True)
st.sidebar.header("📌 Select Option")
selected_option = st.sidebar.radio("", ["Home", "Upload Image", "About"])

if "uploaded_image_key" not in st.session_state:
    st.session_state["uploaded_image_key"] = 0

if selected_option == "Home":
    st.markdown('<div class="main-title">🏠 Welcome to License Plate Detection App 🚗</div>', unsafe_allow_html=True)
    st.markdown("### This app uses advanced YOLO and OCR for real-time vehicle license plate detection and validation.")

if selected_option == "Upload Image":
    st.markdown('<div class="main-title">📤 Upload an Image for License Plate Detection</div>', unsafe_allow_html=True)

    # Adding a subtitle to explain the action
    st.markdown("""
    <p style="font-size: 1.2rem; color: #555555; text-align: center;">
        Upload an image of a vehicle to automatically detect and validate the license plate.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="file-upload">📂 Drag and drop an image or click to upload.</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"], key=st.session_state["uploaded_image_key"], label_visibility="collapsed")

    with col2:
        if uploaded_image:
            st.markdown("""
            <div style="text-align: center; margin-top: 20px;">
                <button style="background-color: #1e90ff; color: white; padding: 12px 24px; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer;">
                    🔍 Start Detection
                </button>
            </div>
            """, unsafe_allow_html=True)
            predict_and_validate_license_plate(uploaded_image)
            # st.write("Detection complete.")
            autoplay_audio(generate_audio("Detection complete."))
            # st.write("Message: Detection process completed and audio played.")
            st.session_state["uploaded_image_key"] += 1
        # else:
        #     st.markdown("""
        #     <p style="font-size: 1.2rem; color: #888888; text-align: center;">
        #         Please upload an image to begin.
        #     </p>
        #     """, unsafe_allow_html=True)

elif selected_option == "About":
    st.markdown("""
    <div class="main-title">ℹ️ About This App</div>
    <p style="font-size: 1.2rem; color: #555555; text-align: justify; line-height: 1.6;">
    Welcome to our License Plate Detection App!🚗 This innovative tool leverages cutting-edge technologies like 
    YOLO (You Only Look Once) for real-time object detection and OCR (Optical Character Recognition)
    to automatically detect and read vehicle license plates with incredible accuracy.
    </p>

    ## Why Choose This App?
    <p style="font-size: 1.2rem; color: #555555; text-align: justify; line-height: 1.6;">
        Our app is designed to be user-friendly, quick, and efficient in recognizing vehicle plates, ensuring high 
        accuracy and reliability for various applications.
    </p>

    ## Key Features of the App:
    <ul style="font-size: 1.2rem; color: #555555; line-height: 1.8;">
        <li><strong>🚗 Real-Time License Plate Detection</strong>: Uses YOLO for high-speed, real-time object detection of vehicle plates.</li>
        <li><strong>🔠 Optical Character Recognition (OCR)</strong>: Extracts and processes the alphanumeric characters from plates efficiently.</li>
        <li><strong>✅ Reliable Validation</strong>: The app verifies if a detected license plate matches a valid format.</li>
        <li><strong>💡 Easy to Use</strong>: Simply upload an image and let the app handle the detection process for you!</li>
    </ul>

    ## Applications:
    <p style="font-size: 1.2rem; color: #555555; text-align: justify; line-height: 1.6;">
        Our license plate detection system can be applied in a variety of industries and sectors such as:
    </p>

    <ul style="font-size: 1.2rem; color: #555555; line-height: 1.8;">
        <li><strong>🏙️ Traffic Management</strong>: Helps with vehicle monitoring, law enforcement, and toll collection.</li>
        <li><strong>🚗 Parking Systems</strong>: Simplifies automated parking access and fee collection.</li>
        <li><strong>⚖️ Law Enforcement</strong>: Enhances vehicle identification for crime prevention and investigations.</li>
        <li><strong>🚓 Security Systems</strong>: Monitors and tracks vehicles for surveillance purposes.</li>
    </ul>
    """, unsafe_allow_html=True)
