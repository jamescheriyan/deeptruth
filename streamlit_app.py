import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Load model
@st.cache_resource
def load_deepfake_model():
    model_path = os.path.join("model", "MesoInception_DF.h5")
    model = load_model(model_path)
    return model

model = load_deepfake_model()

# Image pre-processing
def preprocess_frame(frame):
    face = cv2.resize(frame, (256, 256))
    face = face.astype(np.float32) / 255.0
    return np.expand_dims(face, axis=0)

# Run detection
def predict_deepfake(frame):
    input_frame = preprocess_frame(frame)
    prediction = model.predict(input_frame)[0][0]
    return prediction

# UI
st.title("🕵️‍♂️ DeepTruth - Deepfake Video Detector")
st.markdown("Upload a video and we'll analyze frames to detect if it's real or fake.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    tfile = os.path.join("temp_video.mp4")
    with open(tfile, "wb") as f:
        f.write(uploaded_file.read())

    st.video(tfile)
    st.write("🔍 Analyzing...")

    cap = cv2.VideoCapture(tfile)
    frames_checked = 0
    fake_scores = []

    while frames_checked < 10:
        ret, frame = cap.read()
        if not ret:
            break
        if frames_checked % 3 == 0:  # sample every 3rd frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            score = predict_deepfake(frame_rgb)
            fake_scores.append(score)
        frames_checked += 1

    cap.release()

    avg_score = np.mean(fake_scores)
    st.subheader("🧠 Deepfake Score: {:.2f}".format(avg_score))

    if avg_score > 0.5:
        st.error("⚠️ Likely a Deepfake")
    else:
        st.success("✅ Likely Real")

