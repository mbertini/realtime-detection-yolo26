import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
import av
import torch
from streamlit_webrtc import webrtc_streamer

def get_device():
    """Detect and return the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

@st.cache_resource
def load_model():
    device = get_device()
    st.info(f"Using device: {device.upper()}")
    model = YOLO("yoloe-26l-seg.pt")
    model.to(device)
    return model

model = load_model()
device = get_device()

class VideoProcessor:
    current_class = ''
    current_conf = 0.25

    @classmethod
    def update_class(cls, new_class):
        cls.current_class = new_class
        names = [new_class]
        model.set_classes(names, model.get_text_pe(names))

    @classmethod
    def update_conf(cls, conf: float):
        cls.current_conf = conf

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if VideoProcessor.current_class:
            results = model.predict(img, device=device, conf=VideoProcessor.current_conf)
            plotted_img = results[0].plot()
        else:
            plotted_img = img  # no detection
        return av.VideoFrame.from_ndarray(plotted_img, format="bgr24")

if 'running' not in st.session_state:
    st.session_state.running = False

st.title("YOLO26 Object Detection")

mode = st.selectbox("Select mode", ["Upload Image", "Upload Video", "Live Camera"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    word = st.text_input("Enter a class to detect (e.g., person, bus)")
    conf = st.slider("Minimum confidence", 0.01, 1.0, 0.25, 0.01)

    if uploaded_file is not None and word.strip():
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Set classes for text prompt
        names = [word.strip()]
        model.set_classes(names, model.get_text_pe(names))

        # Run prediction
        results = model.predict(image, device=device, conf=conf)

        # Get the plotted result image
        plotted_img = results[0].plot()

        # Display the result
        st.image(plotted_img, channels="BGR", width='stretch')
elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    word = st.text_input("Enter a class to detect (e.g., person, bus)")
    conf = st.slider("Minimum confidence", 0.01, 1.0, 0.25, 0.01)
    frame_skip = st.slider("Process every Nth frame (higher = faster, less detailed)", 1, 10, 5)

    if uploaded_video is not None and word.strip():
        # Save uploaded video to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open video file")
        else:
            frame_placeholder = st.empty()
            st.write("Processing video...")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                # Set classes for text prompt
                names = [word.strip()]
                model.set_classes(names, model.get_text_pe(names))

                # Run prediction
                results = model.predict(frame, device=device, conf=conf)

                # Get the plotted result image
                plotted_img = results[0].plot()

                # Display the result
                frame_placeholder.image(plotted_img, channels="BGR", width='stretch')

            cap.release()
            st.write("Video processing completed.")
elif mode == "Live Camera":
    word = st.text_input("Enter a class to detect (e.g., person, bus)")
    conf = st.slider("Minimum confidence", 0.01, 1.0, 0.25, 0.01)

    if word.strip():
        VideoProcessor.update_class(word.strip())
        VideoProcessor.update_conf(conf)

        if not os.path.exists('/dev/video0'):
            st.error("Camera device not found. Make sure /dev/video0 exists and camera is connected.")
        else:
            webrtc_streamer(
                key="yolo-live",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
            )
