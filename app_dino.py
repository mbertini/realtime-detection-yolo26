import streamlit as st
import cv2
import numpy as np
import os
import av
import torch
from pathlib import Path
from streamlit_webrtc import webrtc_streamer
from groundeddino_vl.utils.inference import Model
from groundeddino_vl.weights_manager import download_model_weights
import groundeddino_vl.utils.inference as _dino_inference
import supervision as sv


def get_device():
    """Detect and return the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def _patch_predict_device(device: str):
    """Patch the library's predict() default device so MPS/CPU work correctly on non-CUDA systems."""
    import functools
    original_predict = _dino_inference.predict

    @functools.wraps(original_predict)
    def patched_predict(model, image, caption, box_threshold, text_threshold, device=device):
        return original_predict(model, image, caption, box_threshold, text_threshold, device)

    _dino_inference.predict = patched_predict


@st.cache_resource
def load_model():
    device = get_device()
    _patch_predict_device(device)
    st.info(f"Using device: {device.upper()}")
    config_path, weights_path = download_model_weights()
    model = Model(
        model_config_path=config_path,
        model_checkpoint_path=weights_path,
        device=device,
    )
    return model, device


model, device = load_model()


def annotate_image(image_bgr: np.ndarray, detections: sv.Detections, labels: list) -> np.ndarray:
    """Draw bounding boxes and labels on an image using Supervision annotators."""
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    labels_with_conf = [
        f"{label} {conf:.2f}"
        for label, conf in zip(labels, detections.confidence)
    ]
    annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels_with_conf,
    )
    return annotated


class VideoProcessor:
    current_caption = ''

    @classmethod
    def update_caption(cls, caption: str):
        cls.current_caption = caption

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if VideoProcessor.current_caption:
            detections, labels = model.predict_with_caption(
                image=img,
                caption=VideoProcessor.current_caption,
            )
            plotted_img = annotate_image(img, detections, labels)
        else:
            plotted_img = img
        return av.VideoFrame.from_ndarray(plotted_img, format="bgr24")


st.title("Grounding DINO Object Detection")

mode = st.selectbox("Select mode", ["Upload Image", "Upload Video", "Live Camera"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    word = st.text_input("Enter objects to detect (e.g., person . car . dog)")

    if uploaded_file is not None and word.strip():
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        detections, labels = model.predict_with_caption(
            image=image,
            caption=word.strip(),
        )

        plotted_img = annotate_image(image, detections, labels)
        st.image(plotted_img, channels="BGR", width='stretch')

elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    word = st.text_input("Enter objects to detect (e.g., person . car . dog)")
    frame_skip = st.slider("Process every Nth frame (higher = faster, less detailed)", 1, 10, 5)

    if uploaded_video is not None and word.strip():
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

                detections, labels = model.predict_with_caption(
                    image=frame,
                    caption=word.strip(),
                )

                plotted_img = annotate_image(frame, detections, labels)
                frame_placeholder.image(plotted_img, channels="BGR", width='stretch')

            cap.release()
            st.write("Video processing completed.")

elif mode == "Live Camera":
    word = st.text_input("Enter objects to detect (e.g., person . car . dog)")

    if word.strip():
        VideoProcessor.update_caption(word.strip())

        if not os.path.exists('/dev/video0'):
            st.error("Camera device not found. Make sure /dev/video0 exists and camera is connected.")
        else:
            webrtc_streamer(
                key="dino-live",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
            )
