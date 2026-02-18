import streamlit as st
import cv2
import numpy as np
import os
import av
import torch
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


def parse_classes(text: str) -> list[str]:
    """Split a user-supplied string into a list of class names.

    Accepts both dot-separated ('person . car . dog') and
    comma-separated ('person, car, dog') formats.
    """
    import re
    parts = re.split(r'[.,]+', text)
    return [p.strip() for p in parts if p.strip()]


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


def annotate_image(image_bgr: np.ndarray, detections: sv.Detections, classes: list[str]) -> np.ndarray:
    """Draw bounding boxes and labels on an image using Supervision annotators."""
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    labels = []
    for i, (class_id, conf) in enumerate(zip(detections.class_id, detections.confidence)):
        class_name = classes[class_id] if class_id is not None and class_id < len(classes) else "unknown"
        labels.append(f"{class_name} {conf:.2f}")

    annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels,
    )
    return annotated


class VideoProcessor:
    current_classes: list[str] = []
    current_box_threshold = 0.35
    current_text_threshold = 0.25

    @classmethod
    def update_classes(cls, classes: list[str]):
        cls.current_classes = classes

    @classmethod
    def update_thresholds(cls, box_threshold: float, text_threshold: float):
        cls.current_box_threshold = box_threshold
        cls.current_text_threshold = text_threshold

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if VideoProcessor.current_classes:
            detections = model.predict_with_classes(
                image=img,
                classes=VideoProcessor.current_classes,
                box_threshold=VideoProcessor.current_box_threshold,
                text_threshold=VideoProcessor.current_text_threshold,
            )
            plotted_img = annotate_image(img, detections, VideoProcessor.current_classes)
        else:
            plotted_img = img
        return av.VideoFrame.from_ndarray(plotted_img, format="bgr24")


st.title("Grounding DINO Object Detection")

mode = st.selectbox("Select mode", ["Upload Image", "Upload Video", "Live Camera"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    word = st.text_input("Enter objects to detect (e.g., person . car . dog)")
    box_threshold = st.slider("Box threshold", 0.01, 1.0, 0.35, 0.01)
    text_threshold = st.slider("Text threshold", 0.01, 1.0, 0.25, 0.01)

    if uploaded_file is not None and word.strip():
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        classes = parse_classes(word)

        detections = model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        plotted_img = annotate_image(image, detections, classes)
        st.image(plotted_img, channels="BGR", width='stretch')

elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    word = st.text_input("Enter objects to detect (e.g., person . car . dog)")
    box_threshold = st.slider("Box threshold", 0.01, 1.0, 0.35, 0.01)
    text_threshold = st.slider("Text threshold", 0.01, 1.0, 0.25, 0.01)
    frame_skip = st.slider("Process every Nth frame (higher = faster, less detailed)", 1, 10, 5)

    if uploaded_video is not None and word.strip():
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        classes = parse_classes(word)
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

                detections = model.predict_with_classes(
                    image=frame,
                    classes=classes,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )

                plotted_img = annotate_image(frame, detections, classes)
                frame_placeholder.image(plotted_img, channels="BGR", width='stretch')

            cap.release()
            st.write("Video processing completed.")

elif mode == "Live Camera":
    word = st.text_input("Enter objects to detect (e.g., person . car . dog)")
    box_threshold = st.slider("Box threshold", 0.01, 1.0, 0.35, 0.01)
    text_threshold = st.slider("Text threshold", 0.01, 1.0, 0.25, 0.01)

    if word.strip():
        VideoProcessor.update_classes(parse_classes(word))
        VideoProcessor.update_thresholds(box_threshold, text_threshold)

        if not os.path.exists('/dev/video0'):
            st.error("Camera device not found. Make sure /dev/video0 exists and camera is connected.")
        else:
            webrtc_streamer(
                key="dino-live",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
            )
