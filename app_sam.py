import re
import tempfile

import av
import cv2
import numpy as np
import os
import streamlit as st
import torch
from streamlit_webrtc import webrtc_streamer
from ultralytics.models.sam import SAM3SemanticPredictor


def get_device():
    """Detect and return the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def parse_classes(text: str) -> list[str]:
    """Split a user-supplied string into a list of class names.

    Accepts both dot-separated ('person . car . dog') and
    comma-separated ('person, car, dog') formats.
    """
    parts = re.split(r'[.,]+', text)
    return [p.strip() for p in parts if p.strip()]


@st.cache_resource
def load_predictor():
    device = get_device()
    st.info(f"Using device: {device.upper()}")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="sam3.pt",
        verbose=False,
        device=device,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    return predictor, device


def annotate_masks(image_bgr: np.ndarray, result, classes: list[str]) -> np.ndarray:
    """Overlay coloured semi-transparent masks and bounding-box labels on image_bgr."""
    annotated = image_bgr.copy()

    if result.masks is None or len(result.masks) == 0:
        return annotated

    masks = result.masks.data.cpu().numpy()          # (N, H, W) bool/float
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else None
    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else None

    colours = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
    ]

    h, w = annotated.shape[:2]
    overlay = annotated.copy()

    for i, mask in enumerate(masks):
        colour = colours[i % len(colours)]
        # Resize mask to image size if needed
        if mask.shape != (h, w):
            mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask.astype(np.uint8)

        overlay[mask_resized > 0] = colour

    # Blend overlay
    annotated = cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)

    # Draw bounding boxes and labels
    if boxes is not None:
        for i, box in enumerate(boxes):
            colour = colours[i % len(colours)]
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            if classes and class_ids is not None and i < len(class_ids):
                cls_idx = class_ids[i]
                label = classes[cls_idx] if cls_idx < len(classes) else "unknown"
            else:
                label = ""

            conf_str = f" {confs[i]:.2f}" if confs is not None and i < len(confs) else ""
            text = f"{label}{conf_str}".strip()
            if text:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


class VideoProcessor:
    current_classes: list[str] = []
    current_conf: float = 0.25

    @classmethod
    def update_classes(cls, classes: list[str]):
        cls.current_classes = classes

    @classmethod
    def update_conf(cls, conf: float):
        cls.current_conf = conf

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        classes = VideoProcessor.current_classes
        if classes:
            predictor.set_image(img)
            results = predictor(text=classes, conf=VideoProcessor.current_conf)
            plotted_img = annotate_masks(img, results[0], classes)
        else:
            plotted_img = img
        return av.VideoFrame.from_ndarray(plotted_img, format="bgr24")


predictor, device = load_predictor()

st.title("SAM3 Semantic Segmentation")

mode = st.selectbox("Select mode", ["Upload Image", "Upload Video", "Live Camera"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    word = st.text_input("Enter objects to segment (e.g., person, car, dog)")
    conf = st.slider("Minimum confidence", 0.01, 1.0, 0.25, 0.01)

    if uploaded_file is not None and word.strip():
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        classes = parse_classes(word)

        predictor.set_image(image)
        results = predictor(text=classes, conf=conf)

        plotted_img = annotate_masks(image, results[0], classes)
        st.image(plotted_img, channels="BGR", width='stretch')

elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    word = st.text_input("Enter objects to segment (e.g., person, car, dog)")
    conf = st.slider("Minimum confidence", 0.01, 1.0, 0.25, 0.01)
    frame_skip = st.slider("Process every Nth frame (higher = faster, less detailed)", 1, 10, 5)

    if uploaded_video is not None and word.strip():
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

                predictor.set_image(frame)
                results = predictor(text=classes, conf=conf)

                plotted_img = annotate_masks(frame, results[0], classes)
                frame_placeholder.image(plotted_img, channels="BGR", width='stretch')

            cap.release()
            st.write("Video processing completed.")

elif mode == "Live Camera":
    word = st.text_input("Enter objects to segment (e.g., person, car, dog)")
    conf = st.slider("Minimum confidence", 0.01, 1.0, 0.25, 0.01)

    if word.strip():
        VideoProcessor.update_classes(parse_classes(word))
        VideoProcessor.update_conf(conf)

        if not os.path.exists('/dev/video0'):
            st.error("Camera device not found. Make sure /dev/video0 exists and camera is connected.")
        else:
            webrtc_streamer(
                key="sam3-live",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
            )
