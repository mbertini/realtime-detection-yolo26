import re
import tempfile
import os
import av
import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor
import supervision as sv

# --- Utility Functions ---

def get_device():
    """Detect and return the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def parse_classes(text: str) -> list[str]:
    """Split a user-supplied string into a list of class names."""
    parts = re.split(r'[.,]+', text)
    return [p.strip() for p in parts if p.strip()]

# --- Model Loading (Cached) ---

@st.cache_resource
def load_yolo_model():
    device = get_device()
    # YOLO here refers to YOLOE-26L specifically in this project
    model = YOLO("yoloe-26l-seg.pt")
    model.to(device)
    return model

@st.cache_resource
def load_dino_model():
    from groundeddino_vl.utils.inference import Model
    from groundeddino_vl.weights_manager import download_model_weights
    import groundeddino_vl.utils.inference as _dino_inference
    device = get_device()

    # Patch for DINO to work correctly on non-CUDA systems
    import functools
    original_predict = _dino_inference.predict
    @functools.wraps(original_predict)
    def patched_predict(model, image, caption, box_threshold, text_threshold, device=device):
        return original_predict(model, image, caption, box_threshold, text_threshold, device)
    _dino_inference.predict = patched_predict

    config_path, weights_path = download_model_weights()
    model = Model(
        model_config_path=config_path,
        model_checkpoint_path=weights_path,
        device=device,
    )
    return model

@st.cache_resource
def load_sam_predictor():
    device = get_device()
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="sam3.pt",
        verbose=False,
        device=device,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    return predictor

# --- Annotation Functions ---

def annotate_dino_image(image_bgr: np.ndarray, detections: sv.Detections, classes: list[str]) -> np.ndarray:
    """Draw bounding boxes and labels for Grounding DINO."""
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

def annotate_sam_masks(image_bgr: np.ndarray, result, classes: list[str]) -> np.ndarray:
    """Overlay coloured semi-transparent masks and labels for SAM3."""
    annotated = image_bgr.copy()
    if result.masks is None or len(result.masks) == 0:
        return annotated

    masks = result.masks.data.cpu().numpy()
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
        if mask.shape != (h, w):
            mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask.astype(np.uint8)
        overlay[mask_resized > 0] = colour

    annotated = cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)

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
                cv2.putText(annotated, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return annotated

# --- Video Processor ---

class UnifiedVideoProcessor:
    model_type = "YOLO"
    current_classes: list[str] = []
    conf = 0.25
    box_threshold = 0.35
    text_threshold = 0.25
    device = "cpu"

    @classmethod
    def update_params(cls, model_type, classes, conf, box_threshold, text_threshold, device):
        cls.model_type = model_type
        cls.current_classes = classes
        cls.conf = conf
        cls.box_threshold = box_threshold
        cls.text_threshold = text_threshold
        cls.device = device

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not self.current_classes:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            if self.model_type == "YOLO":
                model = load_yolo_model()
                names = self.current_classes
                model.set_classes(names, model.get_text_pe(names))
                results = model.predict(img, device=self.device, conf=self.conf)
                plotted_img = results[0].plot()
            elif self.model_type == "Grounding DINO":
                model = load_dino_model()
                detections = model.predict_with_classes(
                    image=img,
                    classes=self.current_classes,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                )
                plotted_img = annotate_dino_image(img, detections, self.current_classes)
            elif self.model_type == "SAM3":
                predictor = load_sam_predictor()
                predictor.set_image(img)
                results = predictor(text=self.current_classes, conf=self.conf)
                plotted_img = annotate_sam_masks(img, results[0], self.current_classes)
            else:
                plotted_img = img
        except Exception as e:
            # Fallback to raw image on error during live processing
            plotted_img = img
            
        return av.VideoFrame.from_ndarray(plotted_img, format="bgr24")

# --- Streamlit UI ---

st.set_page_config(page_title="Vision Model Hub", layout="wide")
st.title("Real-Time Vision Model Hub")

# Sidebar - Model Selection
st.sidebar.title("Configuration")
model_choice = st.sidebar.selectbox("Select Model Backend", ["YOLO", "Grounding DINO", "SAM3"])
device = get_device()
st.sidebar.info(f"Using device: {device.upper()}")

# Sidebar - Common Parameters
st.sidebar.subheader("Parameters")
if model_choice == "YOLO":
    word = st.sidebar.text_input("Enter a class (e.g., person)")
    conf = st.sidebar.slider("Confidence", 0.01, 1.0, 0.25)
    classes = [word.strip()] if word.strip() else []
    box_threshold = 0.35 # defaults
    text_threshold = 0.25
elif model_choice == "Grounding DINO":
    word = st.sidebar.text_input("Enter objects (e.g., person . car)")
    box_threshold = st.sidebar.slider("Box Threshold", 0.01, 1.0, 0.35)
    text_threshold = st.sidebar.slider("Text Threshold", 0.01, 1.0, 0.25)
    classes = parse_classes(word)
    conf = 0.25 # default
elif model_choice == "SAM3":
    word = st.sidebar.text_input("Enter objects (e.g., person, car)")
    conf = st.sidebar.slider("Confidence", 0.01, 1.0, 0.25)
    classes = parse_classes(word)
    box_threshold = 0.35
    text_threshold = 0.25

# Main Content - Mode Selection
mode = st.selectbox("Select Mode", ["Upload Image", "Upload Video", "Live Camera"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None and classes:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner(f"Running {model_choice}..."):
            if model_choice == "YOLO":
                model = load_yolo_model()
                model.set_classes(classes, model.get_text_pe(classes))
                results = model.predict(image, device=device, conf=conf)
                plotted_img = results[0].plot()
            elif model_choice == "Grounding DINO":
                model = load_dino_model()
                detections = model.predict_with_classes(image=image, classes=classes, box_threshold=box_threshold, text_threshold=text_threshold)
                plotted_img = annotate_dino_image(image, detections, classes)
            elif model_choice == "SAM3":
                predictor = load_sam_predictor()
                predictor.set_image(image)
                results = predictor(text=classes, conf=conf)
                plotted_img = annotate_sam_masks(image, results[0], classes)
        
        st.image(plotted_img, channels="BGR", use_container_width=True)

elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    frame_skip = st.slider("Process every Nth frame", 1, 20, 5)
    
    if uploaded_video is not None and classes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open video file")
        else:
            frame_placeholder = st.empty()
            st.write("Processing video...")
            frame_count = 0
            
            # Pre-load model
            if model_choice == "YOLO": model = load_yolo_model()
            elif model_choice == "Grounding DINO": model = load_dino_model()
            elif model_choice == "SAM3": predictor = load_sam_predictor()

            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                if frame_count % frame_skip != 0: continue

                if model_choice == "YOLO":
                    model.set_classes(classes, model.get_text_pe(classes))
                    results = model.predict(frame, device=device, conf=conf)
                    plotted_img = results[0].plot()
                elif model_choice == "Grounding DINO":
                    detections = model.predict_with_classes(image=frame, classes=classes, box_threshold=box_threshold, text_threshold=text_threshold)
                    plotted_img = annotate_dino_image(frame, detections, classes)
                elif model_choice == "SAM3":
                    predictor.set_image(frame)
                    results = predictor(text=classes, conf=conf)
                    plotted_img = annotate_sam_masks(frame, results[0], classes)
                
                frame_placeholder.image(plotted_img, channels="BGR", use_container_width=True)
            cap.release()
            st.write("Video processing completed.")
            os.unlink(video_path)

elif mode == "Live Camera":
    if classes:
        UnifiedVideoProcessor.update_params(model_choice, classes, conf, box_threshold, text_threshold, device)
        webrtc_streamer(
            key=f"unified-{model_choice.lower().replace(' ', '-')}",
            video_processor_factory=UnifiedVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )
    else:
        st.info("Please enter objects/classes to detect in the sidebar to start the live camera.")
