#!/usr/bin/env python3
"""
Vision Model Hub CLI - Unified Command Line Interface for YOLO, Grounding DINO, and SAM3
Supports image, video, and webcam detection with OpenCV window display
"""

import argparse
import cv2
import torch
import sys
import re
import numpy as np
from pathlib import Path
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
    if not text:
        return []
    parts = re.split(r'[.,]+', text)
    return [p.strip() for p in parts if p.strip()]

# --- DINO Specific Helpers ---

def get_dino_config_path() -> str:
    """Return the bundled GroundingDINO SwinT config path."""
    try:
        import groundeddino_vl
        pkg_dir = Path(groundeddino_vl.__file__).parent
        candidates = [
            pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py",
            pkg_dir / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
            pkg_dir / "GroundingDINO_SwinT_OGC.py",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        matches = list(pkg_dir.rglob("GroundingDINO_SwinT_OGC.py"))
        if matches:
            return str(matches[0])
    except ImportError:
        pass
    return None

# --- Annotation Functions ---

def annotate_dino_image(image_bgr, detections: sv.Detections, labels: list):
    """Draw bounding boxes and confidence labels using Supervision."""
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
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

# --- Predictor Wrappers ---

class YOLOPredictor:
    def __init__(self, model_path, device):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
    
    def predict(self, image, classes, **kwargs):
        if classes:
            self.model.set_classes(classes, self.model.get_text_pe(classes))
        results = self.model.predict(image, device=self.device, conf=kwargs.get("conf", 0.25), verbose=False)
        return results[0].plot()

class DINOPredictor:
    def __init__(self, device, config_path=None, weights_path=None):
        from groundeddino_vl.utils.inference import Model
        from groundeddino_vl.weights_manager import download_model_weights
        if weights_path is None:
            weights_path = download_model_weights()
        if config_path is None:
            config_path = get_dino_config_path()
        self.model = Model(model_config_path=config_path, model_checkpoint_path=weights_path, device=device)
    
    def predict(self, image, classes, **kwargs):
        caption = " . ".join(classes)
        detections, labels = self.model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=kwargs.get("box_threshold", 0.35),
            text_threshold=kwargs.get("text_threshold", 0.25)
        )
        return annotate_dino_image(image, detections, labels)

class SAMPredictor:
    def __init__(self, device, model_path="sam3.pt"):
        overrides = dict(conf=0.25, task="segment", mode="predict", model=model_path, verbose=False, device=device)
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
    
    def predict(self, image, classes, **kwargs):
        self.predictor.set_image(image)
        results = self.predictor(text=classes, conf=kwargs.get("conf", 0.25))
        return annotate_sam_masks(image, results[0], classes)

# --- Core Processing Logic ---

def process_mode(predictor, args, classes):
    if args.mode == "image":
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not read image {args.input}")
            return
        
        plotted_img = predictor.predict(
            image, classes, 
            conf=args.conf, 
            box_threshold=args.box_threshold, 
            text_threshold=args.text_threshold
        )
        
        if args.output:
            cv2.imwrite(args.output, plotted_img)
            print(f"Saved to {args.output}")
        
        if not args.no_show:
            cv2.imshow("Detection", plotted_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif args.mode == "video":
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open video {args.input}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                if frame_count % args.frame_skip != 0:
                    if writer: writer.write(frame)
                    continue
                
                plotted_img = predictor.predict(
                    frame, classes,
                    conf=args.conf,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold
                )
                
                if writer: writer.write(plotted_img)
                if not args.no_show:
                    cv2.imshow("Detection - Press Q to quit", plotted_img)
                    if cv2.waitKey(1) & 0xFF == ord("q"): break
        finally:
            cap.release()
            if writer: writer.release()
            cv2.destroyAllWindows()

    elif args.mode == "webcam":
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera_id}")
            return
        
        print("Press 'q' to quit, 's' to save current frame")
        saved_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                plotted_img = predictor.predict(
                    frame, classes,
                    conf=args.conf,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold
                )
                
                cv2.imshow("Webcam Detection - Q to quit, S to save", plotted_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"): break
                elif key == ord("s"):
                    save_name = f"capture_{saved_count:04d}.jpg"
                    cv2.imwrite(save_name, plotted_img)
                    print(f"Saved {save_name}")
                    saved_count += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Unified Vision CLI")
    parser.add_argument("mode", choices=["image", "video", "webcam"], help="Processing mode")
    parser.add_argument("input", nargs="?", help="Input file (not for webcam)")
    parser.add_argument("--type", choices=["yolo", "dino", "sam"], default="yolo", help="Model type")
    parser.add_argument("--model", type=str, help="Model path (default depends on type)")
    parser.add_argument("--prompt", type=str, help='Classes/Prompt (e.g., "person . car")')
    parser.add_argument("--output", "-o", type=str, help="Output path")
    parser.add_argument("--no-show", action="store_true", help="No GUI")
    parser.add_argument("--device", default="auto", help="mps, cuda, cpu, or auto")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="DINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="DINO text threshold")
    parser.add_argument("--frame-skip", type=int, default=1, help="Video frame skip")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam ID")

    args = parser.parse_args()

    if args.mode in ["image", "video"] and not args.input:
        parser.error(f"{args.mode} requires an input file")

    device = get_device() if args.device == "auto" else args.device
    
    classes = parse_classes(args.prompt)

    try:
        if args.type == "yolo":
            model_path = args.model or "yoloe-26l-seg.pt"
            predictor = YOLOPredictor(model_path, device)
        elif args.type == "dino":
            predictor = DINOPredictor(device, config_path=None, weights_path=args.model)
        elif args.type == "sam":
            model_path = args.model or "sam3.pt"
            predictor = SAMPredictor(device, model_path)
        
        process_mode(predictor, args, classes)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

