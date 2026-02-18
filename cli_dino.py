#!/usr/bin/env python3
"""
Grounding DINO CLI - Command Line Interface for Zero-Shot Object Detection
Supports image, video, and webcam detection with OpenCV window display
"""

import argparse
import cv2
import torch
import sys
from pathlib import Path

import supervision as sv
from groundeddino_vl.utils.inference import Model
from groundeddino_vl.weights_manager import download_model_weights


def get_device():
    """Detect and return the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_model_config_path() -> str:
    """Return the bundled GroundingDINO SwinT config path from the installed package."""
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
    raise FileNotFoundError(
        "GroundingDINO_SwinT_OGC.py config not found in groundeddino_vl package. "
        "Pass --config to specify it manually."
    )


def load_model(device: str, config_path: str = None, checkpoint_path: str = None):
    """Load Grounding DINO model with auto-downloaded weights."""
    if checkpoint_path is None:
        print("Downloading/loading model weights...")
        checkpoint_path = download_model_weights()
    if config_path is None:
        config_path = get_model_config_path()
    print(f"Loading model config: {config_path}")
    print(f"Loading model weights: {checkpoint_path}")
    print(f"Using device: {device.upper()}")
    model = Model(
        model_config_path=config_path,
        model_checkpoint_path=checkpoint_path,
        device=device,
    )
    return model


def annotate_image(image_bgr, detections: sv.Detections, labels: list):
    """Draw bounding boxes and confidence labels using Supervision annotators."""
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


def process_image(model, image_path: str, caption: str,
                  save_path: str = None, show: bool = True):
    """Process a single image."""
    print(f"Processing image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    print(f"Detecting: {caption}")
    detections, labels = model.predict_with_caption(
        image=image,
        caption=caption,
    )

    print(f"Found {len(detections)} detection(s)")
    for label, conf in zip(labels, detections.confidence):
        print(f"  {label}: {conf:.2f}")

    annotated = annotate_image(image, detections, labels)

    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Saved result to: {save_path}")

    if show:
        cv2.imshow('Grounding DINO Detection', annotated)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(model, video_path: str, caption: str,
                  save_path: str = None, show: bool = True, frame_skip: int = 1):
    """Process a video file."""
    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Processing every {frame_skip} frame(s)")
    print(f"Detecting: {caption}")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"Saving output to: {save_path}")

    frame_count = 0
    processed_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip != 0:
                if writer:
                    writer.write(frame)
                continue

            detections, labels = model.predict_with_caption(
                image=frame,
                caption=caption,
            )
            annotated = annotate_image(frame, detections, labels)
            processed_count += 1

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow('Grounding DINO Detection - Press Q to quit', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break

            if processed_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        print(f"\nProcessed {processed_count} frames")
        print("Done!")


def process_webcam(model, caption: str, camera_id: int = 0):
    """Process webcam feed in real-time."""
    print(f"Opening webcam (camera_id={camera_id})...")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print(f"Detecting: {caption}")
    print("Webcam opened successfully!")
    print("Press 'q' to quit, 's' to save current frame")

    frame_count = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break

            frame_count += 1

            detections, labels = model.predict_with_caption(
                image=frame,
                caption=caption,
            )
            annotated = annotate_image(frame, detections, labels)

            if frame_count > 30:
                cv2.putText(annotated, f"Frame: {frame_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            cv2.imshow('Grounding DINO Webcam - Press Q to quit, S to save', annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopped by user")
                break
            elif key == ord('s'):
                save_name = f"dino_capture_{saved_count:04d}.jpg"
                cv2.imwrite(save_name, annotated)
                print(f"Saved: {save_name}")
                saved_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        if saved_count > 0:
            print(f"Saved {saved_count} frame(s)")
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Grounding DINO CLI - Zero-Shot Object Detection with OpenCV Display',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect persons in an image
  %(prog)s image input.jpg --caption "person"

  # Detect multiple objects (use " . " as separator)
  %(prog)s image input.jpg --caption "person . car . dog"

  # Process video and save output
  %(prog)s video input.mp4 --caption "car . truck" --output result.mp4

  # Use webcam in real-time
  %(prog)s webcam --caption "person"

  # Process video without display (headless)
  %(prog)s video input.mp4 --caption "bus" --no-show --output output.mp4

  # Use custom model config and weights
  %(prog)s image photo.jpg --caption "cat" --config path/to/config.py --weights path/to/weights.pth
        """
    )

    parser.add_argument('mode', choices=['image', 'video', 'webcam'],
                        help='Processing mode')
    parser.add_argument('input', nargs='?',
                        help='Input file path (not needed for webcam mode)')
    parser.add_argument('--caption', type=str, required=True,
                        help='Text prompt: objects separated by " . " (e.g., "person . car . dog")')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to GroundingDINO config .py file (auto-detected if omitted)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to GroundingDINO weights .pth file (auto-downloaded if omitted)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path for saving results')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display OpenCV window (useful for headless systems)')
    parser.add_argument('--device', type=str, choices=['auto', 'mps', 'cuda', 'cpu'],
                        default='auto',
                        help='Device to use for inference (default: auto)')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera ID for webcam mode (default: 0)')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process every Nth frame in video mode (default: 1)')
    parser.add_argument('--box-threshold', type=float, default=0.35,
                        help='Bounding box confidence threshold (default: 0.35)')
    parser.add_argument('--text-threshold', type=float, default=0.25,
                        help='Text-image matching threshold (default: 0.25)')

    args = parser.parse_args()

    if args.mode in ['image', 'video'] and not args.input:
        parser.error(f"{args.mode} mode requires input file path")

    if args.mode in ['image', 'video']:
        if not Path(args.input).exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
        if device == 'mps' and not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            device = 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = 'cpu'

    try:
        model = load_model(device, config_path=args.config, checkpoint_path=args.weights)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Patch thresholds onto the model if non-default values supplied
    if args.box_threshold != 0.35 or args.text_threshold != 0.25:
        model._box_threshold = args.box_threshold
        model._text_threshold = args.text_threshold

    try:
        if args.mode == 'image':
            process_image(model, args.input, args.caption,
                          args.output, not args.no_show)

        elif args.mode == 'video':
            process_video(model, args.input, args.caption,
                          args.output, not args.no_show, args.frame_skip)

        elif args.mode == 'webcam':
            if args.output:
                print("Warning: --output is not supported in webcam mode")
            process_webcam(model, args.caption, args.camera_id)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
