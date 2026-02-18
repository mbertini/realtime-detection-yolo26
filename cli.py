#!/usr/bin/env python3
"""
YOLO26 CLI - Command Line Interface for Object Detection
Supports image, video, and webcam detection with OpenCV window display
"""

import argparse
import cv2
import torch
import sys
from pathlib import Path
from ultralytics import YOLO


def get_device():
    """Detect and return the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model(model_path: str, device: str):
    """Load YOLO model and move to specified device"""
    print(f"Loading model: {model_path}")
    print(f"Using device: {device.upper()}")
    model = YOLO(model_path)
    model.to(device)
    return model


def process_image(model, image_path: str, class_name: str, device: str,
                 save_path: str = None, show: bool = True):
    """Process a single image"""
    print(f"Processing image: {image_path}")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Set classes for text prompt
    if class_name:
        names = [class_name]
        model.set_classes(names, model.get_text_pe(names))
        print(f"Detecting: {class_name}")

    # Run prediction
    results = model.predict(image, device=device)

    # Get the plotted result image
    plotted_img = results[0].plot()

    # Display results
    if results[0].boxes is not None:
        num_detections = len(results[0].boxes)
        print(f"Found {num_detections} detection(s)")

    # Save if output path provided
    if save_path:
        cv2.imwrite(save_path, plotted_img)
        print(f"Saved result to: {save_path}")

    # Show in OpenCV window
    if show:
        cv2.imshow('YOLO26 Detection', plotted_img)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(model, video_path: str, class_name: str, device: str,
                 save_path: str = None, show: bool = True, frame_skip: int = 1):
    """Process a video file"""
    print(f"Processing video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Processing every {frame_skip} frame(s)")

    # Set classes for text prompt
    if class_name:
        names = [class_name]
        model.set_classes(names, model.get_text_pe(names))
        print(f"Detecting: {class_name}")

    # Setup video writer if saving
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

            # Skip frames if specified
            if frame_count % frame_skip != 0:
                if writer:
                    writer.write(frame)
                continue

            # Run prediction
            results = model.predict(frame, device=device, verbose=False)
            plotted_img = results[0].plot()
            processed_count += 1

            # Write to output video
            if writer:
                writer.write(plotted_img)

            # Show in OpenCV window
            if show:
                cv2.imshow('YOLO26 Detection - Press Q to quit', plotted_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break

            # Progress indicator
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


def process_webcam(model, class_name: str, device: str, camera_id: int = 0):
    """Process webcam feed in real-time"""
    print(f"Opening webcam (camera_id={camera_id})...")

    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    # Set classes for text prompt
    if class_name:
        names = [class_name]
        model.set_classes(names, model.get_text_pe(names))
        print(f"Detecting: {class_name}")

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

            # Run prediction
            results = model.predict(frame, device=device, verbose=False)
            plotted_img = results[0].plot()

            # Add FPS counter
            if frame_count > 30:  # After warmup
                cv2.putText(plotted_img, f"Frame: {frame_count}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)

            # Show in OpenCV window
            cv2.imshow('YOLO26 Webcam Detection - Press Q to quit, S to save', plotted_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopped by user")
                break
            elif key == ord('s'):
                # Save current frame
                save_name = f"webcam_capture_{saved_count:04d}.jpg"
                cv2.imwrite(save_name, plotted_img)
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
        description='YOLO26 CLI - Object Detection with OpenCV Display',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect persons in an image
  %(prog)s image input.jpg --class person
  
  # Process video and save output
  %(prog)s video input.mp4 --class car --output result.mp4
  
  # Use webcam in real-time
  %(prog)s webcam --class person
  
  # Process video without display (headless)
  %(prog)s video input.mp4 --class bus --no-show --output output.mp4
  
  # Use custom model
  %(prog)s image photo.jpg --class cat --model custom_model.pt
        """
    )

    parser.add_argument('mode', choices=['image', 'video', 'webcam'],
                       help='Processing mode')
    parser.add_argument('input', nargs='?',
                       help='Input file path (not needed for webcam mode)')
    parser.add_argument('--class', dest='class_name', type=str,
                       help='Class name to detect (e.g., person, car, dog)')
    parser.add_argument('--model', '-m', type=str, default='yoloe-26l-seg.pt',
                       help='Path to YOLO model file (default: yoloe-26l-seg.pt)')
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

    args = parser.parse_args()

    # Validate input file for image/video modes
    if args.mode in ['image', 'video'] and not args.input:
        parser.error(f"{args.mode} mode requires input file path")

    # Validate input file exists
    if args.mode in ['image', 'video']:
        if not Path(args.input).exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

    # Determine device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
        # Validate device availability
        if device == 'mps' and not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            device = 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = 'cpu'

    # Check if model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Load model
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Process based on mode
    try:
        if args.mode == 'image':
            process_image(model, args.input, args.class_name, device,
                         args.output, not args.no_show)

        elif args.mode == 'video':
            process_video(model, args.input, args.class_name, device,
                         args.output, not args.no_show, args.frame_skip)

        elif args.mode == 'webcam':
            if args.output:
                print("Warning: --output is not supported in webcam mode")
            process_webcam(model, args.class_name, device, args.camera_id)

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

