# YOLO26 Real-Time Object Detection


https://github.com/user-attachments/assets/0b77b6c8-b7a9-4112-b82e-6873e8463654




A powerful application for real-time object detection and segmentation using YOLOE-26L with text prompts. Detect any object by simply typing what you're looking for. Available as both a web UI (Streamlit) and command-line interface (CLI).

## Features

- **Text-Prompted Detection**: Enter any object class (e.g., "person", "car", "dog") and the model will detect it
- **Two Interfaces**:
  - **Streamlit Web UI**: Interactive browser-based interface
  - **CLI**: Command-line interface with OpenCV window display
- **Multiple Input Modes**:
  - Upload/Process Images
  - Upload/Process Videos with configurable frame skip
  - Live Camera/Webcam: Real-time detection
- **Multiple Model Backends**:
  - **YOLOE-26L**: Fast real-time detection and segmentation
  - **Grounding DINO**: High-accuracy zero-shot object detection
  - **SAM3**: Advanced semantic segmentation
- **Instance Segmentation**: Uses YOLOE-26L-Seg and SAM3 models for pixel-level object segmentation
- **Hardware Acceleration**: 
  - **MPS** (Metal Performance Shaders) for Apple Silicon Macs - up to 10x faster
  - **CUDA** for NVIDIA GPUs
  - Automatic fallback to CPU

## Requirements

- Python 3.14+ (or 3.10+ for manual setup)
- macOS (with Apple Silicon for MPS), Linux, or Windows
- Webcam (optional, for live camera mode)
- Model file: `yoloe-26l-seg.pt`

## Quick Start

### Installation (using UV)

1. **Install UV** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and setup**:
```bash
git clone <repository-url>
cd realtime-detection-yolo26
# Install with all model backends (YOLO, DINO, SAM)
uv sync --all-extras
```

3. **Run the application**:

**Web UI:**
```bash
uv run streamlit run app.py
```

**CLI:**
```bash
# Detect persons in an image using YOLO (default)
uv run python cli.py image photo.jpg --prompt person

# Detect using Grounding DINO
uv run python cli.py image photo.jpg --type dino --prompt "person . car"

# Segment using SAM3
uv run python cli.py image photo.jpg --type sam --prompt "person"
```

See [QUICKSTART.md](QUICKSTART.md) for more details.

### Alternative Installation (using pip)

If you prefer not to use `uv`, you can install using standard `pip`:

1. Clone the repository:
```bash
git clone <repository-url>
cd realtime-detection-yolo26
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install .[yolo,dino,sam]
```

4. Run the application:
```bash
# Web UI
streamlit run app.py

# CLI
python cli.py image photo.jpg --prompt person
```

## Usage

### Streamlit Web UI

We provide a unified web interface that allows you to switch between different model backends in the sidebar:

| Model | Best For |
|-------|----------|
| YOLOE-26L | Real-time speed, basic segmentation |
| Grounding DINO | Complex queries, highest detection accuracy |
| SAM3 | Precise semantic segmentation |

To run the app:
```bash
uv run streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

#### Modes

**Upload Image**
1. Select "Upload Image" from the dropdown
2. Upload a JPG, PNG, or JPEG image
3. Enter the object class to detect
4. View the detection results with segmentation masks

**Upload Video**
1. Select "Upload Video" from the dropdown
2. Upload an MP4, AVI, or MOV video file
3. Enter the object class to detect
4. Adjust the frame skip slider (higher values = faster processing)
5. Watch the processed video with detections

**Live Camera**
1. Select "Live Camera" from the dropdown
2. Enter the object class to detect
3. Allow browser access to your webcam
4. View real-time detections

### Command Line Interface (CLI)

The unified CLI tool (`cli.py`) supports all model backends using the `--type` flag (`yolo`, `dino`, or `sam`).

#### Examples

```bash
# YOLO (Default) - Fast processing
uv run python cli.py image photo.jpg --prompt person

# Grounding DINO - High accuracy zero-shot detection
# Use " . " as a separator for multiple classes.
uv run python cli.py video input.mp4 --type dino --prompt "car . truck" --output result.mp4

# SAM3 - Advanced semantic segmentation
uv run python cli.py webcam --type sam --prompt "person"
```

For complete CLI documentation, see [CLI_GUIDE.md](CLI_GUIDE.md).

#### CLI Features

- **OpenCV Window Display**: Real-time visualization of detections
- **Save Results**: Output processed images and videos
- **Frame Skipping**: Process every Nth frame for faster video processing
- **Interactive Controls**: 
  - Press 'Q' to quit during video/webcam
  - Press 'S' to save frames during webcam mode
- **Device Selection**: Auto-detect or manually specify MPS/CUDA/CPU
- **Headless Mode**: Run without display for servers/scripts

#### CLI Examples

```bash
# Use specific camera
uv run python cli.py webcam --prompt face --camera-id 1

# Process every 5th frame
uv run python cli.py video large.mp4 --prompt bus --frame-skip 5

# Force CPU usage
uv run python cli.py image photo.jpg --prompt person --device cpu

# Use custom YOLO model
uv run python cli.py image photo.jpg --type yolo --prompt custom --model my_model.pt
```

For complete CLI documentation, see [CLI_GUIDE.md](CLI_GUIDE.md).

## Hardware Acceleration

The application automatically detects and uses the best available hardware:

- **MPS (Metal Performance Shaders)**: Apple Silicon Macs (M1/M2/M3)
  - Requires macOS 12.3 or later
  - Up to 10x faster than CPU
  - Enabled by default on compatible systems

- **CUDA**: NVIDIA GPUs on Linux/Windows
  - Automatically detected when available
  - Significant speedup for inference

- **CPU**: Fallback for all systems
  - Works everywhere but slower

To verify which device is being used:
- **Web UI**: Look for "Using device: MPS/CUDA/CPU" message
- **CLI**: Device is printed during model loading

For more details, see [DEVICE_SUPPORT.md](DEVICE_SUPPORT.md).

## Models

This project supports multiple state-of-the-art vision models:

1.  **YOLOE-26L-Seg**: A text-promptable object detection and segmentation model from Ultralytics. Supports open-vocabulary detection.
2.  **Grounding DINO**: A state-of-the-art zero-shot object detector that can detect any object described by a text prompt with high accuracy.
3.  **SAM3 (Segment Anything Model 3)**: The latest version of the Segment Anything Model, providing high-quality semantic segmentation.

## Project Structure

```
realtime-detection-yolo26/
├── app.py                 # Unified Streamlit app (YOLO/DINO/SAM)
├── cli.py                 # Unified CLI tool (YOLO/DINO/SAM)
├── pyproject.toml         # Project config (UV/pip)
├── uv.lock               # UV dependency lock file
├── yoloe-26l-seg.pt      # YOLO model weights (not in repo)
├── sam3.pt               # SAM3 model weights (not in repo)
├── test_device.py        # Device detection test script
├── README.md             # This file
├── QUICKSTART.md         # Quick start guide
├── CLI_GUIDE.md          # Unified CLI documentation
├── DINO_SAM_GUIDE.md     # DINO and SAM usage guide
└── DEVICE_SUPPORT.md     # Hardware acceleration details
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started quickly with all models
- **[CLI_GUIDE.md](CLI_GUIDE.md)** - YOLO CLI documentation and examples
- **[DINO_SAM_GUIDE.md](DINO_SAM_GUIDE.md)** - Detailed guide for DINO and SAM
- **[DEVICE_SUPPORT.md](DEVICE_SUPPORT.md)** - MPS/CUDA/CPU support details

## Testing

Test that everything is working:

```bash
# Test device detection
uv run python test_device.py

# Test web UI (YOLO)
uv run streamlit run app.py
```

## Performance Tips

1. **Use hardware acceleration**: MPS on Mac, CUDA on Linux/Windows with NVIDIA GPU
2. **Frame skipping**: For videos, use `--frame-skip` or adjust slider in web UI
3. **Resolution**: Lower resolution cameras/videos process faster
4. **First inference**: Initial model warmup takes longer, subsequent frames are faster

## Troubleshooting

### MPS Not Available
- Ensure you have macOS 12.3 or later
- Verify you have Apple Silicon (M1/M2/M3)
- Will automatically fall back to CPU

### Camera Not Found
- Check camera permissions in System Settings (macOS)
- Try different `--camera-id` values (0, 1, 2, etc.)
- Ensure camera is not in use by another application

### Model File Not Found
- Ensure `yoloe-26l-seg.pt` is in the project directory
- Or specify path with `--model` flag in CLI

For more troubleshooting, see [QUICKSTART.md](QUICKSTART.md) and [CLI_GUIDE.md](CLI_GUIDE.md).

## License

See the [Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for model usage terms.
