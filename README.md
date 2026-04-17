# YOLO26 Real-Time Object Detection

A unified application for text-prompted object detection and segmentation using multiple backends:
- **YOLOE-26L-Seg** for fast real-time inference
- **Grounding DINO** for high-accuracy zero-shot detection
- **SAM3** for semantic segmentation

You can run the project as either:
- **Web UI** (`app.py`, Streamlit)
- **CLI** (`cli.py`, OpenCV window or headless mode)

## Features

- Text-prompted detection (`--prompt`)
- Multiple input modes: image, video, webcam
- Backend selection: `yolo`, `dino`, `sam`
- Automatic hardware selection: **MPS > CUDA > CPU**
- Optional output saving for image/video workflows
- Headless processing for scripts/servers

## Requirements

- Python `>=3.11` (from `pyproject.toml`)
- macOS, Linux, or Windows
- Optional webcam (for live mode)
- Model files in project root:
  - `yoloe-26l-seg.pt` (YOLO backend)
  - `sam3.pt` (SAM backend)

## Installation

### Recommended (UV)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repository-url>
cd realtime-detection-yolo26
uv sync --all-extras
```

### Selective extras

```bash
uv sync --extra yolo
uv sync --extra dino --extra sam
```

### Alternative (pip)

```bash
python -m venv venv
source venv/bin/activate
pip install .[yolo,dino,sam]
```

## Quick Start

### Web UI

```bash
uv run streamlit run app.py
```

Then open `http://localhost:8501` and select backend + mode in the sidebar.

### CLI

```bash
# YOLO (default)
uv run python cli.py image photo.jpg --prompt person

# Grounding DINO
uv run python cli.py image photo.jpg --type dino --prompt "person . car"

# SAM3
uv run python cli.py image photo.jpg --type sam --prompt "person"
```

## CLI Reference

### Syntax

```bash
uv run python cli.py <mode> [input] --type <yolo|dino|sam> --prompt "<classes>" [options]
```

### Modes

- `image` - process one image
- `video` - process a video file
- `webcam` - process live camera frames

### Common examples

```bash
# Save image output
uv run python cli.py image input.jpg --prompt car --output result.jpg

# Process video with frame skipping
uv run python cli.py video input.mp4 --prompt person --frame-skip 3 --output out.mp4

# Headless mode
uv run python cli.py video input.mp4 --prompt person --no-show --output out.mp4

# Webcam with a non-default camera
uv run python cli.py webcam --prompt person --camera-id 1
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--type` | Backend: `yolo`, `dino`, `sam` | `yolo` |
| `--prompt` | Prompt/classes string | None |
| `--model` | Custom model path | backend-specific |
| `--output`, `-o` | Output file path | None |
| `--no-show` | Disable OpenCV display | `False` |
| `--device` | `auto`, `mps`, `cuda`, `cpu` | `auto` |
| `--camera-id` | Webcam id | `0` |
| `--frame-skip` | Process every Nth video frame | `1` |
| `--conf` | Confidence (YOLO/SAM) | `0.25` |
| `--box-threshold` | Box threshold (DINO) | `0.35` |
| `--text-threshold` | Text threshold (DINO) | `0.25` |

### Interactive controls

- Webcam: press `q` to quit, `s` to save frame
- Video preview: press `q` to quit

## Hardware Acceleration

The app auto-selects the best available device:
- **MPS** (Apple Silicon M1/M2/M3/M4)
- **CUDA** (NVIDIA GPU)
- **CPU** fallback

Verify detection setup:

```bash
uv run python test_device.py
```

## Troubleshooting

### Camera not found
- Grant camera permissions to terminal/IDE
- Try a different `--camera-id` (0, 1, 2...)

### Model file missing
- Ensure `yoloe-26l-seg.pt` / `sam3.pt` exist in project root
- Or provide `--model /path/to/model.pt`

### MPS not available on macOS
- Requires macOS 12.3+ and Apple Silicon
- The app falls back to CPU automatically

### Slow inference
- First inference is slower due to model warmup
- Use `--frame-skip` for large videos
- Reduce input resolution if needed

## Project Structure

```text
realtime-detection-yolo26/
|- app.py
|- cli.py
|- test_device.py
|- yoloe-26l-seg.pt
|- sam3.pt
|- README.md
|- ARCHITECTURE.MD
|- pyproject.toml
`- uv.lock
```

## Documentation

- `README.md` - setup, quick start, and complete CLI usage
- `ARCHITECTURE.MD` - backend architecture, device/runtime model, and design details

## License

See the [Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for model usage terms.
