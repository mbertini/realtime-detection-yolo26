# Architecture and Runtime Model

This document consolidates model/backend behavior and hardware support details for the project.

## 1) System Overview

The repository exposes two entry points over the same backend family:

- `app.py`: Streamlit web UI with three modes (image, video, live camera)
- `cli.py`: command-line workflow with three modes (`image`, `video`, `webcam`)

Both interfaces support the same backend choices:
- `yolo` / `YOLO`: YOLOE-26L-Seg via Ultralytics
- `dino` / `Grounding DINO`: zero-shot text-guided detection
- `sam` / `SAM3`: semantic segmentation via Ultralytics SAM3 predictor

## 2) Backend Architecture

### YOLOE-26L-Seg

- Loads from `yoloe-26l-seg.pt`
- Prompt classes are injected dynamically per request/frame
- Best for low-latency, real-time use

### Grounding DINO

- Uses `groundeddino_vl` inference wrapper
- Prompts are caption-like strings (typically class names joined by ` . `)
- Exposes `box_threshold` and `text_threshold` to tune precision/recall behavior

### SAM3

- Loads from `sam3.pt`
- Performs text-prompted semantic segmentation
- Produces masks that are blended and annotated with boxes/labels when available

## 3) Runtime Flow

### Web UI (`app.py`)

- User selects backend and mode from sidebar/UI controls
- Input is decoded (image/video/webcam frame)
- Selected backend runs inference
- Output is rendered with backend-specific annotation

### CLI (`cli.py`)

- Argparse dispatches by `mode`
- Predictor wrapper selected by `--type`
- Frames/images are processed and optionally displayed (`--no-show`) and/or saved (`--output`)
- Video loop supports `--frame-skip`

## 4) Device Selection and Execution Policy

Both entry points implement the same priority order:

```python
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
```

### Priority

1. **MPS** (Apple Silicon)
2. **CUDA** (NVIDIA GPU)
3. **CPU** fallback

### Manual override (CLI)

```bash
uv run python cli.py image photo.jpg --prompt person --device cpu
uv run python cli.py image photo.jpg --prompt person --device mps
uv run python cli.py image photo.jpg --prompt person --device cuda
```

## 5) Platform Support Matrix

| Platform | Acceleration | Notes |
|----------|--------------|-------|
| macOS (Apple Silicon M1/M2/M3/M4) | MPS | Requires macOS 12.3+ |
| Linux / Windows + NVIDIA GPU | CUDA | Uses CUDA-enabled torch stack |
| Any platform | CPU | Always available |

## 6) Dependency Layout (from `pyproject.toml`)

- Core dependencies include Streamlit/OpenCV/torch runtime and shared utilities.
- Optional backend extras:
  - `yolo`: Ultralytics + CLIP dependency
  - `sam`: Ultralytics + CLIP dependency
  - `dino`: `groundeddino_vl`, `supervision`, and `transformers`
- CUDA-specific packages (`nvidia-*-cu12`, `triton`) are excluded on macOS via platform markers.

## 7) Prompt and Threshold Semantics

### Prompt parsing

- CLI accepts `--prompt` and splits prompt text into classes.
- DINO usage is most reliable with ` . ` separators for multi-class prompts.
- SAM/YOLO also accept single-class prompts cleanly.

### Thresholds

- `--conf`: confidence threshold for YOLO and SAM
- `--box-threshold`: DINO box filtering
- `--text-threshold`: DINO text-image matching

## 8) Performance Notes

- First inference is slower because of model/runtime warmup.
- DINO and SAM are typically heavier than YOLO.
- Video throughput can be improved with `--frame-skip`.
- Lower input resolution generally improves latency.

## 9) Verification and Diagnostics

Use the bundled script to verify available acceleration:

```bash
uv run python test_device.py
```

Typical Apple Silicon output includes:
- `MPS available: True`
- `CUDA available: False`
- `Selected device: MPS`

## 10) Common Failure Modes

### MPS unavailable on macOS

- Confirm Apple Silicon hardware and macOS 12.3+
- Check torch MPS availability in Python
- Fallback to CPU is automatic

### CUDA unavailable on Linux/Windows

- Verify NVIDIA driver/toolkit installation (`nvidia-smi`)
- Re-sync dependencies when needed (`uv sync --reinstall`)

### Slow runtime

- Expect warmup on first inference
- Increase `--frame-skip` for video
- Use lower resolution inputs
