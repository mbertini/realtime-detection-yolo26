# AGENTS.md

## Scope
- This repo is a unified vision app with two entry points over shared backend concepts: Streamlit UI in `app.py` and CLI in `cli.py`.
- Backends are intentionally parallel across both surfaces: YOLOE (`yoloe-26l-seg.pt`), Grounding DINO (`groundeddino_vl`), and SAM3 (`sam3.pt`).

## Architecture You Need First
- `app.py` owns interactive web flow: sidebar picks backend + thresholds, then mode (`Upload Image`, `Upload Video`, `Live Camera`).
- `cli.py` owns automation/script flow: argparse mode dispatch (`image|video|webcam`) + predictor wrappers (`YOLOPredictor`, `DINOPredictor`, `SAMPredictor`).
- Device policy is duplicated and must stay consistent in both files: `get_device()` chooses `mps > cuda > cpu`.
- Live camera path differs by interface: `streamlit_webrtc` + `UnifiedVideoProcessor.recv()` in `app.py` vs OpenCV capture loop in `cli.py`.

## Backend Contracts and Data Flow
- Prompt parsing uses `parse_classes()` with regex split on commas/periods in both `app.py` and `cli.py`.
- DINO is tuned around caption-style prompts and thresholds (`box_threshold`, `text_threshold`); CLI joins classes with `" . "` in `DINOPredictor.predict()`.
- YOLO requires per-request class injection before predict (`model.set_classes(classes, model.get_text_pe(classes))`).
- SAM3 path is stateful per image/frame (`predictor.set_image(...)` before inference) and returns masks that are manually overlay-annotated.
- DINO/SAM visualization is custom (`annotate_dino_image`, `annotate_sam_masks`), not Ultralytics default plotting.

## Dependencies and Integration Points
- Dependency truth lives in `pyproject.toml`; backend-specific packages are in optional extras (`[project.optional-dependencies]`).
- CUDA wheels are excluded on macOS via markers; do not remove `sys_platform != 'darwin'` guards unless you also adjust runtime docs (`DEVICE_SUPPORT.md`).
- DINO model weights/config are resolved dynamically (`download_model_weights()`, `get_dino_config_path()`).

## Developer Workflows (Canonical)
```bash
uv sync --all-extras
uv run streamlit run app.py
uv run python cli.py image photo.jpg --prompt person
uv run python cli.py image photo.jpg --type dino --prompt "person . car"
uv run python test_device.py
```

## Project-Specific Conventions
- Keep CLI and Streamlit option semantics aligned (same defaults for `conf`, `box_threshold`, `text_threshold`, prompt behavior).
- Preserve model default filenames (`yoloe-26l-seg.pt`, `sam3.pt`) because docs and quickstart assume root-level weights.
- For video mode, maintain frame-skip behavior where skipped frames are still written when output writer is active (`process_mode` in `cli.py`).
- Prefer minimal backend branching by extending predictor wrappers in `cli.py` and mirrored branches in `app.py`.

## Change Guardrails for Agents
- If you change device selection, update all of: `app.py`, `cli.py`, `test_device.py`, and docs (`README.md`, `DEVICE_SUPPORT.md`).
- If you change prompt parsing or threshold names/defaults, update both interfaces and `CLI_GUIDE.md`/`DINO_SAM_GUIDE.md` examples.
- Validate at least one real command path after edits (CLI or Streamlit startup) and include exact command in PR notes.

