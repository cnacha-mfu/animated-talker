# PRD: Animated Talker — Lip-Sync Talking Avatar Video Generator

**Version:** 2.0
**Date:** 2026-02-06
**Status:** Draft

---

## 1. Executive Summary

Animated Talker is a Python library and CLI tool that generates lip-synced talking avatar videos from a single portrait image and an audio file (MP3/WAV). The system uses MuseTalk v1.5 for lip-sync generation with optional post-processing for quality enhancement. Fully self-hosted and open-source.

---

## 2. Problem Statement

Creating talking-head videos from a still portrait and audio requires stitching together multiple ML models (face detection, lip sync, video encoding) with significant glue code. There is no simple, production-ready Python tool that takes an image + audio and outputs a lip-synced video in one call.

---

## 3. Goals & Non-Goals

### Goals

| # | Goal |
|---|------|
| G1 | Single image + audio → lip-synced video in one function call |
| G2 | Accurate lip sync using MuseTalk v1.5 |
| G3 | Optional face restoration (CodeFormer) and upscaling (Real-ESRGAN) |
| G4 | Simple Python API: `generate(image, audio, output)` |
| G5 | CLI for non-developers |
| G6 | Auto-download model weights from HuggingFace Hub |
| G7 | Self-hosted, no external API calls |

### Non-Goals

- Text-to-speech (TTS) — user provides audio
- Voice cloning
- Real-time streaming
- Web UI
- Plugin/backend abstraction system
- Training custom models
- Full-body animation

---

## 4. Pipeline

```
avatar.png + speech.mp3
        │
        ▼
  ┌─────────────┐
  │ Input Handler│  Face detection, alignment, audio loading
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  MuseTalk   │  Audio features → lip-synced frames
  │  v1.5       │  (single-step latent inpainting)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Post-Process│  (optional) CodeFormer + Real-ESRGAN
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Video Encode│  FFmpeg H.264 + audio mux
  └──────┬──────┘
         ▼
    output.mp4
```

---

## 5. API Design

### Python API

```python
from animated_talker import generate

# Simple usage
generate(image="avatar.png", audio="speech.mp3", output="result.mp4")

# With options
generate(
    image="avatar.png",
    audio="speech.mp3",
    output="result.mp4",
    fps=25,
    face_restore=True,      # CodeFormer
    upscale=True,            # Real-ESRGAN 4x
    bbox_shift=0,            # Jaw boundary adjustment
)
```

### CLI

```bash
animated-talker --image avatar.png --audio speech.mp3 --output result.mp4
animated-talker --image avatar.png --audio speech.mp3 --output result.mp4 --face-restore --upscale
animated-talker download-models  # Pre-download all weights
```

---

## 6. Technical Requirements

### Hardware

| Tier | GPU | VRAM |
|------|-----|------|
| Minimum | NVIDIA RTX 3060 | 8 GB |
| Recommended | NVIDIA RTX 4080 | 16 GB |

### Software

| Dependency | Purpose |
|------------|---------|
| Python ≥ 3.10 | Runtime |
| PyTorch ≥ 2.0 | Model inference |
| CUDA ≥ 11.8 | GPU acceleration |
| FFmpeg ≥ 5.0 | Video encoding |

### Model Weights (~2.5 GB)

| Model | Size | License |
|-------|------|---------|
| MuseTalk v1.5 (UNet + config) | ~900 MB | MIT |
| SD VAE (ft-mse) | ~330 MB | MIT |
| Whisper tiny | ~150 MB | MIT |
| Face parsing (BiSeNet) | ~50 MB | MIT |
| CodeFormer (optional) | ~350 MB | Apache 2.0 |
| Real-ESRGAN (optional) | ~64 MB | BSD-3 |

---

## 7. Project Structure

```
animated-talker/
├── pyproject.toml
├── src/animated_talker/
│   ├── __init__.py          # Public API
│   ├── pipeline.py          # Orchestrator
│   ├── input_handler.py     # Image + audio loading, face detection
│   ├── musetalk_engine.py   # MuseTalk v1.5 inference
│   ├── video_encoder.py     # FFmpeg encoding + audio mux
│   ├── postprocess.py       # CodeFormer + Real-ESRGAN
│   └── model_manager.py     # Model weight downloads
├── tests/
│   ├── test_input_handler.py
│   ├── test_video_encoder.py
│   └── test_pipeline.py
└── cli.py                   # CLI entry point
```

---

## 8. Licensing

MIT License for the library. Default models (MuseTalk, SD VAE, Whisper, CodeFormer, Real-ESRGAN) are all permissively licensed for commercial use.
