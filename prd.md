# PRD: Animated Talker — Lip-Sync Talking Avatar Video Generator

**Version:** 3.0
**Date:** 2026-02-07
**Status:** Draft

---

## 1. Executive Summary

Animated Talker is a Python library and CLI tool that generates lip-synced talking avatar videos from a single portrait image and an audio file (MP3/WAV). The system uses Ditto-TalkingHead for audio-driven talking head synthesis with optional post-processing for quality enhancement. Fully self-hosted and open-source.

---

## 2. Problem Statement

Creating talking-head videos from a still portrait and audio requires stitching together multiple ML models (face detection, audio encoding, motion generation, neural rendering, video encoding) with significant glue code. There is no simple, production-ready Python tool that takes an image + audio and outputs a lip-synced video in one call.

---

## 3. Goals & Non-Goals

### Goals

| # | Goal |
|---|------|
| G1 | Single image + audio → lip-synced video in one function call |
| G2 | High-quality lip sync using Ditto-TalkingHead (diffusion-based) |
| G3 | Controllable emotion, head pose, and gaze |
| G4 | Optional face restoration (CodeFormer) and upscaling (Real-ESRGAN) |
| G5 | Simple Python API: `generate(image, audio, output)` |
| G6 | CLI for non-developers |
| G7 | Auto-download model weights from HuggingFace Hub |
| G8 | Self-hosted, no external API calls |

### Non-Goals

- Text-to-speech (TTS) — user provides audio
- Voice cloning
- Real-time streaming (future consideration)
- Web UI
- Training custom models
- Full-body animation (Ditto supports head-to-body stitching but not full body)

---

## 4. Pipeline

```
avatar.png + speech.mp3
        │
        ▼
  ┌─────────────┐
  │  Validation  │  Verify image & audio are readable
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   Ditto     │  HuBERT audio encoding
  │ TalkingHead │  → LMDM diffusion motion generation
  │             │  → Neural rendering & compositing
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Post-Process│  (optional) CodeFormer + Real-ESRGAN
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
    emotion=4,               # Emotion style (0-8)
    sampling_timesteps=50,    # Diffusion steps
    face_restore=True,        # CodeFormer
    upscale=True,             # Real-ESRGAN
)
```

### CLI

```bash
animated-talker generate --image avatar.png --audio speech.mp3 --output result.mp4
animated-talker generate --image avatar.png --audio speech.mp3 --output result.mp4 --emotion 2
animated-talker download-models  # Pre-download all weights + clone Ditto repo
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
| Git | Clone Ditto repo |

### Model Weights (~6.9 GB)

| Model | Purpose | License |
|-------|---------|---------|
| Ditto LMDM (v0.4 HuBERT) | Audio → motion diffusion | Apache-2.0 |
| Appearance Extractor | Identity features | Apache-2.0 |
| Motion Extractor | Motion parameters | Apache-2.0 |
| Warp Network | 3D mesh warping | Apache-2.0 |
| Decoder | Neural rendering | Apache-2.0 |
| Stitch Network | Head-to-body compositing | Apache-2.0 |
| HuBERT ONNX | Audio feature extraction | Apache-2.0 |
| Face detection ONNX | Face detection & landmarks | Non-commercial (InsightFace) |
| CodeFormer (optional) | Face restoration | Apache-2.0 |
| Real-ESRGAN (optional) | Super-resolution | BSD-3 |

---

## 7. Project Structure

```
animated-talker/
├── pyproject.toml
├── src/animated_talker/
│   ├── __init__.py          # Public API
│   ├── pipeline.py          # Orchestrator
│   ├── ditto_engine.py      # Ditto-TalkingHead inference engine
│   ├── input_handler.py     # Input validation
│   ├── video_encoder.py     # FFmpeg encoding + audio mux
│   ├── postprocess.py       # CodeFormer + Real-ESRGAN
│   └── model_manager.py     # Model weight + repo downloads
├── tests/
│   ├── test_pipeline.py
│   ├── test_input_handler.py
│   ├── test_model_manager.py
│   └── test_video_encoder.py
└── README.md
```

---

## 8. Licensing

MIT License for the library. Ditto-TalkingHead is Apache-2.0 licensed. Note: InsightFace models used for face detection have a non-commercial license, which may restrict commercial use.
