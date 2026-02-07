# PRD: Animated Talker — Talking Avatar Video Generator

**Version:** 3.0
**Date:** 2026-02-07
**Status:** Draft

---

## 1. Executive Summary

Animated Talker is a Python library and CLI tool that generates talking avatar videos with natural head motion and lip sync from a single portrait image and an audio file (MP3/WAV). The system uses SadTalker (CVPR 2023) for audio-driven 3D face animation with optional post-processing for quality enhancement. Fully self-hosted and open-source.

---

## 2. Problem Statement

Creating talking-head videos from a still portrait and audio requires stitching together multiple ML models (face reconstruction, audio-to-expression, audio-to-pose, face rendering, video encoding) with significant glue code. There is no simple, production-ready Python tool that takes an image + audio and outputs a talking-head video with head motion in one call.

---

## 3. Goals & Non-Goals

### Goals

| # | Goal |
|---|------|
| G1 | Single image + audio -> talking-head video with head motion in one function call |
| G2 | Natural head movement and accurate lip sync using SadTalker (CVPR 2023) |
| G3 | Optional face restoration (CodeFormer) and upscaling (Real-ESRGAN) |
| G4 | Simple Python API: `generate(image, audio, output)` |
| G5 | CLI for non-developers |
| G6 | Auto-download model weights from HuggingFace Hub |
| G7 | Self-hosted, no external API calls |

### Non-Goals

- Text-to-speech (TTS) -- user provides audio
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
        |
        v
  +---------------+
  | Input Handler  |  Face detection, alignment, audio loading
  +-------+-------+
          v
  +---------------+
  | Face Recon     |  ResNet50 -> 257-dim 3DMM coefficients
  +-------+-------+
          v
  +---------------+
  | Audio2Exp      |  Mel spectrogram -> expression deltas (64-dim)
  | Audio2Pose     |  Mel spectrogram -> head pose sequence (6-DOF)
  +-------+-------+
          v
  +---------------+
  | Face Renderer  |  face-vid2vid: KPDetector + MappingNet + Generator
  | (SadTalker)    |  3DMM coefficients -> animated face frames
  +-------+-------+
          v
  +---------------+
  | Post-Process   |  (optional) CodeFormer + Real-ESRGAN
  +-------+-------+
          v
  +---------------+
  | Video Encode   |  FFmpeg H.264 + audio mux
  +-------+-------+
          v
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
    still_mode=True,         # Less head motion
    expression_scale=1.2,    # More expressive
    pose_style=5,            # Head motion style
    face_restore=True,       # CodeFormer
    upscale=True,            # Real-ESRGAN 4x
)
```

### CLI

```bash
animated-talker generate --image avatar.png --audio speech.mp3 --output result.mp4
animated-talker generate --image avatar.png --audio speech.mp3 --output result.mp4 --still-mode --expression-scale 1.2
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
| Python >= 3.10 | Runtime |
| PyTorch >= 2.0 | Model inference |
| CUDA >= 11.8 | GPU acceleration |
| FFmpeg >= 5.0 | Video encoding |

### Model Weights (~1.5 GB)

| Model | Size | Source |
|-------|------|--------|
| SadTalker V0.0.2 (KPDetector + HEEstimator + Generator) | ~725 MB | vinthony/SadTalker-V002rc |
| MappingNet (full motion) | ~156 MB | vinthony/SadTalker-V002rc |
| MappingNet (still mode) | ~156 MB | vinthony/SadTalker-V002rc |
| Face reconstruction (epoch_20) | ~289 MB | vinthony/SadTalker |
| Audio2Exp | ~34 MB | vinthony/SadTalker |
| Audio2Pose | ~96 MB | vinthony/SadTalker |
| CodeFormer (optional) | ~350 MB | Apache 2.0 |
| Real-ESRGAN (optional) | ~64 MB | BSD-3 |

---

## 7. Project Structure

```
animated-talker/
+-- pyproject.toml
+-- src/animated_talker/
|   +-- __init__.py          # Public API
|   +-- pipeline.py          # Orchestrator
|   +-- input_handler.py     # Image + audio loading, face detection
|   +-- sadtalker_engine.py  # SadTalker inference engine
|   +-- video_encoder.py     # FFmpeg encoding + audio mux
|   +-- postprocess.py       # CodeFormer + Real-ESRGAN
|   +-- model_manager.py     # Model weight downloads
|   +-- _sadtalker/          # Vendored network architectures
|       +-- audio_models.py  # Audio2Exp, Audio2Pose, CVAE
|       +-- renderer.py      # KPDetector, Generator, MappingNet
|       +-- animation.py     # Keypoint transformation
+-- tests/
|   +-- test_input_handler.py
|   +-- test_video_encoder.py
|   +-- test_model_manager.py
|   +-- test_pipeline.py
+-- cli.py                   # CLI entry point
```

---

## 8. Licensing

MIT License for the library. SadTalker models are released under MIT license for research use.
