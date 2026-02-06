# PRD: Animated Talker — High-Quality Lip-Sync Talking Avatar Engine

**Version:** 1.0
**Date:** 2026-02-06
**Status:** Draft

---

## 1. Executive Summary

Animated Talker is a Python library and CLI tool that generates high-quality lip-synced talking avatar videos from a single portrait image and text (or audio) input. The system targets HeyGen-level quality — photorealistic facial animation with natural head motion, micro-expressions, accurate lip sync, and optional emotion-aware delivery — while being fully self-hosted and open-source.

The product serves two audiences: developers who integrate it as a Python library (`pip install animated-talker`) into their own applications, and end-users who run a CLI or web UI to produce talking avatar videos directly.

---

## 2. Problem Statement

Creating professional talking-head videos today requires either:

1. **Commercial SaaS** (HeyGen, D-ID, Synthesia) — expensive at scale ($30–500+/month), data leaves your infrastructure, limited customization, vendor lock-in.
2. **Raw open-source models** (Wav2Lip, SadTalker, MuseTalk) — fragmented ecosystem, each model solves one piece, significant glue code needed, no unified pipeline from text to final video, inconsistent quality.

There is no production-ready, self-hosted Python library that stitches together the full pipeline (TTS + lip sync + face restoration + video encoding) with quality approaching commercial solutions.

---

## 3. Goals & Non-Goals

### Goals

| # | Goal | Success Metric |
|---|------|----------------|
| G1 | Single-image-to-video: generate a talking avatar from one portrait photo + text | Works with any front-facing portrait (photo, illustration, AI-generated) |
| G2 | HeyGen-competitive lip sync accuracy | LSE-D ≤ 7.0 on LRS2 benchmark; MOS ≥ 4.0 in human eval (5-point scale) |
| G3 | Natural head motion and micro-expressions | Head pose variance comparable to real video; emotion labels supported |
| G4 | High visual fidelity | Output at 512×512 minimum, upscalable to 1080p; FID ≤ 15 on HDTF |
| G5 | Multilingual TTS with voice cloning | Support 10+ languages out of the box; 5-second reference clip for cloning |
| G6 | Developer-friendly Python API | `pip install`, 3-line quickstart, async support, typed interfaces |
| G7 | CLI and Gradio web UI | Non-developers can generate videos without writing code |
| G8 | Real-time streaming mode | ≤ 500ms first-frame latency for interactive avatar use cases |
| G9 | Self-hosted, no external API calls required | All inference runs locally; no data leaves the machine |

### Non-Goals (v1)

- Training custom models from scratch (we consume pretrained weights)
- Full-body avatar animation (torso, hands, gestures)
- Live camera-driven face reenactment (driving video input)
- Mobile or edge deployment (GPU server assumed)
- SaaS hosting or multi-tenant API platform

---

## 4. Target Users

| Persona | Description | Primary Use Case |
|---------|-------------|------------------|
| **App Developer** | Integrates avatar generation into their product (e-learning, marketing, chatbots) | Python library API |
| **Content Creator** | Makes talking-head videos for social media, courses, presentations | CLI / Web UI |
| **Enterprise Team** | Needs on-prem avatar generation for compliance (healthcare, finance, gov) | Self-hosted deployment |
| **Researcher** | Experiments with talking-head generation, benchmarks models | Modular pipeline, swappable backends |

---

## 5. System Architecture

### 5.1 High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                        Animated Talker                           │
│                                                                  │
│  ┌─────────┐   ┌───────────┐   ┌──────────┐   ┌─────────────┐  │
│  │  Input   │──▶│    TTS    │──▶│ Lip Sync │──▶│    Post-    │  │
│  │ Handler  │   │  Engine   │   │  Engine  │   │ Processing  │  │
│  └─────────┘   └───────────┘   └──────────┘   └─────────────┘  │
│       │              │               │               │           │
│       ▼              ▼               ▼               ▼           │
│  [Portrait]    [Audio WAV]    [Raw Frames]    [Final Video]     │
│  [Text/Audio]  [Timestamps]   [256×256]       [1080p MP4]       │
│                [Phonemes]                                        │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Module Breakdown

#### Module 1: Input Handler
- Accepts portrait image (JPEG/PNG, any aspect ratio)
- Accepts text string OR pre-recorded audio file (WAV/MP3)
- Face detection and alignment using MediaPipe or InsightFace
- Validates face is front-facing, well-lit, unobstructed
- Crops and normalizes face region to model input size

#### Module 2: TTS Engine (Text-to-Speech)
- **Primary backend:** Kokoro TTS (Apache 2.0, 82M params, CPU-capable)
- **Voice cloning backend:** Coqui XTTS-v2 (6-second reference clip)
- **Alternative backends:** Fish Audio S1-mini, Chatterbox (MIT), OpenAI TTS API
- Outputs: WAV audio + word-level timestamps + phoneme alignment
- Pluggable interface — any TTS that produces audio + timestamps can be used

#### Module 3: Lip Sync Engine (Core)
- **Primary backend:** MuseTalk v1.5 (MIT license, 30+ FPS on V100)
- **Alternative backends:** FLOAT (ICCV 2025), LivePortrait + audio driver, Wav2Lip
- Consumes: portrait image + audio waveform
- Produces: frame sequence with synchronized lip/jaw/face motion
- Supports emotion labels (neutral, happy, sad, surprised, angry) for expression control
- Face region size: 256×256 native, composited onto original image

#### Module 4: Post-Processing Pipeline
- **Face restoration:** CodeFormer or GFPGAN for artifact removal and detail enhancement
- **Super-resolution:** Real-ESRGAN for upscaling to 1080p
- **Temporal smoothing:** Gaussian-weighted frame blending to reduce inter-frame jitter
- **Video encoding:** FFmpeg with H.264/H.265, configurable bitrate and FPS
- **Audio muxing:** Merge generated/input audio with video output

### 5.3 Streaming Architecture (Real-Time Mode)

```
Client (WebSocket/WebRTC)
    │
    ▼
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Audio Chunk   │────▶│  Lip Sync      │────▶│  Frame         │
│  Receiver      │     │  (per-frame)   │     │  Streamer      │
└────────────────┘     └────────────────┘     └────────────────┘
                                                     │
                                              WebRTC/MJPEG ──▶ Client
```

- FastAPI backend with WebSocket endpoint for audio input
- Per-frame generation using MuseTalk's single-step inference (not diffusion)
- Frame streaming via WebRTC (preferred) or MJPEG fallback
- Target: ≤500ms start-up latency, 30 FPS sustained

---

## 6. Technical Requirements

### 6.1 Hardware Requirements

| Tier | GPU | VRAM | Use Case |
|------|-----|------|----------|
| Minimum | NVIDIA RTX 3060 | 8 GB | Offline generation, batch mode |
| Recommended | NVIDIA RTX 4080 / A10 | 16 GB | Real-time mode, 1080p output |
| Production | NVIDIA A100 / H100 | 40–80 GB | Multi-stream, high throughput |
| CPU-only | Any modern CPU | 16 GB RAM | TTS-only (Kokoro); no lip sync |

### 6.2 Software Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | ≥ 3.10, < 3.13 | Runtime |
| PyTorch | ≥ 2.2 | Model inference |
| CUDA | ≥ 11.8 | GPU acceleration |
| FFmpeg | ≥ 5.0 | Video encoding/muxing |
| MediaPipe / InsightFace | Latest | Face detection & alignment |

### 6.3 Model Weights & Storage

| Model | Size (approx.) | License |
|-------|----------------|---------|
| MuseTalk v1.5 | ~2.5 GB | MIT |
| Kokoro TTS | ~350 MB | Apache 2.0 |
| Coqui XTTS-v2 | ~1.8 GB | Non-commercial* |
| CodeFormer | ~350 MB | Apache 2.0 |
| Real-ESRGAN | ~64 MB | BSD-3 |
| FLOAT | ~1.5 GB | Research** |

\* XTTS-v2 is Coqui Public Model License (non-commercial). For commercial use, substitute with Chatterbox (MIT) or Kokoro.
\** FLOAT license to be confirmed; fallback to MuseTalk (MIT) for commercial.

Auto-download on first use via Hugging Face Hub. Total storage: ~7 GB for full pipeline.

---

## 7. API Design

### 7.1 Python Library API

```python
from animated_talker import Talker, TTSEngine, LipSyncBackend

# Initialize with defaults
talker = Talker()

# Simple: text + image → video
video_path = talker.generate(
    image="avatar.png",
    text="Hello, welcome to our product demo.",
    output="output.mp4",
)

# Advanced: full control
talker = Talker(
    tts_engine=TTSEngine.KOKORO,
    lip_sync_backend=LipSyncBackend.MUSETALK,
    face_restore=True,
    upscale=True,           # 1080p via Real-ESRGAN
    output_fps=30,
    emotion="happy",
)

video_path = talker.generate(
    image="avatar.png",
    text="This is amazing technology!",
    voice="en-us-1",        # or path to reference audio for cloning
    output="output.mp4",
)

# Audio input instead of text
video_path = talker.generate(
    image="avatar.png",
    audio="narration.wav",
    output="output.mp4",
)

# Async support
video_path = await talker.agenerate(
    image="avatar.png",
    text="Async generation for web apps.",
    output="output.mp4",
)
```

### 7.2 CLI Interface

```bash
# Basic usage
animated-talker generate \
    --image avatar.png \
    --text "Hello world" \
    --output output.mp4

# With voice cloning
animated-talker generate \
    --image avatar.png \
    --text "Cloned voice speaking" \
    --voice-ref my_voice.wav \
    --tts xtts \
    --output output.mp4

# From audio file
animated-talker generate \
    --image avatar.png \
    --audio narration.wav \
    --emotion happy \
    --upscale \
    --output output.mp4

# Real-time streaming server
animated-talker serve \
    --image avatar.png \
    --host 0.0.0.0 \
    --port 8000

# Download models
animated-talker download --all
animated-talker download --models musetalk kokoro codeformer
```

### 7.3 Web UI (Gradio)

```bash
animated-talker ui
# Opens browser at http://localhost:7860
```

Features:
- Drag-and-drop portrait image upload
- Text input with language/voice selection
- Audio file upload as alternative to TTS
- Emotion selector (neutral, happy, sad, surprised, angry)
- Quality presets (draft / standard / high)
- Side-by-side preview before download
- Real-time progress bar with ETA

### 7.4 Streaming API

```python
from animated_talker import StreamingTalker

streamer = StreamingTalker(image="avatar.png")

# Start streaming session
session = await streamer.start()

# Send audio chunks, receive video frames
async for frame in session.stream_audio(audio_chunk):
    yield frame  # numpy array or encoded bytes

await session.stop()
```

---

## 8. Quality Strategy

### 8.1 Quality Targets (HeyGen Parity)

| Dimension | Metric | Target | HeyGen Estimated |
|-----------|--------|--------|------------------|
| Lip sync accuracy | LSE-D (lower=better) | ≤ 7.0 | ~6.5 |
| Lip sync confidence | LSE-C (higher=better) | ≥ 7.5 | ~8.0 |
| Visual quality | FID (lower=better) | ≤ 15 | ~10–12 |
| Identity preservation | CSIM (higher=better) | ≥ 0.85 | ~0.90 |
| Video smoothness | FVD (lower=better) | ≤ 200 | ~150 |
| Perceptual quality | MOS 1-5 (higher=better) | ≥ 4.0 | ~4.3 |
| Resolution | Pixels | 1080p | 1080p / 4K |
| Frame rate | FPS | 25–30 | 24–30 |

### 8.2 Quality Enhancement Stack

The pipeline applies successive quality enhancements:

1. **Raw lip sync output** (256×256, may have artifacts) → baseline
2. **+ CodeFormer** (face restoration, fidelity weight 0.7) → removes artifacts, sharpens facial details
3. **+ Real-ESRGAN** (4× upscale) → brings to 1024×1024 or higher
4. **+ Temporal smoothing** (3-frame Gaussian kernel) → reduces jitter between frames
5. **+ Color correction** (histogram matching to source portrait) → preserves skin tone consistency

Each stage is optional and togglable via API/CLI flags for speed vs. quality tradeoff.

### 8.3 Evaluation Framework

Built-in evaluation commands for benchmarking:

```bash
animated-talker evaluate \
    --dataset hdtf \
    --metrics lse-d lse-c fid csim fvd \
    --output results.json
```

- Automated evaluation on standard datasets (HDTF, LRS2, LRS3)
- Human eval toolkit: generates paired comparisons for MOS scoring
- Regression testing: CI runs evaluation suite against quality thresholds on every release

---

## 9. Backend Abstraction & Extensibility

### 9.1 Plugin Interface

Every pipeline stage uses an abstract base class so backends are swappable:

```python
class TTSBackend(Protocol):
    def synthesize(self, text: str, voice: str, language: str) -> TTSResult:
        """Returns audio waveform + word timestamps + phonemes."""
        ...

class LipSyncBackend(Protocol):
    def generate(self, image: np.ndarray, audio: np.ndarray,
                 emotion: str | None) -> Iterator[np.ndarray]:
        """Yields frames with lip-synced face."""
        ...

class FaceRestorationBackend(Protocol):
    def restore(self, frame: np.ndarray, fidelity: float) -> np.ndarray:
        """Returns restored frame."""
        ...

class SuperResolutionBackend(Protocol):
    def upscale(self, frame: np.ndarray, scale: int) -> np.ndarray:
        """Returns upscaled frame."""
        ...
```

### 9.2 Built-in Backends

| Stage | Backend | Default | License |
|-------|---------|---------|---------|
| TTS | Kokoro | Yes | Apache 2.0 |
| TTS | Coqui XTTS-v2 | No | Non-commercial |
| TTS | Chatterbox | No | MIT |
| TTS | Fish Audio S1-mini | No | CC-BY-NC |
| TTS | OpenAI TTS (API) | No | Commercial |
| Lip Sync | MuseTalk v1.5 | Yes | MIT |
| Lip Sync | FLOAT | No | Research |
| Lip Sync | Wav2Lip | No | Non-commercial |
| Lip Sync | LivePortrait | No | MIT |
| Face Restore | CodeFormer | Yes | Apache 2.0 |
| Face Restore | GFPGAN | No | Apache 2.0 |
| Upscale | Real-ESRGAN | Yes | BSD-3 |

### 9.3 Custom Backend Registration

```python
from animated_talker import register_backend

@register_backend("tts", "my_custom_tts")
class MyTTS:
    def synthesize(self, text, voice, language):
        # Custom implementation
        ...

talker = Talker(tts_engine="my_custom_tts")
```

---

## 10. Project Structure

```
animated-talker/
├── pyproject.toml              # Package metadata, dependencies
├── README.md
├── LICENSE                     # MIT
├── prd.md                      # This document
│
├── src/
│   └── animated_talker/
│       ├── __init__.py         # Public API exports
│       ├── talker.py           # Main Talker class
│       ├── streaming.py        # StreamingTalker for real-time
│       ├── cli.py              # CLI entry point (Click/Typer)
│       ├── ui.py               # Gradio web UI
│       ├── config.py           # Configuration & defaults
│       ├── types.py            # Shared types / dataclasses
│       │
│       ├── input/
│       │   ├── __init__.py
│       │   ├── face_detect.py  # Face detection & alignment
│       │   ├── image.py        # Image loading & preprocessing
│       │   └── audio.py        # Audio loading & validation
│       │
│       ├── tts/
│       │   ├── __init__.py
│       │   ├── base.py         # TTSBackend protocol
│       │   ├── kokoro.py       # Kokoro TTS backend
│       │   ├── xtts.py         # Coqui XTTS-v2 backend
│       │   ├── chatterbox.py   # Chatterbox backend
│       │   └── openai.py       # OpenAI TTS API backend
│       │
│       ├── lipsync/
│       │   ├── __init__.py
│       │   ├── base.py         # LipSyncBackend protocol
│       │   ├── musetalk.py     # MuseTalk v1.5 backend
│       │   ├── float_model.py  # FLOAT backend
│       │   ├── wav2lip.py      # Wav2Lip backend
│       │   └── liveportrait.py # LivePortrait backend
│       │
│       ├── postprocess/
│       │   ├── __init__.py
│       │   ├── face_restore.py # CodeFormer / GFPGAN
│       │   ├── upscale.py      # Real-ESRGAN
│       │   ├── temporal.py     # Temporal smoothing
│       │   └── color.py        # Color correction
│       │
│       ├── video/
│       │   ├── __init__.py
│       │   ├── encoder.py      # FFmpeg video encoding
│       │   └── composer.py     # Frame compositing onto original
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   └── downloader.py   # HuggingFace Hub model downloader
│       │
│       └── eval/
│           ├── __init__.py
│           ├── metrics.py      # LSE-D, LSE-C, FID, CSIM, FVD
│           └── benchmark.py    # Dataset evaluation runner
│
├── tests/
│   ├── test_talker.py
│   ├── test_tts.py
│   ├── test_lipsync.py
│   ├── test_postprocess.py
│   └── test_streaming.py
│
├── examples/
│   ├── basic_usage.py
│   ├── voice_cloning.py
│   ├── streaming_server.py
│   └── batch_processing.py
│
├── docker/
│   ├── Dockerfile              # GPU-enabled container
│   └── docker-compose.yml      # Full stack with Gradio UI
│
└── docs/
    ├── quickstart.md
    ├── backends.md
    └── deployment.md
```

---

## 11. Phased Delivery Plan

### Phase 1: Core Pipeline (MVP)

**Objective:** End-to-end text → talking avatar video generation.

| Deliverable | Details |
|-------------|---------|
| Input handler | Face detection, alignment, preprocessing via MediaPipe |
| Kokoro TTS integration | Text → audio + word timestamps |
| MuseTalk v1.5 integration | Audio + portrait → lip-synced frames |
| Basic video encoder | FFmpeg H.264, 25 FPS, 512×512 |
| Python API | `Talker.generate()` with text + image input |
| CLI | `animated-talker generate` command |
| Model downloader | Auto-download from Hugging Face Hub |
| Tests | Unit tests for each module, integration test for full pipeline |

**Exit criteria:** Generate a 30-second talking avatar video from any front-facing portrait + English text. LSE-D ≤ 8.0.

### Phase 2: Quality Enhancement

**Objective:** Reach HeyGen-competitive visual quality.

| Deliverable | Details |
|-------------|---------|
| CodeFormer integration | Face restoration post-processing |
| Real-ESRGAN integration | 4× upscale to 1080p output |
| Temporal smoothing | Reduce inter-frame jitter |
| Color correction | Histogram matching to preserve skin tones |
| Emotion support | Emotion labels passed to lip sync backend |
| Evaluation framework | Automated LSE-D, FID, CSIM benchmarking |
| Quality presets | draft / standard / high profiles |

**Exit criteria:** LSE-D ≤ 7.0, FID ≤ 15, MOS ≥ 4.0 on HDTF evaluation set.

### Phase 3: Voice & Language

**Objective:** Multilingual support and voice cloning.

| Deliverable | Details |
|-------------|---------|
| XTTS-v2 backend | Voice cloning from 6-second reference |
| Chatterbox backend | MIT-licensed alternative for commercial use |
| Multi-language TTS | 10+ languages via Kokoro + XTTS |
| Audio input mode | Skip TTS, use pre-recorded audio directly |
| Phoneme alignment | Improved sync via forced alignment (Montreal Forced Aligner) |

**Exit criteria:** Voice cloning MOS ≥ 3.8 for similarity. Works in EN, ES, FR, DE, ZH, JA, KO, PT, IT, RU.

### Phase 4: Real-Time Streaming

**Objective:** Interactive avatar for chatbot/assistant use cases.

| Deliverable | Details |
|-------------|---------|
| StreamingTalker API | Async frame-by-frame generation |
| FastAPI server | WebSocket endpoint for audio streaming |
| WebRTC output | Low-latency video delivery to browser |
| Gradio web UI | Interactive playground with all features |
| Session management | Multiple concurrent avatar sessions |

**Exit criteria:** ≤500ms first-frame latency. 30 FPS sustained on RTX 4080. 5+ concurrent sessions on A100.

### Phase 5: Production Hardening

**Objective:** Production-ready deployment.

| Deliverable | Details |
|-------------|---------|
| Docker images | GPU-enabled containers with all models pre-loaded |
| ONNX / TensorRT export | Optimized inference for 2–3× speedup |
| Batch processing | Process multiple videos in parallel |
| FLOAT backend | ICCV 2025 model for premium quality tier |
| LivePortrait backend | Alternative high-quality animation |
| Monitoring & logging | Structured logging, Prometheus metrics |
| Documentation | Full API reference, deployment guide, tutorials |

**Exit criteria:** Stable API (semantic versioning). Docker deployment works out of the box. Throughput: 10+ videos/minute on A100.

---

## 12. Performance Targets

| Scenario | Hardware | Target | Metric |
|----------|----------|--------|--------|
| 30s video, standard quality | RTX 4080 | ≤ 60s | Generation time |
| 30s video, high quality (1080p) | RTX 4080 | ≤ 120s | Generation time |
| 30s video, draft quality | RTX 3060 | ≤ 45s | Generation time |
| Real-time streaming | RTX 4080 | 30 FPS | Sustained framerate |
| Real-time first frame | RTX 4080 | ≤ 500ms | Latency |
| Batch (10 videos) | A100 | ≤ 5 min | Total throughput |
| Model loading (cold start) | Any GPU | ≤ 15s | Time to first inference |

---

## 13. Licensing Strategy

The library itself is released under the **MIT License** for maximum adoption.

Default backends (Kokoro + MuseTalk + CodeFormer + Real-ESRGAN) are all permissively licensed (MIT / Apache 2.0 / BSD-3), enabling **full commercial use** out of the box without any licensing restrictions.

Non-commercial backends (XTTS-v2, Wav2Lip, Fish Audio) are clearly marked in documentation and require explicit opt-in. The library emits a warning when a non-commercial backend is selected.

---

## 14. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| MuseTalk quality insufficient for HeyGen parity | High | Medium | FLOAT and LivePortrait as fallback backends; community fine-tuning |
| Model weights become unavailable on HuggingFace | Medium | Low | Mirror weights; include download checksums; support local paths |
| VRAM constraints limit adoption | High | Medium | Offer quality tiers; CPU fallback for TTS; quantization (FP16/INT8) |
| Temporal jitter in generated videos | Medium | High | Temporal smoothing post-processing; explore video-level generation models |
| Voice cloning quality/similarity gap | Medium | Medium | Multiple TTS backends; user can supply pre-recorded audio |
| Upstream model breaking changes | Medium | Medium | Pin model versions; abstract behind backend interface |
| Legal/ethical misuse (deepfakes) | High | Medium | Watermarking in output metadata; content policy in docs; optional visible watermark |

---

## 15. Ethical Considerations & Safeguards

| Safeguard | Implementation |
|-----------|---------------|
| Invisible watermark | C2PA-compatible metadata embedded in every generated video |
| Visible watermark | Optional overlay indicating AI-generated content |
| Consent framework | Documentation requiring users to have consent for likeness |
| Content policy | Terms of use prohibiting impersonation, fraud, non-consensual content |
| Face verification | Optional mode that checks input image is not a known public figure |
| Audit logging | Generation requests logged with timestamps and input hashes |

---

## 16. Success Criteria

The project is considered successful when:

1. **Functional:** A user can `pip install animated-talker`, provide a portrait + text, and receive a lip-synced video in under 2 minutes on consumer GPU hardware.
2. **Quality:** Blind A/B tests show the output is preferred over SadTalker/Wav2Lip baselines ≥80% of the time, and is within striking distance of HeyGen (preferred ≤40% of the time against HeyGen).
3. **Adoption:** 500+ GitHub stars and 50+ pip installs/day within 3 months of public release.
4. **Reliability:** Full pipeline succeeds on ≥95% of valid input portraits without manual intervention.
5. **Extensibility:** A new TTS or lip-sync backend can be integrated in under 100 lines of code.

---

## 17. References & Prior Art

### Open-Source Models
- [Wav2Lip](https://www.wav2lip.org/) — Industry standard lip-sync for dubbing
- [SadTalker](https://github.com/OpenTalker/SadTalker) — CVPR 2023, single-image talking head with expression
- [MuseTalk](https://github.com/TMElyralab/MuseTalk) — Tencent, real-time 30+ FPS lip sync (MIT license)
- [LivePortrait](https://github.com/KlingTeam/LivePortrait) — Kuaishou, high-fidelity portrait animation
- [FLOAT](https://github.com/deepbrainai-research/float) — ICCV 2025, flow matching for audio-driven portraits
- [Linly-Talker](https://github.com/Kedreamix/Linly-Talker) — Multi-model digital human system

### TTS Engines
- [Kokoro TTS](https://github.com/hexgrad/kokoro) — 82M params, Apache 2.0, CPU-capable
- [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) — Voice cloning from 6-second clip
- [Chatterbox](https://github.com/resemble-ai/chatterbox) — Resemble AI, MIT license
- [Fish Audio](https://github.com/fishaudio/fish-speech) — Multilingual voice cloning

### Post-Processing
- [CodeFormer](https://github.com/sczhou/CodeFormer) — Transformer-based face restoration
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — GAN-based face restoration
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — Super-resolution upscaling

### Evaluation
- [THEval](https://arxiv.org/html/2511.04520) — Multi-dimensional talking head evaluation framework
- LSE-D / LSE-C metrics via [SyncNet](https://www.robots.ox.ac.uk/~vgg/software/lipsync/)

### Commercial Benchmarks
- [HeyGen](https://www.heygen.com/) — Avatar IV, 1080p+, emotion-aware, streaming API
- [D-ID](https://www.d-id.com/) — Real-time talking head API
- [Synthesia](https://www.synthesia.io/) — Enterprise avatar video platform

### Streaming Architecture
- [LiveAvatar (Alibaba)](https://github.com/Alibaba-Quark/LiveAvatar) — 45 FPS real-time streaming
- [SoulX-FlashTalk](https://arxiv.org/html/2512.23379) — Sub-second latency streaming avatars
- [HeyGen Streaming API](https://docs.heygen.com/docs/streaming-api) — WebRTC-based avatar streaming
