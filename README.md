# Animated Talker

Generate lip-synced talking avatar videos from a single portrait image and an audio file.

**Input:** Avatar image (JPEG/PNG) + Audio file (MP3/WAV)
**Output:** Lip-synced video (MP4)

Powered by [Ditto-TalkingHead](https://github.com/antgroup/ditto-talkinghead) — a diffusion-based framework for controllable, real-time talking head synthesis.

## Quick Start

### Python API

```python
from animated_talker import generate

generate(image="avatar.png", audio="speech.mp3", output="result.mp4")
```

### CLI

```bash
animated-talker generate --image avatar.png --audio speech.mp3 --output result.mp4
```

## Installation

### Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) (must be on PATH)
- NVIDIA GPU with CUDA 11.8+ (recommended) or CPU (slower)
- Git (for cloning the Ditto repo during model download)

### Install the package

```bash
pip install -e .
```

With optional post-processing (face restoration + upscaling):

```bash
pip install -e ".[postprocess]"
```

### Download model weights and Ditto repo (~6.9 GB)

```bash
animated-talker download-models
```

This will:
1. Clone the [Ditto-TalkingHead](https://github.com/antgroup/ditto-talkinghead) repository
2. Download PyTorch model weights from HuggingFace (`digital-avatar/ditto-talkinghead`)
3. Download ONNX models for audio processing and face detection

## Usage

### Basic generation

```bash
animated-talker generate \
    --image avatar.png \
    --audio speech.mp3 \
    --output result.mp4
```

### With emotion and diffusion control

```bash
animated-talker generate \
    --image avatar.png \
    --audio speech.mp3 \
    --output result.mp4 \
    --emotion 4 \
    --sampling-timesteps 50
```

### With post-processing

```bash
animated-talker generate \
    --image avatar.png \
    --audio speech.mp3 \
    --output result.mp4 \
    --face-restore \
    --upscale
```

### All options

```
--image, -i           Path to portrait image (JPEG/PNG) [required]
--audio, -a           Path to audio file (MP3/WAV) [required]
--output, -o          Path for output video (MP4) [required]
--fps                 Video frame rate (default: 25)
--emotion             Emotion style index 0-8 (default: 4 = neutral)
--sampling-timesteps  Diffusion denoising steps (default: 50, lower = faster)
--fade-in             Frames to fade in at start (-1 = disabled)
--fade-out            Frames to fade out at end (-1 = disabled)
--face-restore        Apply CodeFormer face restoration
--upscale             Apply Real-ESRGAN upscaling
--upscale-factor      Upscale multiplier (default: 4)
--restore-fidelity    CodeFormer fidelity 0.0-1.0 (default: 0.7)
--no-float16          Disable FP16 inference (use FP32)
--device              PyTorch device: auto, cuda, mps, cpu (default: auto)
--model-dir           Custom model directory
--verbose, -v         Enable verbose logging
```

### Check model status

```bash
animated-talker check
```

## How It Works

Ditto-TalkingHead uses a three-stage pipeline:

1. **Audio Feature Extraction** — HuBERT encodes audio into semantic features
2. **Motion Generation** — A Latent Motion Diffusion Model (LMDM) based on a Diffusion Transformer generates identity-agnostic facial motion sequences from audio features
3. **Photorealistic Rendering** — Appearance extraction, 3D mesh warping, and neural decoding synthesize video frames, which are composited back onto the original portrait

The system supports fine-grained control over emotion, gaze, and head pose, and achieves real-time inference speeds on modern GPUs.

## Testing on GCP with GPU

If you don't have a local GPU, you can spin up a GCP VM with a GPU to test.

### 1. Create a GPU VM

```bash
gcloud compute instances create animated-talker-test \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --boot-disk-size=100GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE
```

### 2. SSH and install

```bash
gcloud compute ssh animated-talker-test --zone=us-central1-a

git clone https://github.com/cnacha-mfu/animated-talker.git
cd animated-talker
pip install -e .
animated-talker download-models
```

### 3. Generate

```bash
animated-talker generate \
    --image avatar.png \
    --audio speech.mp3 \
    --output result.mp4 \
    -v
```

### 4. Clean up

```bash
# Stop VM (keeps disk)
gcloud compute instances stop animated-talker-test --zone=us-central1-a

# Or delete entirely
gcloud compute instances delete animated-talker-test --zone=us-central1-a
```

## Project Structure

```
src/animated_talker/
    __init__.py          # Public API: generate()
    pipeline.py          # Orchestrator
    ditto_engine.py      # Ditto-TalkingHead inference engine
    input_handler.py     # Input validation
    video_encoder.py     # FFmpeg encoding + audio mux
    postprocess.py       # CodeFormer + Real-ESRGAN (optional)
    model_manager.py     # Model weight + repo downloads
```

## License

MIT (this library). Ditto-TalkingHead is licensed under Apache-2.0.
