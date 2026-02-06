# Animated Talker

Generate lip-synced talking avatar videos from a single portrait image and an audio file.

**Input:** Avatar image (JPEG/PNG) + Audio file (MP3/WAV)
**Output:** Lip-synced video (MP4)

Powered by [MuseTalk v1.5](https://github.com/TMElyralab/MuseTalk) for real-time lip-sync generation.

## Quick Start

### 1. Install the package

```bash
git clone https://github.com/cnacha-mfu/animated-talker.git
cd animated-talker
pip install -e .
```

### 2. Setup MuseTalk (clone repo + install deps + download models)

**Run this first before generating any video:**

```bash
animated-talker setup
```

This will:
- Clone the MuseTalk repo to `~/.cache/animated_talker/MuseTalk`
- Install MuseTalk's Python dependencies (mmlab, etc.)
- Download all model weights (~2.5 GB) to the correct locations

To skip dependency installation (if you already have them):

```bash
animated-talker setup --no-deps
```

### 3. Generate a video

```bash
animated-talker generate \
    --image avatar.png \
    --audio speech.mp3 \
    --output result.mp4 \
    -v
```

### Python API

```python
from animated_talker import generate

generate(image="avatar.png", audio="speech.mp3", output="result.mp4")
```

## Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) (must be on PATH)
- [Git](https://git-scm.com/) (to clone MuseTalk repo)
- NVIDIA GPU with CUDA 11.8+ (recommended) or CPU (slower)

## CLI Reference

### Generate video

```bash
animated-talker generate \
    --image avatar.png \
    --audio speech.mp3 \
    --output result.mp4
```

Options:

```
--image, -i         Path to portrait image (JPEG/PNG) [required]
--audio, -a         Path to audio file (MP3/WAV) [required]
--output, -o        Path for output video (MP4) [required]
--fps               Video frame rate (default: 25)
--bbox-shift        Jaw boundary shift in pixels (default: 0)
--batch-size        Frames per inference batch (default: 8)
--no-float16        Disable FP16 inference (use FP32)
--device            PyTorch device: auto, cuda, mps, cpu (default: auto)
--extra-margin      Extra margin below jaw in pixels (default: 10)
--verbose, -v       Enable verbose logging
```

### Check status

```bash
animated-talker check
```

Shows which components are installed and which models are downloaded.

## Testing on Google Colab (Free T4 GPU)

```python
# Clone the repo
!git clone https://github.com/cnacha-mfu/animated-talker.git
%cd animated-talker

# Install the package
!pip install -e .

# Setup MuseTalk (clone + deps + models) — do this first!
!animated-talker setup -v

# Upload your image and audio, then generate
!animated-talker generate \
    --image avatar.png \
    --audio speech.mp3 \
    --output result.mp4 \
    -v
```

If you need to pull updates:

```python
!git fetch origin claude/add-lip-sync-video-XQKYM
!git reset --hard origin/claude/add-lip-sync-video-XQKYM
!pip install -e .
```

## Testing on GCP with GPU (Windows)

If you don't have a local GPU, you can spin up a GCP VM with a GPU to test.

### 1. Install Google Cloud CLI

Download and install the Google Cloud SDK for Windows:
https://cloud.google.com/sdk/docs/install#windows

After installation, open **PowerShell** and authenticate:

```powershell
gcloud init
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Request GPU quota (first time only)

Before creating a GPU VM, you need GPU quota in your project:

1. Go to https://console.cloud.google.com/iam-admin/quotas
2. Filter by **"GPUs (all regions)"**
3. Select the quota and click **Edit Quotas**
4. Request an increase to **1**
5. Wait for approval (usually a few minutes)

### 3. Create a GPU VM

Run in PowerShell:

```powershell
gcloud compute instances create animated-talker-test `
    --zone=us-central1-a `
    --machine-type=n1-standard-8 `
    --accelerator=type=nvidia-tesla-t4,count=1 `
    --boot-disk-size=100GB `
    --image-family=pytorch-latest-gpu `
    --image-project=deeplearning-platform-release `
    --maintenance-policy=TERMINATE
```

> The `pytorch-latest-gpu` image includes Python, PyTorch, CUDA, and FFmpeg pre-installed.

### 4. SSH into the VM

```powershell
gcloud compute ssh animated-talker-test --zone=us-central1-a
```

### 5. Verify GPU is available

```bash
nvidia-smi
```

You should see the T4 GPU with 16 GB VRAM.

### 6. Clone and install on the VM

```bash
git clone https://github.com/cnacha-mfu/animated-talker.git
cd animated-talker
pip install -e .
```

### 7. Setup MuseTalk (clone repo + install deps + download models)

**Run this before generating:**

```bash
animated-talker setup -v
```

### 8. Upload test files from Windows

Open a **new PowerShell window** on your local machine:

```powershell
gcloud compute scp C:\path\to\avatar.png C:\path\to\speech.mp3 `
    animated-talker-test:~/animated-talker/ `
    --zone=us-central1-a
```

### 9. Generate the video (on the VM)

```bash
cd ~/animated-talker
animated-talker generate \
    --image avatar.png \
    --audio speech.mp3 \
    --output result.mp4 \
    -v
```

### 10. Download the result to Windows

In your local PowerShell:

```powershell
gcloud compute scp `
    animated-talker-test:~/animated-talker/result.mp4 `
    C:\Users\YourName\Downloads\ `
    --zone=us-central1-a
```

### 11. Stop or delete the VM (avoid charges)

```powershell
# Stop the VM (keeps disk, no compute charges)
gcloud compute instances stop animated-talker-test --zone=us-central1-a

# Start it again later
gcloud compute instances start animated-talker-test --zone=us-central1-a

# Or delete entirely when done
gcloud compute instances delete animated-talker-test --zone=us-central1-a
```

### Cost estimate

| GPU | Cost/hr | VRAM | ~Speed (30s video) |
|-----|---------|------|--------------------|
| T4  | ~$0.35  | 16 GB | ~30-60s           |
| L4  | ~$0.70  | 24 GB | ~20-40s           |
| A100 | ~$3.67 | 40 GB | ~10-15s           |

> Use Spot VMs for ~70% discount by adding `--provisioning-model=SPOT` to the create command.

## Running on CPU (no GPU)

The tool auto-detects your hardware. On a machine without a GPU, it falls back to CPU automatically:

```bash
animated-talker generate \
    --image avatar.png \
    --audio speech.mp3 \
    --output result.mp4 \
    --batch-size 2
```

- FP16 is auto-disabled on CPU (not supported)
- Use `--batch-size 2` to reduce peak memory usage
- Expect slower generation (~5-10x compared to GPU)
- Use short audio clips (3-5 seconds) for quick testing

## Architecture

Animated Talker is a thin wrapper around MuseTalk's inference code:

```
src/animated_talker/
    __init__.py          # Public API: generate()
    pipeline.py          # Orchestrator (device detection, setup, engine call)
    musetalk_engine.py   # Thin wrapper around MuseTalk's inference functions
    model_manager.py     # Clone MuseTalk repo + download model weights
    postprocess.py       # CodeFormer + Real-ESRGAN (optional, future)
    cli.py               # CLI: generate, setup, download-models, check
```

Models are stored at `~/.cache/animated_talker/MuseTalk/models/`:

```
models/
    musetalkV15/         # UNet + config
    sd-vae/              # Stable Diffusion VAE
    whisper/             # Whisper tiny (audio features)
    dwpose/              # DWPose (face landmark detection)
    face-parse-bisent/   # BiSeNet (face parsing for blending)
```

## License

MIT
