# Animated Talker

Generate lip-synced talking avatar videos from a single portrait image and an audio file.

**Input:** Avatar image (JPEG/PNG) + Audio file (MP3/WAV)
**Output:** Lip-synced video (MP4)

Powered by [MuseTalk v1.5](https://github.com/TMElyralab/MuseTalk) for real-time lip-sync generation.

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

### Install the package

```bash
pip install -e .
```

With optional post-processing (face restoration + upscaling):

```bash
pip install -e ".[postprocess]"
```

### Download model weights (~2.5 GB)

```bash
animated-talker download-models
```

## Usage

### Basic generation

```bash
animated-talker generate ^
    --image avatar.png ^
    --audio speech.mp3 ^
    --output result.mp4
```

### With post-processing

```bash
animated-talker generate ^
    --image avatar.png ^
    --audio speech.mp3 ^
    --output result.mp4 ^
    --face-restore ^
    --upscale
```

### All options

```
--image, -i         Path to portrait image (JPEG/PNG) [required]
--audio, -a         Path to audio file (MP3/WAV) [required]
--output, -o        Path for output video (MP4) [required]
--fps               Video frame rate (default: 25)
--bbox-shift        Jaw boundary shift in pixels (default: 0)
--face-restore      Apply CodeFormer face restoration
--upscale           Apply Real-ESRGAN 4x upscaling
--upscale-factor    Upscale multiplier (default: 4)
--restore-fidelity  CodeFormer fidelity 0.0-1.0 (default: 0.7)
--batch-size        Frames per inference batch (default: 8)
--no-float16        Disable FP16 inference (use FP32)
--device            PyTorch device: auto, cuda, mps, cpu (default: auto)
--model-dir         Custom model directory
--crf               Video quality 0-51, lower is better (default: 18)
--verbose, -v       Enable verbose logging
```

### Check model status

```bash
animated-talker check
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

### 7. Download model weights

```bash
animated-talker download-models
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
animated-talker generate ^
    --image avatar.png ^
    --audio speech.mp3 ^
    --output result.mp4 ^
    --batch-size 2
```

- FP16 is auto-disabled on CPU (not supported)
- Use `--batch-size 2` to reduce peak memory usage
- Expect slower generation (~5-10x compared to GPU)
- Use short audio clips (3-5 seconds) for quick testing

## Project Structure

```
src/animated_talker/
    __init__.py          # Public API: generate()
    pipeline.py          # Orchestrator
    input_handler.py     # Face detection + audio loading
    musetalk_engine.py   # MuseTalk v1.5 inference
    video_encoder.py     # FFmpeg encoding + audio mux
    postprocess.py       # CodeFormer + Real-ESRGAN (optional)
    model_manager.py     # Model weight downloads
```

## License

MIT
