# Running OpenShorts fully local + campaign tools

This fork runs the clip pipeline on your own machine: transcription, scene
detection, reframing and rendering in Docker, and the AI moment picker on a
local model through Ollama. No Gemini key, no Upload-Post, nothing leaves the box.

## 1. One-time setup (needs sudo, run these yourself)

The Arch `ollama` package is CPU-only. For the GPU build:

```bash
sudo pacman -S ollama-cuda
```

Optional, for GPU transcription/rendering inside Docker (the container runs on
CPU without it, which works but is slower):

```bash
sudo pacman -S nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 2. Start

```bash
local/start-ollama.sh            # terminal 1: Ollama on the Docker bridge, 16k context
local/up.sh -d                   # terminal 2: GPU image if Docker has the nvidia runtime
```

`local/up.sh` adds `docker-compose.local-gpu.yml` (Whisper on CUDA, NVENC
encoding) when the NVIDIA runtime is configured, otherwise it runs on CPU.
Measured on an RTX 4060 Ti: a 15-minute podcast → 6 captioned clips in ~3.5 min.

Open http://localhost:5175. The model is set in `docker-compose.local.yml`
(`LLM_MODEL`, default `qwen2.5:14b`). Pull another one with
`OLLAMA_HOST=172.17.0.1:11434 ollama pull <model>`.

Note: features that must look at video frames (automatic layout picker,
on-screen content detection, silent videos) need Gemini and are skipped
locally; clips fall back to face-tracking crops. Leave auto-posting off.

## 3. Clip for a campaign

```bash
python campaign.py new "Brand Podcast"        # creates campaigns/brand-podcast.json
# fill it in from the brief: length, @mentions, hashtags, disclosure, audio, banned words
python campaign.py check brand-podcast
python campaign.py env brand-podcast          # clip length to set in the dashboard (Advanced)
```

Run the job in the dashboard, then apply the brief to its clips:

```bash
python campaign.py prepare brand-podcast output/<job-id>
```

You get `campaign_out/brand-podcast/<job-id>/` with each passing clip, one
caption file per platform and `report.md` (what was rejected and why, plus the
checklist). Then track posting:

```bash
python campaign.py posted <id> tiktok @myaccount https://tiktok.com/...
python campaign.py submitted <id>
python campaign.py status                     # what to post / submit / record next
```

`campaign.py` needs only Python 3 and ffmpeg on the host; it also runs in the
container: `docker compose exec backend python campaign.py ...`.
