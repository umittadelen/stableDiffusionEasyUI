<div align="center">

# EasyUI V2.2.1
</div>
> A free, open-source local text-to-image generation UI — run advanced AI models on your own hardware with full parameter control.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/umittadelen/stableDiffusionEasyUI/blob/main/LICENSE.txt)
![Version](https://img.shields.io/badge/version-2.2.1-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

> [!NOTE]
> The `git clone` version is always more up-to-date than the official releases, but may be unstable and does not include embedded Python.

---

## What's New in V2.2.1

- **Theme Customizer extension** — New built-in extension to set a custom background image or solid color, with overlay tint color and opacity controls. Colors for all three UI tones are adjustable via color pickers. Settings persist in `settings.json` and are applied instantly on every page load.
- **Extension action bar buttons** — Extensions can now register a button in the main action bar via `window.registerExtensionButton(label, url, icon)`, making extension UIs directly accessible from the main page.

<details>
<summary>

## Previous Versions</summary>

**V2.2.0**
- **Async NSFW scoring** — Scores are computed in a background thread after generation and persisted in PNG metadata and server cache. The `/status` response includes `nsfw_score` per image so blurring applies immediately without blocking generation.
- **NoobAI model support** — NoobAI is now available as a selectable model type in the UI.
- **Improved exception handling** — Manual stops no longer produce misleading error messages. The generator checks `generation_stopped` before setting error status, preventing duplicate or incorrect updates.

</details>

---

## Features

| Feature | Description |
|---|---|
| Progress Tracking | Real-time generation progress |
| Seed Control | Reproducible or randomized outputs |
| CFG Scale | Control how closely the AI follows your prompt |
| Multiple Schedulers | Choose from various sampling schedulers (default: Euler A) |
| ControlNet | Canny/Depth fully supported; Normal Map for SD only |
| Img2Img & Txt2Img | Both generation modes supported |
| NSFW Blur | Auto-detects and blurs NSFW content in the gallery (hover to reveal) |
| Mobile Layout | Responsive UI that works on small screens |
| Extension System | Drop a folder into `extensions/` to add new features |
| Theme Customizer | Custom background image, solid color, overlay tint, and UI color controls |
| Drop to Fill | Drag & drop any EasyUI-generated image to restore its generation parameters |

---

## Requirements

- A CUDA-capable NVIDIA GPU
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Python 3.10+ (or use the bundled embedded Python)

---

## Installation

```bash
git clone https://github.com/umittadelen/stableDiffusionEasyUI.git
cd stableDiffusionEasyUI
```

**Option A — Embedded Python (recommended):**
Run `start.bat`. Dependencies are installed automatically on first launch.

**Option B — Your own Python:**
```bash
pip install -r requirements.txt
python run.py
```

The server starts at `http://localhost:8080` by default. The host and port can be changed from the Settings page.

---

## Usage

### 1. Download a Model

Open the **Model Editor** and enter a CivitAI model ID and version ID (found in the `AIR` field on the model page, e.g. `AIR: 123456 @ 654321`).

### 2. Configure Parameters

| Parameter | Description |
|---|---|
| Model Type | SD1.5 / SDXL / FLUX / NoobAI — auto-set when using Model Editor |
| Scheduler | Sampling scheduler (default: Euler A) |
| Prompt | Describe the image. Avoid very short prompts. Use `prompt1§prompt2` for multi-prompt. |
| Negative Prompt | Describe what to exclude from the image |
| Width / Height | Output image dimensions |
| CFG Scale | Prompt adherence strength |
| Sampling Steps | Number of denoising steps (30 recommended) |
| Seed | Set to `-1` for random variation |
| Batch Size | Number of images to generate |

> [!TIP]
> See the [Better Prompting Guide](https://umittadelen.github.io/better_prompting/) for tips on writing effective prompts.

### 3. Generate

Click **Generate Images** to start. Use **Stop Generation** to cancel at any time.

---

## Additional Tools

- **Clear Images** — Remove all images from the current gallery session
- **Preview ControlNet** — Preview the ControlNet preprocessor output before generating
- **Get Metadata** — Extract generation parameters from any EasyUI PNG
- **Model Editor** — Download and manage models via CivitAI IDs

---

## License

MIT License — see [LICENSE.txt](https://github.com/umittadelen/easyUI/blob/main/LICENSE.txt) for details.

---

Made by [umittadelen](https://umittadelen.net/)

---

### More About Me
[www.umittadelen.net](https://umittadelen.net)