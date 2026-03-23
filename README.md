<div align="center">

# EasyUI V2.3.1
</div>
> A free, open-source local text-to-image generation UI — run advanced AI models on your own hardware with full parameter control.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/umittadelen/stableDiffusionEasyUI/blob/main/LICENSE.txt)
![Version](https://img.shields.io/badge/version-2.3.1-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

> [!NOTE]
> The `git clone` version is always more up-to-date than the official releases, but may be unstable and does not include embedded Python.

---

## What's New in V2.3.1

- **Extension Manager** — Built-in manager at `/extension_manager/` to enable/disable extensions, control load order with ▲/▼ buttons, and clone new extensions directly from a Git URL. The manager itself is always loaded first and cannot be disabled.
- **Image History extension** — Every generated image is automatically copied to a persistent history that survives **Clear**. Gallery UI with search, infinite scroll, and per-image delete. Clicking an image opens the existing metadata viewer.
- **Booru Tag Helper extension** — Live tag autocomplete in the prompt and negative prompt fields. Queries the Danbooru API as you type and shows suggestions sorted by post count. Keyboard navigable (↑/↓, Enter/Tab, Escape).
- **Glass blur control** — `--glass-blur` CSS variable is now adjustable via a slider in the Theme Customizer (0–40px).

<details>
<summary>

## Previous Versions</summary>

**V2.2.1**
- **Theme Customizer extension** — New built-in extension to set a custom background image or solid color, with overlay tint color and opacity controls. Colors for all three UI tones are adjustable via color pickers. Settings persist in `settings.json` and are applied instantly on every page load.
- **Extension action bar buttons** — Extensions can now register a button in the main action bar via `window.registerExtensionButton(label, url, icon)`, making extension UIs directly accessible from the main page.

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
| Extension Manager | Enable/disable extensions, set load order, clone from Git URL |
| Theme Customizer | Custom background image, solid color, overlay tint, UI colors, and glass blur |
| Image History | Persistent image gallery that survives Clear, with search and metadata viewer |
| Booru Tag Helper | Autocomplete booru-style tags in prompt fields, sorted by post count |
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

## Extensions

Extensions live in the `extensions/` folder. Each extension is a folder containing an `__init__.py` with a `setup(app, gconfig, hooks, api)` function.

### Built-in Extensions

| Extension | URL | Description |
|---|---|---|
| Extension Manager | `/extension_manager/` | Enable/disable extensions, set load order, clone from Git |
| Theme Customizer | `/theme_customizer/` | UI colors, background image, glass blur |
| Image History | `/image_history/` | Persistent image gallery with search |
| Booru Tag Helper | *(injects into prompt fields)* | Live tag autocomplete from Danbooru |

### Installing Extensions

Open the **Extension Manager** and paste a Git URL in the format `https://github.com/user/repo.git`. The extension will be cloned into `extensions/` and loaded on the next server restart.

> [!WARNING]
> Extensions run arbitrary Python code on your machine. Only install extensions from sources you trust.

### Extension Load Order

The Extension Manager always loads first. All other extensions load in the order shown in the manager UI — drag with ▲/▼ buttons to reorder. Order is saved to `extensions/order.json`.

### Disabling Extensions

Toggle any extension on or off in the Extension Manager. Disabled extensions are listed in `extensions/disabled.json` and skipped on startup. Changes take effect after restarting the server.

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
