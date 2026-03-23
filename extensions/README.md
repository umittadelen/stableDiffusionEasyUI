# Extensions

Drop extension folders here. Each one is auto-loaded when the app starts.

## Folder structure

```
extensions/
  my_extension/          ← folder name = extension ID
    __init__.py          ← required entry point
    any_other_file.py    ← optional supporting files
    static/              ← optional static assets
```

## Minimal `__init__.py`

```python
EXTENSION_NAME        = "My Extension"
EXTENSION_VERSION     = "1.0.0"
EXTENSION_DESCRIPTION = "Does something cool."

def setup(app, gconfig, hooks, api):
    # app     → Flask instance, use to add routes
    # gconfig → global config dict (read/write)
    # hooks   → HookRegistry (see events below)
    # api     → AppAPI (see below)
    pass
```

## The `api` object

`api` gives you direct access to app internals:

| Attribute / Method | Type | Description |
|---|---|---|
| `api.device` | `torch.device` | Current compute device (`cuda` or `cpu`) |
| `api.dtype` | `torch.dtype` | Model dtype (`torch.float16`) |
| `api.pipeline_cache` | `dict` | Live `{"pipe": …, "key": …}` — read or swap the loaded pipeline |
| `api.load_pipeline(model_name, model_type, generation_type, scheduler_name, clip_skip)` | fn | Load / build a diffusers pipeline |
| `api.load_scheduler(pipe, scheduler_name)` | fn | Apply a scheduler to a pipeline |
| `api.generateImage(pipe, model, prompt, …)` | fn | Full image-generation call |
| `api.resize_image(image, width, height)` | fn | BICUBIC PIL resize |
| `api.image_to_base64(img)` | fn | Convert PIL image to base64 data-URI |
| `api.latents_to_img(latents)` | fn | Preview latents as a PIL image |
| `api.get_model_configs()` | fn | Scan `./models/` and return config list |
| `api.controlNets` | class | Static methods: `get_canny_image`, `get_depth_map`, `get_normal_map` |

## Adding Flask routes

```python
def setup(app, gconfig, hooks):
    from flask import jsonify

    @app.route("/my_extension/api")
    def my_api():
        return jsonify(hello="world")
```

## Hook events

Register a listener with `hooks.register(event, callback)`.

| Event | When fired | Keyword args passed |
|---|---|---|
| `on_app_start` | Once, after all extensions load, before the server starts | `app`, `gconfig` |
| `before_generate` | Before each image batch starts | `prompts`, `image_count`, `width`, `height`, `model_name` |
| `after_generate` | After each image is saved | `image_path`, `seed`, `prompt` |
| `before_load_pipeline` | Before a model pipeline is loaded | `model_name`, `model_type` |
| `after_load_pipeline` | After a model pipeline finishes loading | `pipe`, `model_name`, `model_type` |

```python
def setup(app, gconfig, hooks):
    hooks.register("after_generate", on_image_done)

def on_image_done(image_path, seed, prompt, **kwargs):
    print(f"New image: {image_path}")
```

> **Tip:** always add `**kwargs` to callbacks so your extension keeps working when new arguments are added later.

## Listing loaded extensions

`GET /extensions` returns a JSON array of all discovered extensions and their status.

## Adding a button to the main action bar

The main page exposes a global `registerExtensionButton(label, url, svgString)` function.
Call it from a script you inject via `app.after_request`:

```javascript
// inside your injected <script> or .js file
window.registerExtensionButton(
    "My Tool",           // button label
    "/my_extension/",    // URL opened in a new tab
    `<svg .../>`,        // optional SVG icon string (or empty string)
);
```

Buttons appear in a dedicated extension slot in the action bar, separated from the built-in buttons by a divider. The function is a no-op on pages that don't have the slot (settings, metadata, etc.).

## Example extension

See `extensions/example_extension/` for a fully commented reference implementation.
