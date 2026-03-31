"""
example_extension — A minimal reference extension for stableDiffusionEasyUI
============================================================================

This extension demonstrates:
  • Adding a new Flask API route  ( GET /example_extension/hello )
  • Inspecting device / dtype via the api object
  • Accessing the live pipeline cache
  • Listening to every available hook event
  • Installing extension-specific dependencies via requirements.txt

You can copy this folder, rename it, and start building your own feature.
"""

EXTENSION_NAME        = "Example Extension"
EXTENSION_VERSION     = "1.0.0"
EXTENSION_DESCRIPTION = "A minimal reference extension demonstrating all api / hook features."

from chromaconsole import Color, Style

def setup(app, gconfig, hooks, api):
    """Called once by the extension loader at startup.

    Parameters
    ----------
    app     : Flask application instance — use it to register routes.
    gconfig : The global config dict shared by the whole application.
    hooks   : HookRegistry — call hooks.register(event, callback) to listen
              or hooks.fire(event, ...) to emit your own events.
    api     : AppAPI — access to device, dtype, pipeline_cache, and all
              core functions (load_pipeline, generateImage, resize_image, …)
    """

    # ------------------------------------------------------------------
    # 1. Register a new API route — shows device / dtype / pipeline info
    # ------------------------------------------------------------------
    from flask import jsonify

    @app.route("/example_extension/hello")
    def example_hello():
        pipe_loaded = api.pipeline_cache.get("pipe") is not None
        pipe_key    = str(api.pipeline_cache.get("key"))
        return jsonify(
            extension=EXTENSION_NAME,
            version=EXTENSION_VERSION,
            message="Hello from the example extension!",
            device=str(api.device),
            dtype=str(api.dtype),
            pipeline_loaded=pipe_loaded,
            pipeline_key=pipe_key,
        )

    # ------------------------------------------------------------------
    # 2. Hook into every available event
    # ------------------------------------------------------------------
    hooks.register("on_app_start",          _on_app_start)
    hooks.register("before_generate",       _on_before_generate)
    hooks.register("after_generate",        _on_after_generate)
    hooks.register("before_load_pipeline",  _on_before_load_pipeline)
    hooks.register("after_load_pipeline",   _on_after_load_pipeline)

    # Store the api reference so hook callbacks can use it
    global _api
    _api = api

    print(f"{Color.text('#FF00EA')}[{EXTENSION_NAME}]{Color.default_text()} Setup complete on {api.device} — visit /example_extension/hello")


_api = None   # set during setup()


# ------------------------------------------------------------------
# Hook callbacks
# ------------------------------------------------------------------

def _on_app_start(app, gconfig, **kwargs):
    print(f"{Color.text('#FF00EA')}[{EXTENSION_NAME}]{Color.default_text()} {Color.text('#00FF00')}App started.{Color.default_text()} "
          f"{Color.text('#00E1FF')}Host:{Color.default_text()} {gconfig.get('host')}:{gconfig.get('port')}  "
          f"{Color.text('#6F00FF')}Device:{Color.default_text()} {_api.device}  {Color.text('#008CFF')}Dtype:{Color.default_text()} {_api.dtype}")


def _on_before_generate(prompts, image_count, width, height, model_name, **kwargs):
    print(f"{Color.text('#FF00EA')}[{EXTENSION_NAME}]{Color.default_text()} {Color.text('#00FF00')}Generation starting.{Color.default_text()} "
          f"{image_count} image(s) at {Style.italic()}{width}x{height}{Style.not_italic()} with {Style.bold()}'{model_name}'{Style.reset()}")


def _on_after_generate(image_path, seed, prompt, **kwargs):
    print(f"{Color.text('#FF00EA')}[{EXTENSION_NAME}]{Color.default_text()} {Color.text('#0400FF')}Image saved.{Color.default_text()} → {image_path}  {Color.text('#00E1FF')}(seed={seed}){Color.default_text()}")


def _on_before_load_pipeline(model_name, model_type, **kwargs):
    print(f"{Color.text('#FF00EA')}[{EXTENSION_NAME}]{Color.default_text()} {Color.text('#FF9100')}Loading pipeline.{Color.default_text()} → {model_name!r}  type={model_type}")


def _on_after_load_pipeline(pipe, model_name, model_type, **kwargs):
    print(f"{Color.text('#FF00EA')}[{EXTENSION_NAME}]{Color.default_text()} {Color.text('#BBFF00')}Pipeline ready.{Color.default_text()} → {type(pipe).__name__}  model={model_name!r}")
    # Example: access any scheduler config from the loaded pipeline
    if hasattr(pipe, "scheduler"):
        print(f"{Color.text('#FF00EA')}[{EXTENSION_NAME}]{Color.default_text()} {Color.text('#8C00FF')}Scheduler: {Color.default_text()}{type(pipe.scheduler).__name__}")