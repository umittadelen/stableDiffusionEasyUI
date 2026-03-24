"""
extension_loader.py — Extension / Mod system for stableDiffusionEasyUI
-----------------------------------------------------------------------
Drop a folder into ./extensions/ that contains an __init__.py and the
loader will import it on startup and call its setup() function.

Minimal extension layout
------------------------
extensions/
  my_extension/
    __init__.py        ← required — must define setup(app, gconfig, hooks, api)

Optional fields in __init__.py
-------------------------------
    EXTENSION_NAME        = "Human Readable Name"
    EXTENSION_VERSION     = "1.0.0"
    EXTENSION_DESCRIPTION = "What this extension does."

    def setup(app, gconfig, hooks, api):
        # app     — Flask application instance
        # gconfig — global config dict (read/write)
        # hooks   — HookRegistry  (register events / fire custom events)
        # api     — AppAPI object (see below)
        ...

AppAPI attributes
-----------------
    api.device           — torch.device (cuda / cpu)
    api.dtype            — torch dtype used for models (default: torch.float16)
    api.pipeline_cache   — live dict: {"pipe": <pipeline|None>, "key": <tuple|None>}

    api.load_pipeline(model_name, model_type, generation_type,
                      scheduler_name, clip_skip)   → pipe
    api.load_scheduler(pipe, scheduler_name)        → pipe
    api.generateImage(pipe, model, prompt, ...)     → image_path | False
    api.resize_image(image, width, height)          → PIL.Image
    api.image_to_base64(img)                        → "data:image/png;base64,..."
    api.latents_to_img(latents)                     → PIL.Image
    api.get_model_configs()                         → list[dict]
    api.controlNets                                 — class with get_canny_image /
                                                       get_depth_map / get_normal_map

Available hook events (fire in order)
--------------------------------------
    "on_app_start"     — fired once after all extensions load, before server starts
                         kwargs: app, gconfig

    "before_generate"  — fired before each batch generation starts
                         kwargs: prompts, image_count, width, height, model_name

    "after_generate"   — fired once per successfully saved image
                         kwargs: image_path, seed, prompt

    "before_load_pipeline"  — fired before a model pipeline is loaded
                              kwargs: model_name, model_type

    "after_load_pipeline"   — fired after a model pipeline finishes loading
                              kwargs: pipe, model_name, model_type
"""

import os
import sys
import importlib.util
import traceback


class HookRegistry:
    """Lightweight event/hook bus for inter-extension and core communication."""

    def __init__(self):
        self._hooks: dict[str, list] = {}

    def register(self, event: str, callback):
        """Register *callback* for *event*.  Multiple callbacks are supported."""
        self._hooks.setdefault(event, []).append(callback)

    def fire(self, event: str, *args, **kwargs) -> list:
        """Call every registered callback for *event*.

        Returns a list of return values.  Exceptions inside callbacks are
        caught and logged so one bad extension cannot break the whole app.
        """
        results = []
        for cb in self._hooks.get(event, []):
            try:
                results.append(cb(*args, **kwargs))
            except Exception:
                print(f"[extensions] Error in hook '{event}' (callback {cb!r}):\n"
                      f"{traceback.format_exc()}")
        return results


class AppAPI:
    """Container for all app internals that extensions can access.

    An instance is constructed in app.py and passed to every extension's
    setup() as the 4th argument ``api``.

    Attributes
    ----------
    device           torch.device — cuda or cpu
    dtype            torch dtype used for model loading (torch.float16)
    pipeline_cache   live dict shared with the core app:
                       {"pipe": <pipeline|None>, "key": <tuple|None>}

    Methods  (all are the real app functions — not copies)
    -------
    load_pipeline(model_name, model_type, generation_type,
                  scheduler_name, clip_skip)  -> pipe
    load_scheduler(pipe, scheduler_name)       -> pipe
    generateImage(pipe, model, prompt, ...)    -> image_path | False
    resize_image(image, width, height)         -> PIL.Image
    image_to_base64(img)                       -> str
    latents_to_img(latents)                    -> PIL.Image
    get_model_configs()                        -> list[dict]

    Attributes (classes / objects)
    ------
    controlNets      — class with static methods:
                         get_canny_image(image, w, h)
                         get_depth_map(image, w, h)
                         get_normal_map(image, w, h)
    """

    def __init__(self, *, device, dtype, pipeline_cache,
                 load_pipeline, load_scheduler, generateImage,
                 resize_image, image_to_base64, latents_to_img,
                 get_model_configs, controlNets):
        self.device          = device
        self.dtype           = dtype
        self.pipeline_cache  = pipeline_cache
        self.load_pipeline   = load_pipeline
        self.load_scheduler  = load_scheduler
        self.generateImage   = generateImage
        self.resize_image    = resize_image
        self.image_to_base64 = image_to_base64
        self.latents_to_img  = latents_to_img
        self.get_model_configs = get_model_configs
        self.controlNets     = controlNets


class ExtensionLoader:
    """Discovers, loads, and tracks all extensions in *extensions_dir*."""

    def __init__(self, extensions_dir: str = "./extensions"):
        self.extensions_dir = os.path.abspath(extensions_dir)
        self.hooks = HookRegistry()
        # name → metadata dict
        self._registry: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self, app, gconfig, api):
        """Scan *extensions_dir* and load every valid extension.

        Safe to call multiple times — already-loaded extensions are skipped.
        """
        if not os.path.isdir(self.extensions_dir):
            os.makedirs(self.extensions_dir, exist_ok=True)
            return

        disabled_file = os.path.join(self.extensions_dir, "disabled.json")
        disabled = set()
        if os.path.isfile(disabled_file):
            try:
                import json as _json
                with open(disabled_file, "r", encoding="utf-8") as _f:
                    disabled = set(_json.load(_f))
            except Exception:
                pass

        order_file = os.path.join(self.extensions_dir, "order.json")
        order = []
        if os.path.isfile(order_file):
            try:
                import json as _json
                with open(order_file, "r", encoding="utf-8") as _f:
                    order = _json.load(_f)
            except Exception:
                pass

        all_entries = sorted(os.listdir(self.extensions_dir))
        # extension_manager always first, then order.json sequence, then the rest
        ordered = ["extension_manager"] + [e for e in order if e != "extension_manager"]
        remaining = [e for e in all_entries if e not in ordered]
        entries = ordered + remaining

        # Enforce EXTENSION_REQUIRES_AFTER — fix order and persist back to order.json
        entries, order_changed = self._enforce_requires_after(entries)
        if order_changed:
            import json as _json
            # Save only real extension folders (not files like disabled.json)
            saveable = [
                e for e in entries
                if e != "extension_manager"
                and os.path.isdir(os.path.join(self.extensions_dir, e))
                and os.path.isfile(os.path.join(self.extensions_dir, e, "__init__.py"))
            ]
            with open(order_file, "w", encoding="utf-8") as _f:
                _json.dump(saveable, _f, indent=2)

        for entry in entries:
            ext_path = os.path.join(self.extensions_dir, entry)
            if not os.path.isdir(ext_path):
                continue
            if entry in self._registry:
                continue  # already loaded
            if entry in disabled:
                print(f"[extensions] Skipping '{entry}' (disabled)")
                continue
            init_file = os.path.join(ext_path, "__init__.py")
            if not os.path.isfile(init_file):
                continue

            # Check requires_after — skip if any dep wasn't loaded
            requires = self._read_requires_after(os.path.join(ext_path, "__init__.py"))
            unmet = [dep for dep in requires if not self._registry.get(dep, {}).get("loaded")]
            if unmet:
                print(f"[extensions] Force-disabled '{entry}': unmet requirements {unmet}")
                self._registry[entry] = {
                    "folder":         entry,
                    "name":           entry,
                    "loaded":         False,
                    "force_disabled": True,
                    "unmet_requires": unmet,
                    "requires_after": requires,
                    "error":          f"Unmet requirements: {', '.join(unmet)} must be loaded before this extension.",
                }
                continue

            self._load_one(entry, ext_path, init_file, app, gconfig, api)

    def get_info(self) -> list:
        """Return a list of dicts describing every discovered extension."""
        return list(self._registry.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_one(self, folder: str, ext_path: str, init_file: str, app, gconfig, api):
        # Make the extension folder importable so it can do relative imports
        if ext_path not in sys.path:
            sys.path.insert(0, ext_path)

        module_name = f"_ext_{folder}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, init_file,
                                                           submodule_search_locations=[ext_path])
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            name    = getattr(module, "EXTENSION_NAME",        folder)
            version = getattr(module, "EXTENSION_VERSION",     "0.0.0")
            desc    = getattr(module, "EXTENSION_DESCRIPTION", "")

            if hasattr(module, "setup"):
                import inspect
                sig = inspect.signature(module.setup)
                if len(sig.parameters) >= 4:
                    module.setup(app, gconfig, self.hooks, api)
                else:
                    # Backwards-compat: old 3-arg extensions still work
                    module.setup(app, gconfig, self.hooks)

            requires_after = getattr(module, "EXTENSION_REQUIRES_AFTER", [])

            self._registry[folder] = {
                "folder":        folder,
                "name":          name,
                "version":       version,
                "description":   desc,
                "loaded":        True,
                "requires_after": requires_after,
            }
            print(f"[extensions] Loaded '{name}' v{version} ({folder})")

        except Exception:
            err = traceback.format_exc()
            print(f"[extensions] Failed to load '{folder}':\n{err}")
            self._registry[folder] = {
                "folder":  folder,
                "name":    folder,
                "loaded":  False,
                "error":   err,
            }


    def _read_requires_after(self, init_file: str) -> list:
        """Parse EXTENSION_REQUIRES_AFTER from an __init__.py without importing it."""
        import ast
        try:
            tree = ast.parse(open(init_file, encoding="utf-8").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "EXTENSION_REQUIRES_AFTER":
                            if isinstance(node.value, ast.List):
                                return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
        except Exception:
            pass
        return []

    def _enforce_requires_after(self, entries: list) -> tuple:
        """Ensure every EXTENSION_REQUIRES_AFTER dependency appears before
        the extension that declares it. Returns (corrected_entries, was_changed)."""
        result = list(entries)
        any_changed = False
        changed = True
        while changed:
            changed = False
            for i, folder in enumerate(result):
                init = os.path.join(self.extensions_dir, folder, "__init__.py")
                for dep in self._read_requires_after(init):
                    if dep not in result:
                        continue
                    dep_idx = result.index(dep)
                    if dep_idx > i:
                        print(f"[extensions] '{folder}' requires_after '{dep}' — fixing load order")
                        result.pop(i)
                        result.insert(dep_idx + 1, folder)
                        any_changed = True
                        changed = True
                        break
                if changed:
                    break
        return result, any_changed


# Module-level singleton — imported by app.py
extension_loader = ExtensionLoader()
