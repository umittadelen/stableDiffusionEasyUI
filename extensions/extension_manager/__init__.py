EXTENSION_NAME        = "Extension Manager"
EXTENSION_VERSION     = "1.0.0"
EXTENSION_DESCRIPTION = "Built-in manager to enable/disable extensions and clone new ones."

import os, json, subprocess

_EXT_DIR      = os.path.dirname(__file__)
_STATIC       = os.path.join(_EXT_DIR, "static")
_EXTENSIONS   = os.path.dirname(_EXT_DIR)
_DISABLED_FILE = os.path.join(_EXTENSIONS, "disabled.json")
_PROTECTED    = {"extension_manager"}

def _read_disabled():
    if os.path.isfile(_DISABLED_FILE):
        with open(_DISABLED_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def _write_disabled(disabled: set):
    with open(_DISABLED_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(disabled), f, indent=2)

def setup(app, gconfig, hooks, api):
    from flask import jsonify, request, send_from_directory
    from extension_loader import extension_loader

    @app.route("/extension_manager/")
    def em_page():
        with open(os.path.join(_STATIC, "index.html"), encoding="utf-8") as f:
            return f.read()

    @app.route("/extension_manager/inject.js")
    def em_inject_js():
        return send_from_directory(_STATIC, "inject.js", mimetype="application/javascript")

    @app.route("/extension_manager/list")
    def em_list():
        disabled = _read_disabled()

        order_file = os.path.join(_EXTENSIONS, "order.json")
        order = []
        if os.path.isfile(order_file):
            try:
                with open(order_file, "r", encoding="utf-8") as f:
                    order = json.load(f)
            except (OSError, ValueError) as e:
                print(f"[Extension Manager] Could not read order.json: {e}")

        # Scan all folders in extensions/ that have __init__.py
        all_folders = sorted(
            e for e in os.listdir(_EXTENSIONS)
            if e not in _PROTECTED
            and os.path.isdir(os.path.join(_EXTENSIONS, e))
            and os.path.isfile(os.path.join(_EXTENSIONS, e, "__init__.py"))
        )
        ordered = [e for e in order if e in all_folders]
        remaining = [e for e in all_folders if e not in ordered]
        folders = ordered + remaining

        items = []
        for folder in folders:
            reg = extension_loader._registry.get(folder, {})
            items.append({
                "folder":         folder,
                "name":           reg.get("name", folder),
                "version":        reg.get("version", ""),
                "description":    reg.get("description", ""),
                "loaded":         reg.get("loaded", False),
                "error":          reg.get("error", ""),
                "enabled":        folder not in disabled,
                "requires_after": reg.get("requires_after", []),
                "force_disabled": reg.get("force_disabled", False),
                "unmet_requires": reg.get("unmet_requires", []),
            })
        return jsonify(items)

    @app.route("/extension_manager/toggle", methods=["POST"])
    def em_toggle():
        folder = request.get_json().get("folder", "")
        if not folder or folder in _PROTECTED:
            return jsonify(error="protected or invalid"), 400
        disabled = _read_disabled()
        if folder in disabled:
            disabled.discard(folder)
        else:
            disabled.add(folder)
        _write_disabled(disabled)
        return jsonify(enabled=folder not in disabled)

    _ORDER_FILE = os.path.join(_EXTENSIONS, "order.json")

    @app.route("/extension_manager/reorder", methods=["POST"])
    def em_reorder():
        order = request.get_json().get("order", [])
        order = [f for f in order if f not in _PROTECTED]
        with open(_ORDER_FILE, "w", encoding="utf-8") as f:
            json.dump(order, f, indent=2)
        return jsonify(status="ok")

    @app.route("/extension_manager/clone", methods=["POST"])
    def em_clone():
        import re
        url = request.get_json().get("url", "").strip()
        if not url:
            return jsonify(error="no url"), 400
        if not re.fullmatch(r"https://[\w.\-]+/[\w.\-/]+\.git", url):
            return jsonify(error="URL must be https://<host>/<path>.git"), 400
        folder = url.rstrip("/").split("/")[-1].removesuffix(".git")
        if not re.fullmatch(r"[\w\-]+", folder):
            return jsonify(error="invalid repository name"), 400
        dest = os.path.join(_EXTENSIONS, folder)
        if os.path.exists(dest):
            return jsonify(error=f"'{folder}' already exists"), 400
        import shutil as _shutil
        git_bin = _shutil.which("git")
        if not git_bin:
            return jsonify(error="git not found in PATH"), 500
        try:
            result = subprocess.run(
                [git_bin, "clone", url, dest],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                return jsonify(error=result.stderr.strip()), 500
            return jsonify(status="cloned", folder=folder)
        except subprocess.TimeoutExpired:
            return jsonify(error="git clone timed out"), 500

    @app.after_request
    def em_inject(response):
        if response.content_type.startswith("text/html"):
            body = response.get_data(as_text=True)
            tag = '<script src="/extension_manager/inject.js" defer></script>'
            if "</body>" in body and tag not in body:
                body = body.replace("</body>", f"{tag}\n</body>")
                response.set_data(body)
        return response

    print("[Extension Manager] Loaded — visit /extension_manager/")
