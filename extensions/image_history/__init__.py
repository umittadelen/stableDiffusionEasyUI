EXTENSION_NAME        = "Image History"
EXTENSION_VERSION     = "1.0.0"
EXTENSION_DESCRIPTION = "Persists every generated image so they survive Clear."

import os, json, shutil, time

_EXT_DIR   = os.path.dirname(__file__)
_STATIC    = os.path.join(_EXT_DIR, "static")
_HIST_DIR  = os.path.join(_STATIC, "history")
_HIST_JSON = os.path.join(_EXT_DIR, "history.json")
_PAGE_SIZE = 40

os.makedirs(_HIST_DIR, exist_ok=True)

def _read():
    if os.path.isfile(_HIST_JSON):
        with open(_HIST_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _write(data):
    with open(_HIST_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def setup(app, gconfig, hooks, api):
    from flask import send_from_directory, jsonify, request

    def _on_after_generate(image_path, seed, prompt, **_):
        try:
            filename = os.path.basename(image_path)
            dest = os.path.join(_HIST_DIR, filename)
            if not os.path.isfile(dest):
                shutil.copy2(image_path, dest)

            try:
                from PIL import Image as _Img
                with _Img.open(image_path) as img:
                    info      = img.info
                    model     = info.get("Model", "")
                    scheduler = info.get("Scheduler", "")
                    width     = info.get("Width", "")
                    height    = info.get("Height", "")
            except OSError as e:
                print(f"[Image History] could not read metadata: {e}")
                model = scheduler = width = height = ""

            entry = {
                "filename":  filename,
                "prompt":    prompt,
                "seed":      seed,
                "model":     model,
                "scheduler": scheduler,
                "width":     width,
                "height":    height,
                "ts":        time.time(),
            }
            history = _read()
            history.insert(0, entry)
            _write(history)
        except Exception as e:
            print(f"[Image History] after_generate error: {e}")

    hooks.register("after_generate", _on_after_generate)

    @app.route("/image_history/")
    def ih_page():
        with open(os.path.join(_STATIC, "index.html"), encoding="utf-8") as f:
            return f.read()

    @app.route("/image_history/inject.js")
    def ih_inject_js():
        return send_from_directory(_STATIC, "inject.js", mimetype="application/javascript")

    @app.route("/image_history/img/<filename>")
    def ih_serve_img(filename):
        filename = os.path.basename(filename)
        return send_from_directory(_HIST_DIR, filename)

    @app.route("/image_history/list")
    def ih_list():
        q       = request.args.get("q", "").lower()
        page    = max(0, int(request.args.get("page", 0)))
        history = _read()
        if q:
            history = [e for e in history if q in e.get("prompt", "").lower()
                                           or q in e.get("model", "").lower()]
        total = len(history)
        chunk = history[page * _PAGE_SIZE : (page + 1) * _PAGE_SIZE]
        return jsonify(items=chunk, total=total, page=page, pageSize=_PAGE_SIZE)

    @app.route("/image_history/delete", methods=["POST"])
    def ih_delete():
        filename = os.path.basename(request.get_json().get("filename", ""))
        if not filename:
            return jsonify(error="no filename"), 400
        history = [e for e in _read() if e["filename"] != filename]
        _write(history)
        img_path = os.path.join(_HIST_DIR, filename)
        if os.path.isfile(img_path):
            os.remove(img_path)
        return jsonify(status="ok")

    @app.route("/image_history/clear", methods=["POST"])
    def ih_clear():
        _write([])
        for f in os.listdir(_HIST_DIR):
            try:
                os.remove(os.path.join(_HIST_DIR, f))
            except OSError as e:
                print(f"[Image History] could not delete {f}: {e}")
        return jsonify(status="ok")

    @app.after_request
    def ih_inject(response):
        if response.content_type.startswith("text/html"):
            body = response.get_data(as_text=True)
            tag  = '<script src="/image_history/inject.js" defer></script>'
            if "</body>" in body and tag not in body:
                body = body.replace("</body>", f"{tag}\n</body>")
                response.set_data(body)
        return response

    print("[Image History] Loaded — visit /image_history/")
