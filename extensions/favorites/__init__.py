EXTENSION_NAME        = "Favorites"
EXTENSION_VERSION     = "1.0.0"
EXTENSION_DESCRIPTION = "Star any generated image to save it to favorites/ — survives Clear and Image History wipes."
EXTENSION_REQUIRES_AFTER = ["image_history"]

import os, json, shutil

_EXT_DIR  = os.path.dirname(__file__)
_STATIC   = os.path.join(_EXT_DIR, "static")
_FAV_DIR  = os.path.join(_EXT_DIR, "favorites")
_FAV_JSON = os.path.join(_EXT_DIR, "favorites.json")

os.makedirs(_FAV_DIR, exist_ok=True)

def _read():
    if os.path.isfile(_FAV_JSON):
        with open(_FAV_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _write(data):
    with open(_FAV_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def setup(app, gconfig, hooks, api):
    from flask import send_from_directory, jsonify, request
    from extension_loader import extension_loader

    @app.route("/favorites/")
    def fav_page():
        with open(os.path.join(_STATIC, "index.html"), encoding="utf-8") as f:
            return f.read()

    @app.route("/favorites/inject.js")
    def fav_inject_js():
        return send_from_directory(_STATIC, "inject.js", mimetype="application/javascript")

    @app.route("/favorites/img/<filename>")
    def fav_serve_img(filename):
        filename = os.path.basename(filename)
        return send_from_directory(_FAV_DIR, filename)

    @app.route("/favorites/list")
    def fav_list():
        return jsonify(_read())

    @app.route("/favorites/add", methods=["POST"])
    def fav_add():
        data     = request.get_json()
        src_url  = data.get("src", "")        # e.g. /generated/imageXXX.png  or  /image_history/img/imageXXX.png
        # Remove query parameters and fragments robustly
        src_url_clean = src_url.split("?")[0].split("#")[0]
        filename = os.path.basename(src_url_clean)
        if not filename:
            return jsonify(error="no filename"), 400

        # Try to resolve the source path more flexibly
        search_dirs = [
            os.path.join(os.path.dirname(_EXT_DIR), "image_history", "static", "history"),
            os.path.join(os.path.dirname(_EXT_DIR), "..", "generated"),
            _FAV_DIR,
        ]
        src_path = None
        for d in search_dirs:
            candidate = os.path.join(d, filename)
            if os.path.isfile(candidate):
                src_path = candidate
                break
        # Fallback: try to find the file by scanning directories if not found
        if not src_path:
            for d in search_dirs:
                for f in os.listdir(d):
                    if f == filename:
                        src_path = os.path.join(d, f)
                        break
                if src_path:
                    break
        if not src_path:
            return jsonify(error=f"source image not found: {filename}"), 404

        dest = os.path.join(_FAV_DIR, filename)
        if not os.path.isfile(dest):
            shutil.copy2(src_path, dest)

        favs = _read()
        if not any(e["filename"] == filename for e in favs):
            favs.insert(0, {"filename": filename, "src": src_url})
            _write(favs)

        return jsonify(status="ok", filename=filename)

    @app.route("/favorites/remove", methods=["POST"])
    def fav_remove():
        filename = os.path.basename(request.get_json().get("filename", ""))
        if not filename:
            return jsonify(error="no filename"), 400
        favs = [e for e in _read() if e["filename"] != filename]
        _write(favs)
        img_path = os.path.join(_FAV_DIR, filename)
        if os.path.isfile(img_path):
            os.remove(img_path)
        return jsonify(status="ok")

    @app.route("/favorites/check")
    def fav_check():
        filename = os.path.basename(request.args.get("filename", ""))
        favs = _read()
        return jsonify(favorited=any(e["filename"] == filename for e in favs))

    @app.route("/favorites/loaded_extensions")
    def fav_loaded_extensions():
        return jsonify([k for k, v in extension_loader._registry.items() if v.get("loaded")])

    @app.after_request
    def fav_inject(response):
        if response.content_type.startswith("text/html"):
            body = response.get_data(as_text=True)
            tag  = '<script src="/favorites/inject.js" defer></script>'
            if "</body>" in body and tag not in body:
                body = body.replace("</body>", f"{tag}\n</body>")
                response.set_data(body)
        return response

    print("[Favorites] Loaded — visit /favorites/")
