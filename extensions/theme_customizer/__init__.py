EXTENSION_NAME        = "Theme Customizer"
EXTENSION_VERSION     = "1.0.0"
EXTENSION_DESCRIPTION = "Custom background image and color overrides for the UI."

import os
from flask import send_from_directory, render_template_string

_EXT_STATIC = os.path.join(os.path.dirname(__file__), "static")

_INJECT_SNIPPET = """<script src="/theme_customizer/inject.js" defer></script>"""

def setup(app, gconfig, hooks, api):
    import json

    SETTINGS_FILE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "static", "json", "settings.json"
    )

    def _read_settings():
        if os.path.isfile(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _write_settings(data):
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @app.route("/theme_customizer/inject.js")
    def tc_inject_js():
        return send_from_directory(_EXT_STATIC, "inject.js",
                                   mimetype="application/javascript")

    @app.route("/theme_customizer/")
    def tc_page():
        with open(os.path.join(_EXT_STATIC, "index.html"), encoding="utf-8") as f:
            return f.read()

    @app.route("/theme_customizer/save", methods=["POST"])
    def tc_save():
        from flask import request, jsonify
        cfg = request.get_json()
        settings = _read_settings()
        settings["tc_theme"] = cfg
        _write_settings(settings)
        gconfig["tc_theme"] = cfg
        return jsonify(status="ok")

    @app.route("/theme_customizer/load", methods=["GET"])
    def tc_load():
        from flask import jsonify
        settings = _read_settings()
        return jsonify(settings.get("tc_theme", {}))

    # Inject the script tag into every HTML response
    @app.after_request
    def inject_script(response):
        if response.content_type.startswith("text/html"):
            body = response.get_data(as_text=True)
            if "</body>" in body and "/theme_customizer/inject.js" not in body:
                body = body.replace("</body>", f"{_INJECT_SNIPPET}\n</body>")
                response.set_data(body)
        return response

    print(f"[Theme Customizer] Loaded — visit /theme_customizer/ to customize")
