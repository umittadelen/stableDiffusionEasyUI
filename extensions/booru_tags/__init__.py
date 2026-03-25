EXTENSION_NAME        = "Booru Tag Helper"
EXTENSION_VERSION     = "1.1.0"
EXTENSION_DESCRIPTION = "Autocomplete booru-style tags in the prompt fields, sorted by post count."

import os

_STATIC = os.path.join(os.path.dirname(__file__), "static")

def setup(app, gconfig, hooks, api):
    from flask import jsonify, request, send_from_directory
    import requests as _req

    @app.route("/booru_tags/inject.js")
    def bt_inject_js():
        return send_from_directory(_STATIC, "inject.js", mimetype="application/javascript")

    @app.route("/booru_tags/search")
    def bt_search():
        q = request.args.get("q", "").strip().lower()
        if len(q) < 2:
            return jsonify([])
        try:
            resp = _req.get(
                "https://danbooru.donmai.us/tags.json",
                params={"search[name_matches]": f"{q}*", "search[order]": "count", "limit": 20},
                timeout=5,
                headers={"User-Agent": "stableDiffusionEasyUI/1.0"}
            )
            tags = resp.json()
            return jsonify([
                {"name": t["name"].replace("_", " "), "count": t["post_count"]}
                for t in tags if t.get("post_count", 0) > 0
            ])
        except (_req.RequestException, ValueError) as e:
            print(f"[Booru Tag Helper] search error: {e}")
            return jsonify([])

    @app.after_request
    def bt_inject(response):
        if response.content_type.startswith("text/html"):
            body = response.get_data(as_text=True)
            tag = '<script src="/booru_tags/inject.js" defer></script>'
            if "</body>" in body and tag not in body:
                body = body.replace("</body>", f"{tag}\n</body>")
                response.set_data(body)
        return response

    print("[Booru Tag Helper] Loaded")
