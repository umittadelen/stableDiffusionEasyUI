EXTENSION_NAME        = "Terminal"
EXTENSION_VERSION     = "1.5.0"
EXTENSION_DESCRIPTION = "Live server output (stdout + stderr) shown in the main UI. Read-only."

import sys, os, threading, collections

_EXT_DIR   = os.path.dirname(__file__)
_STATIC    = os.path.join(_EXT_DIR, "static")
_MAX_LINES = 500
_lock      = threading.Lock()
_lines     = collections.deque(maxlen=_MAX_LINES)   # each entry is one text chunk
_listeners = []   # list of threading.Event, one per SSE client

# ── Stream interceptor ──────────────────────────────────────────────────────

class _Tee:
    def __init__(self, original):
        self._orig = original

    def write(self, text):
        self._orig.write(text)
        if text:
            with _lock:
                _lines.append(text)
                for ev in _listeners:
                    ev.set()

    def flush(self):
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


sys.stdout = _Tee(sys.stdout)
sys.stderr = _Tee(sys.stderr)

# ── SSE helpers ─────────────────────────────────────────────────────────────

import re

def _encode_sse(text):
    """
    Split by both newline (\n) and carriage return (\r).
    Each chunk becomes a separate SSE 'data' event.
    """
    if not text:
        return ""
    
    # Split by \n or \r (tqdm uses \r)
    # We use a regex to keep the delimiters if needed, 
    # but here we just want the chunks of text.
    lines = re.split(r'[\n\r]', text)
    
    # Filter out empty strings to avoid spamming empty events
    output = []
    for l in lines:
        if l.strip():
            output.append(f"data:{l}")
            
    if not output:
        return ""
        
    return "\n".join(output) + "\n\n"

# ── Extension setup ─────────────────────────────────────────────────────────

def setup(app, gconfig, hooks, api):
    from flask import Response, send_from_directory

    @app.route("/terminal/inject.js")
    def term_inject_js():
        return send_from_directory(_STATIC, "inject.js", mimetype="application/javascript")

    @app.route("/terminal/stream")
    def term_stream():
        ev = threading.Event()
        with _lock:
            _listeners.append(ev)
            # Snapshot current buffer and remember its length as our cursor
            snapshot = list(_lines)
            cursor   = [len(_lines)]   # mutable via list so nested fn can write it

        def generate():
            try:
                # Send full buffer immediately on connect
                if snapshot:
                    msg = _encode_sse("".join(snapshot))
                    if msg:
                        yield msg

                while True:
                    ev.wait(timeout=30)
                    ev.clear()
                    with _lock:
                        current = list(_lines)
                    # Only send chunks added since our cursor
                    new_chunks = current[cursor[0]:]
                    cursor[0] = len(current)
                    if new_chunks:
                        msg = _encode_sse("".join(new_chunks))
                        if msg:
                            yield msg
            finally:
                with _lock:
                    try:
                        _listeners.remove(ev)
                    except ValueError:
                        pass

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control":    "no-cache",
                "X-Accel-Buffering":"no",
            }
        )

    @app.after_request
    def term_inject(response):
        if response.content_type.startswith("text/html"):
            body = response.get_data(as_text=True)
            tag  = '<script src="/terminal/inject.js" defer></script>'
            if "</body>" in body and tag not in body:
                body = body.replace("</body>", f"{tag}\n</body>")
                response.set_data(body)
        return response

    print("[Terminal] Loaded — live output available in the main UI")
