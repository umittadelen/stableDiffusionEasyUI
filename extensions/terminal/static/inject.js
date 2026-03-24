(function () {
    // Only inject on the main index page
    if (!document.getElementById("images")) return;

    // ── Register action bar toggle button ──────────────────────────────────
    let _actionBtn = null;
    function _registerButton() {
        _actionBtn = window.registerExtensionButton(
            "Terminal",
            null,
            `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512" fill="currentColor"><path d="M9.4 86.6C-3.1 74.1-3.1 53.9 9.4 41.4s32.8-12.5 45.3 0l192 192c12.5 12.5 12.5 32.8 0 45.3l-192 192c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3L178.7 256 9.4 86.6zM256 416l288 0c17.7 0 32 14.3 32 32s-14.3 32-32 32l-288 0c-17.7 0-32-14.3-32-32s14.3-32 32-32z"/>`,
            () => toggleTerminal()
        );
    }
    const _regInterval = setInterval(() => {
        if (typeof window.registerExtensionButton === "function") {
            clearInterval(_regInterval);
            _registerButton();
        }
    }, 50);

    // ── ANSI → HTML ────────────────────────────────────────────────────────
    const ANSI_FG = {
        30:"#4c4c4c",31:"#f55",32:"#5f5",33:"#ff5",34:"#55f",
        35:"#f5f",36:"#5ff",37:"#ddd",90:"#888",91:"#f77",
        92:"#7f7",93:"#ff7",94:"#77f",95:"#f7f",96:"#7ff",97:"#fff"
    };
    const ANSI_BG = {
        40:"#111",41:"#500",42:"#050",43:"#550",44:"#005",
        45:"#505",46:"#055",47:"#aaa"
    };

    // Persistent style state across SSE chunks
    let _ansiState = { color:"", background:"", bold:false, italic:false, underline:false };

    function _stateToStyle(s) {
        let st = "";
        if (s.color)      st += `color:${s.color};`;
        if (s.background) st += `background:${s.background};`;
        if (s.bold)       st += "font-weight:bold;";
        if (s.italic)     st += "font-style:italic;";
        if (s.underline)  st += "text-decoration:underline;";
        return st;
    }

    // 256-color palette
    function _256color(n) {
        if (n < 16) {
            const c16 = ["#000","#800","#080","#880","#008","#808","#088","#ccc",
                         "#888","#f55","#5f5","#ff5","#55f","#f5f","#5ff","#fff"];
            return c16[n];
        }
        if (n < 232) {
            n -= 16;
            const b = n % 6, g = Math.floor(n / 6) % 6, r = Math.floor(n / 36);
            const v = x => x ? x * 40 + 55 : 0;
            return `rgb(${v(r)},${v(g)},${v(b)})`;
        }
        const gray = (n - 232) * 10 + 8;
        return `rgb(${gray},${gray},${gray})`;
    }

    function ansiToHtml(text) {
        let result = "";
        const parts = text.split(/(\x1b\[[0-9;]*[A-Za-z])/);
        for (let i = 0; i < parts.length; i++) {
            if (i % 2 === 0) {
                // Plain text segment
                if (!parts[i]) continue;
                const escaped = parts[i].replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
                const style = _stateToStyle(_ansiState);
                result += style ? `<span style="${style}">${escaped}</span>` : escaped;
            } else {
                // ESC sequence — only handle SGR (ends with 'm')
                const seq = parts[i];
                if (!seq.endsWith("m")) continue;
                const param = seq.slice(2, -1);
                const codes = param === "" ? [0] : param.split(";").map(Number);
                for (let j = 0; j < codes.length; j++) {
                    const c = codes[j];
                    if (c === 0)  { _ansiState = { color:"", background:"", bold:false, italic:false, underline:false }; }
                    else if (c === 1) _ansiState.bold = true;
                    else if (c === 3) _ansiState.italic = true;
                    else if (c === 4) _ansiState.underline = true;
                    else if (c === 22) _ansiState.bold = false;
                    else if (c === 23) _ansiState.italic = false;
                    else if (c === 24) _ansiState.underline = false;
                    else if (c === 39) _ansiState.color = "";
                    else if (c === 49) _ansiState.background = "";
                    else if ((c === 38 || c === 48) && codes[j+1] === 2) {
                        const [r,g,b] = [codes[j+2], codes[j+3], codes[j+4]];
                        const css = `rgb(${r},${g},${b})`;
                        if (c === 38) _ansiState.color = css; else _ansiState.background = css;
                        j += 4;
                    } else if ((c === 38 || c === 48) && codes[j+1] === 5) {
                        const idx = codes[j+2];
                        const css = _256color(idx);
                        if (c === 38) _ansiState.color = css; else _ansiState.background = css;
                        j += 2;
                    }
                    else if (ANSI_FG[c]) _ansiState.color = ANSI_FG[c];
                    else if (ANSI_BG[c]) _ansiState.background = ANSI_BG[c];
                }
            }
        }
        return result;
    }

    // ── Build panel ────────────────────────────────────────────────────────
    function buildPanel() {
        const style = document.createElement("style");
        style.textContent = `
            #term-panel {
                display: none;
                flex-direction: column;
                flex-shrink: 0;
                height: 220px;
                min-height: 80px;
                max-height: 60vh;
                resize: vertical;
                overflow: hidden;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 0.78em;
            }
            #term-panel.term-visible { display: flex; }
            #term-toolbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 4px 10px;
                border-bottom: 1px solid rgba(var(--tone3), 0.15);
                flex-shrink: 0;
                font-size: 0.85em;
                font-weight: 600;
                opacity: 0.7;
                font-family: var(--font-family);
                user-select: none;
            }
            #term-toolbar-btns { display: flex; gap: 6px; }
            #term-toolbar-btns button {
                padding: 2px 8px;
                font-size: 0.8em;
                text-transform: none;
                letter-spacing: 0;
                opacity: 0.7;
            }
            #term-output {
                flex: 1;
                overflow-y: auto;
                overflow-x: auto;
                padding: 8px 12px;
                line-height: 1.45;
                color: #ccc;
                word-break: break-all;
            }
            #term-output::-webkit-scrollbar { width: 5px; height: 5px; }
            #term-output::-webkit-scrollbar-thumb {
                background: rgba(var(--tone3), 0.2);
                border-radius: 3px;
            }
        `;
        document.head.appendChild(style);

        const panel = document.createElement("div");
        panel.id = "term-panel";
        panel.className = "glass-panel";
        panel.innerHTML = `
            <div id="term-toolbar">
                <span>Terminal</span>
                <div id="term-toolbar-btns">
                    <button id="term-autoscroll-btn">⬇ Auto-scroll</button>
                    <button id="term-clear-btn">✕ Clear</button>
                </div>
            </div>
            <div id="term-output"></div>
        `;

        const right   = document.querySelector(".panel-right");
        const gallery = document.querySelector(".image-gallery");
        if (right && gallery) right.insertBefore(panel, gallery);
        else { console.warn("[Terminal] fallback: appending to body"); document.body.appendChild(panel); }
        console.log("[Terminal] panel inserted", panel.parentElement);
        return panel;
    }

    // ── State ──────────────────────────────────────────────────────────────
    let autoScroll  = true;
    let visible     = false;
    let _es         = null;
    let _curLine    = null;   // current <div> being written to (tqdm overwrite target)

    function _getOut() { return document.getElementById("term-output"); }

    function _newLine() {
        const out = _getOut();
        if (!out) return null;
        _curLine = document.createElement("div");
        out.appendChild(_curLine);
        return _curLine;
    }

    // Process raw text (with \r and \n) into DOM lines
    function appendText(raw) {
        const out = _getOut();
        if (!out) return;
        if (!_curLine || !out.contains(_curLine)) _newLine();

        // Split on \r and \n while keeping the delimiters
        const tokens = raw.split(/(\r\n|\r|\n)/);
        for (const tok of tokens) {
            if (tok === "\r\n" || tok === "\n") {
                _newLine();
            } else if (tok === "\r") {
                // Carriage return: overwrite current line
                _curLine.innerHTML = "";
            } else if (tok) {
                // Use a document fragment to avoid repeated innerHTML +=
                const html = ansiToHtml(tok);
                if (html) {
                    const frag = document.createDocumentFragment();
                    const temp = document.createElement("div");
                    temp.innerHTML = html;
                    while (temp.firstChild) {
                        frag.appendChild(temp.firstChild);
                    }
                    _curLine.appendChild(frag);
                }
            }
        }
        if (autoScroll) out.scrollTop = out.scrollHeight;
    }

    function startSSE() {
        if (_es) return;
        _es = new EventSource("/terminal/stream");
        _es.onmessage = (e) => {
            appendText(e.data + "\n");
        };
        _es.onerror = () => {
            _es.close();
            _es = null;
            setTimeout(startSSE, 3000);
        };
    }

    function toggleTerminal() {
        visible = !visible;
        const panel = document.getElementById("term-panel");
        if (!panel) { console.warn("[Terminal] panel not found"); return; }
        panel.classList.toggle("term-visible", visible);
        panel.style.display = visible ? "flex" : "none";
        if (_actionBtn) _actionBtn.style.background = visible ? "rgba(var(--tone3), 0.35)" : "";
        if (visible) startSSE();
    }

    // ── Wire up ────────────────────────────────────────────────────────────
    function init() {
        buildPanel();

        document.getElementById("term-autoscroll-btn").addEventListener("click", () => {
            autoScroll = !autoScroll;
            document.getElementById("term-autoscroll-btn").style.opacity = autoScroll ? "1" : "0.4";
        });

        document.getElementById("term-clear-btn").addEventListener("click", () => {
            const out = document.getElementById("term-output");
            if (out) out.innerHTML = "";
            _curLine = null;
            _ansiState = { color:"", background:"", bold:false, italic:false, underline:false };
        });
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})();
