(function () {
    if (!document.getElementById("images")) return;

    // ── Configuration & State ──────────────────────────────────────────────
    const MAX_LINES = 500;
    let autoScroll = true;
    let visible = false;
    let _es = null;
    let _curLine = null; 
    let _ansiState = { color: "", background: "", bold: false, italic: false };

    // ── Google Font Injection ──────────────────────────────────────────────
    function injectFonts() {
        if (document.getElementById("term-inconsolata")) return;
        
        const pre1 = document.createElement("link");
        pre1.rel = "preconnect"; pre1.href = "https://fonts.googleapis.com";
        document.head.appendChild(pre1);

        const pre2 = document.createElement("link");
        pre2.rel = "preconnect"; pre2.href = "https://fonts.gstatic.com";
        pre2.crossOrigin = "anonymous";
        document.head.appendChild(pre2);

        const fontLink = document.createElement("link");
        fontLink.id = "term-inconsolata";
        fontLink.rel = "stylesheet";
        fontLink.href = "https://fonts.googleapis.com/css2?family=Inconsolata:wght@200..900&display=swap";
        document.head.appendChild(fontLink);
    }

    // ── ANSI Helper ────────────────────────────────────────────────────────
    const ANSI_FG = { 30: "#4c4c4c", 31: "#f55", 32: "#5f5", 33: "#ff5", 34: "#55f", 35: "#f5f", 36: "#5ff", 37: "#ddd", 90: "#888", 91: "#f77", 92: "#7f7", 93: "#ff7", 94: "#77f", 95: "#f7f", 96: "#7ff", 97: "#fff" };

    function _256color(n) {
        if (n < 16) return ANSI_FG[n + 30] || ANSI_FG[n + 90 - 8] || "#ccc";
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
        const parts = text.split(/(\x1b\[[0-9;]*[mK])/);
        
        for (let i = 0; i < parts.length; i++) {
            if (i % 2 === 0) {
                if (!parts[i]) continue;
                const escaped = parts[i].replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                
                let style = "";
                if (_ansiState.color)      style += `color:${_ansiState.color};`;
                if (_ansiState.background) style += `background-color:${_ansiState.background};`;
                if (_ansiState.bold)       style += `font-weight: 900 !important;`; // Uses Inconsolata 900
                if (_ansiState.italic)     style += `font-style: italic !important;`;
                
                result += style ? `<span style="${style}">${escaped}</span>` : escaped;
            } else {
                const seq = parts[i];
                if (!seq.endsWith("m")) continue;
                const params = seq.slice(2, -1).split(";").map(p => p === "" ? 0 : Number(p));
                
                for (let j = 0; j < params.length; j++) {
                    const c = params[j];
                    if (c === 0) _ansiState = { color: "", background: "", bold: false, italic: false };
                    else if (c === 1)  _ansiState.bold = true;
                    else if (c === 22) _ansiState.bold = false;
                    else if (c === 3)  _ansiState.italic = true;
                    else if (c === 23) _ansiState.italic = false;
                    else if (c === 38 && params[j + 1] === 2) {
                        _ansiState.color = `rgb(${params[j+2]},${params[j+3]},${params[j+4]})`;
                        j += 4; 
                    }
                    else if (c === 48 && params[j + 1] === 2) {
                        _ansiState.background = `rgb(${params[j+2]},${params[j+3]},${params[j+4]})`;
                        j += 4; 
                    }
                    else if ((c === 38 || c === 48) && params[j + 1] === 5) {
                        const color = _256color(params[j + 2]);
                        if (c === 38) _ansiState.color = color; else _ansiState.background = color;
                        j += 2;
                    }
                    else if (c >= 30 && c <= 37) _ansiState.color = ANSI_FG[c];
                    else if (c >= 40 && c <= 47) _ansiState.background = ANSI_FG[c - 10];
                    else if (c >= 90 && c <= 97) _ansiState.color = ANSI_FG[c];
                    else if (c === 39) _ansiState.color = "";
                    else if (c === 49) _ansiState.background = "";
                }
            }
        }
        return result;
    }

    // ── UI Logic ───────────────────────────────────────────────────────────
    function buildPanel() {
        if (document.getElementById("term-panel")) return;
        injectFonts(); // Inject Inconsolata
        
        const style = document.createElement("style");
        style.textContent = `
            #term-panel { 
                display: none; flex-direction: column; height: 250px; 
                font-family: 'Inconsolata', monospace; 
                font-size: 15px; /* Inconsolata runs slightly smaller than Consolas */
                overflow: hidden; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); 
            }
            #term-panel.term-visible { display: flex; }
            #term-toolbar { display: flex; justify-content: space-between; padding: 5px 12px; background: rgba(0,0,0,0.3); border-bottom: 1px solid rgba(255,255,255,0.1); align-items: center; }
            #term-output { flex: 1; overflow-y: auto; padding: 10px; background: #0c0c0c; color: #ccc; white-space: pre-wrap; word-break: break-all; line-height: 1.4; font-weight: 300; }
            #term-output div { min-height: 1.2em; }
            
            #term-progress-container { display: none; padding: 12px; background: #151515; border-bottom: 1px solid #222; flex-shrink: 0; }
            #term-progress-bar-bg { width: 100%; height: 8px; background: #333; border-radius: 4px; overflow: hidden; border: 1px solid #444; }
            #term-progress-fill { width: 0%; height: 100%; background: #10b981; transition: width 0.2s ease; box-shadow: 0 0 8px #10b981; }
            #term-progress-text { display: block; margin-bottom: 6px; color: #10b981; font-weight: 900; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
            
            .term-btn { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #eee; padding: 2px 8px; cursor: pointer; font-size: 10px; border-radius: 4px; font-family: sans-serif; }
            .term-btn:hover { background: rgba(255,255,255,0.15); }
        `;
        document.head.appendChild(style);

        const panel = document.createElement("div");
        panel.id = "term-panel";
        panel.className = "glass-panel";
        panel.innerHTML = `
            <div id="term-toolbar">
                <span>TERMINAL OUTPUT</span>
                <div style="display:flex; gap:5px;">
                    <button class="term-btn" id="term-auto-btn">Auto-scroll: ON</button>
                    <button class="term-btn" id="term-clear-btn">Clear</button>
                </div>
            </div>
            <div id="term-progress-container">
                <span id="term-progress-text">In Progress... 0%</span>
                <div id="term-progress-bar-bg"><div id="term-progress-fill"></div></div>
            </div>
            <div id="term-output"></div>
        `;
        const actionBar = document.querySelector(".action-bar-panel") || document.querySelector(".action-bar");
        const rightPanel = document.querySelector(".panel-right");

        if (actionBar) {
            actionBar.after(panel);
        } else if (rightPanel) {
            rightPanel.prepend(panel);
        } else {
            document.body.appendChild(panel);
        }

        document.getElementById("term-clear-btn").onclick = () => { document.getElementById("term-output").innerHTML = ""; _curLine = null; };
        document.getElementById("term-auto-btn").onclick = (e) => {
            autoScroll = !autoScroll;
            e.target.innerText = `Auto-scroll: ${autoScroll ? "ON" : "OFF"}`;
        };
    }

    // ── Action Registration ────────────────────────────────────────────────
    let _actionBtn = null;
    function _registerButton() {
        _actionBtn = window.registerExtensionButton(
            "Terminal",
            null,
            `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512" fill="currentColor"><path d="M9.4 86.6C-3.1 74.1-3.1 53.9 9.4 41.4s32.8-12.5 45.3 0l192 192c12.5 12.5 12.5 32.8 0 45.3l-192 192c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3L178.7 256 9.4 86.6zM256 416l288 0c17.7 0 32 14.3 32 32s-14.3 32-32 32l-288 0c-17.7 0-32-14.3-32-32s14.3-32 32-32z"/></svg>`,
            () => toggleTerminal()
        );
    }
    const _regInterval = setInterval(() => {
        if (typeof window.registerExtensionButton === "function") {
            clearInterval(_regInterval);
            _registerButton();
        }
    }, 50);

    function updateVisualProgress(text) {
        const clean = text.replace(/\x1B\[[0-9;]*[A-Za-z]/g, "");
        const match = clean.match(/(\d+)%\|/);
        if (match) {
            const percent = parseInt(match[1]);
            const container = document.getElementById("term-progress-container");
            const fill = document.getElementById("term-progress-fill");
            const label = document.getElementById("term-progress-text");
            if (container && fill) {
                container.style.display = "block";
                fill.style.width = percent + "%";
                const countMatch = clean.match(/(\d+\/\d+)/);
                const prefixMatch = clean.match(/^([^:|]+):/);
                let txt = `Progress: ${percent}%`;
                if (countMatch) txt += ` [${countMatch[0]}]`;
                if (prefixMatch && !prefixMatch[0].includes('%')) txt = `${prefixMatch[1].trim()}: ${percent}%`;
                label.innerText = txt;
                if (percent >= 100) {
                    if (window._barT) clearTimeout(window._barT);
                    window._barT = setTimeout(() => { container.style.display = "none"; }, 4000);
                }
                return true;
            }
        }
        return false;
    }

    function appendText(raw) {
        const out = document.getElementById("term-output");
        if (!out) return;

        // 1. Split the incoming packet by tokens (\r or \n)
        const tokens = raw.split(/(\r\n|\r|\n)/);
        let wasProgressBar = false;

        tokens.forEach(tok => {
            if (tok === "\n" || tok === "\r\n") {
                _curLine = null; 
            } else if (tok === "\r") {
                if (_curLine) _curLine.innerHTML = ""; 
            } else if (tok) {
                const clean = tok.replace(/\x1B\[[0-9;]*[A-Za-z]/g, "");
                const isBar = /(\d+)%\|/.test(clean);

                if (isBar) {
                    updateVisualProgress(tok);
                    wasProgressBar = true;
                    // We don't null _curLine here so the next text 
                    // doesn't accidentally start on a new line if it shouldn't
                } else {
                    if (!_curLine) {
                        _curLine = document.createElement("div");
                        out.appendChild(_curLine);
                        if (out.children.length > MAX_LINES) out.removeChild(out.firstChild);
                    }
                    _curLine.innerHTML += ansiToHtml(tok);
                }
            }
        });

        // 2. THE FIX: Since Python sends one print() as one SSE message,
        // we must "close" the line after the message is processed,
        // BUT ONLY if it wasn't a progress bar update.
        if (!wasProgressBar) {
            _curLine = null; 
        }

        if (autoScroll) out.scrollTop = out.scrollHeight;
    }

    function startSSE() {
        if (_es) return;
        _es = new EventSource("/terminal/stream");
        _es.onmessage = (e) => { if (e.data) appendText(e.data); };
        _es.onerror = () => { _es.close(); _es = null; setTimeout(startSSE, 3000); };
    }

    function toggleTerminal() {
        visible = !visible;
        const panel = document.getElementById("term-panel");
        if (!panel) return;
        panel.classList.toggle("term-visible", visible);
        panel.style.display = visible ? "flex" : "none";
        if (_actionBtn) _actionBtn.style.background = visible ? "rgba(var(--tone3), 0.35)" : "";
        if (visible) startSSE();
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", buildPanel);
    else buildPanel();
})();