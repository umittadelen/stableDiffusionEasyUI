(function () {
    const STORAGE_KEY = "tc_settings";

    // Global helper for any extension to add a button to the action bar
    window.registerExtensionButton = function(label, url, icon) {
        const slot = document.getElementById("ext-buttons");
        if (!slot) return;
        if (!slot.hasChildNodes()) {
            const div = document.createElement("div");
            div.className = "toolbar-divider";
            slot.insertAdjacentElement("beforebegin", div);
        }
        const btn = document.createElement("button");
        btn.type = "button";
        btn.innerHTML = `<div class="btn-container">${icon ? icon : ""}<span>${label}</span></div>`;
        btn.onclick = () => window.open(url);
        slot.appendChild(btn);
    };

    // Register the theme customizer button
    function _register() {
        window.registerExtensionButton("Theme", "/theme_customizer/",
            `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" fill="currentColor"><path d="M512 256c0 .9 0 1.8 0 2.7c-.4 36.5-33.6 61.3-70.1 61.3H344c-26.5 0-48 21.5-48 48c0 3.4 .4 6.7 1 9.9c2.1 10.2 6.5 20 10.8 29.9c6.1 13.8 12.1 27.5 12.1 42c0 31.8-21.6 60.4-53.4 62c-3.5 .2-7 .3-10.6 .3C114.6 512 0 397.4 0 256S114.6 0 256 0S512 114.6 512 256zM128 288a32 32 0 1 0 -64 0 32 32 0 1 0 64 0zm0-96a32 32 0 1 0 0-64 32 32 0 1 0 0 64zM288 96a32 32 0 1 0 -64 0 32 32 0 1 0 64 0zm96 96a32 32 0 1 0 0-64 32 32 0 1 0 0 64z"/></svg>`);
    }

    if (document.getElementById("ext-buttons")) {
        _register();
    } else {
        document.addEventListener("DOMContentLoaded", _register);
    }

    function apply(cfg) {
        if (!cfg || !Object.keys(cfg).length) return;
        const root = document.documentElement;

        if (cfg.tone1) root.style.setProperty("--tone1", cfg.tone1, "important");
        if (cfg.tone2) root.style.setProperty("--tone2", cfg.tone2, "important");
        if (cfg.tone3) root.style.setProperty("--tone3", cfg.tone3, "important");

        if (cfg.bgType === "color" && cfg.bgColor) {
            document.body.style.setProperty("background", cfg.bgColor, "important");
        } else if (cfg.bgType === "image" && cfg.bgImage) {
            document.body.style.setProperty(
                "background",
                `url("${cfg.bgImage}") center/cover no-repeat fixed`,
                "important"
            );
            const hex = cfg.overlayColor || "#121218";
            const op  = cfg.overlayOpacity != null ? parseFloat(cfg.overlayOpacity) : 0.5;
            const r = parseInt(hex.slice(1,3),16);
            const g = parseInt(hex.slice(3,5),16);
            const b = parseInt(hex.slice(5,7),16);
            let el = document.getElementById("tc-overlay-style");
            if (!el) { el = document.createElement("style"); el.id = "tc-overlay-style"; document.head.appendChild(el); }
            el.textContent = `body::before{content:'';position:fixed;inset:0;background:rgba(${r},${g},${b},${op});pointer-events:none;z-index:0;}body>*{position:relative;z-index:1;}`;
        }
    }

    // 1. Paint instantly from localStorage (avoids flash)
    try { apply(JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}")); } catch(_) {}

    // 2. Fetch authoritative value from server, re-apply and sync localStorage
    fetch("/theme_customizer/load")
        .then(r => r.json())
        .then(cfg => {
            if (cfg && Object.keys(cfg).length) {
                localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
                apply(cfg);
            }
        })
        .catch(() => {});
})();
