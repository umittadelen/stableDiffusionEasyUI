(function () {
    const DEBOUNCE = 300;
    const MIN_CHARS = 2;

    const style = document.createElement("style");
    style.textContent = `
        .bt-dropdown {
            position: fixed;
            z-index: 99999;
            background: rgb(var(--tone2));
            border: 1px solid rgba(var(--tone3), 0.25);
            border-radius: 6px;
            max-height: 220px;
            overflow-y: auto;
            box-shadow: 0 4px 16px rgba(0,0,0,0.4);
            min-width: 220px;
        }
        .bt-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 10px;
            cursor: pointer;
            font-size: 0.88rem;
            gap: 12px;
        }
        .bt-item:hover, .bt-item.bt-active {
            background: rgba(var(--tone3), 0.18);
        }
        .bt-name { flex: 1; }
        .bt-count {
            font-size: 0.75rem;
            opacity: 0.5;
            white-space: nowrap;
        }
    `;
    document.head.appendChild(style);

    function fmtCount(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
        if (n >= 1000)    return (n / 1000).toFixed(1) + "k";
        return String(n);
    }

    function getWordBounds(ta) {
        const val = ta.value;
        const pos = ta.selectionStart;
        let start = pos;
        while (start > 0 && val[start - 1] !== ",") start--;
        // skip leading spaces
        while (start < pos && val[start] === " ") start++;
        return { start, end: pos, word: val.slice(start, pos).trim() };
    }

    function attachTo(ta) {
        if (ta._bt_attached) return;
        ta._bt_attached = true;

        const drop = document.createElement("div");
        drop.className = "bt-dropdown";
        drop.style.display = "none";
        document.body.appendChild(drop);

        let timer = null;
        let activeIdx = -1;
        let lastResults = [];

        function reposition() {
            const r = ta.getBoundingClientRect();
            drop.style.left = r.left + "px";
            drop.style.top  = (r.bottom + 2) + "px";
            drop.style.width = Math.max(r.width, 240) + "px";
        }

        function hide() {
            drop.style.display = "none";
            activeIdx = -1;
        }

        function render(tags) {
            lastResults = tags;
            activeIdx = -1;
            drop.innerHTML = "";
            if (!tags.length) { hide(); return; }
            tags.forEach((t, i) => {
                const item = document.createElement("div");
                item.className = "bt-item";

                const name = document.createElement("span");
                name.className = "bt-name";
                name.textContent = t.name;

                const count = document.createElement("span");
                count.className = "bt-count";
                count.textContent = fmtCount(t.count);

                item.appendChild(name);
                item.appendChild(count);
                item.addEventListener("mousedown", e => {
                    e.preventDefault();
                    insert(t.name);
                });
                drop.appendChild(item);
            });
            reposition();
            drop.style.display = "block";
        }

        function insert(tagName) {
            const { start, end } = getWordBounds(ta);
            let before = ta.value.slice(0, start).replace(/\s*$/, "");
            let after  = ta.value.slice(end).replace(/^\s*/, "");

            // Always ensure ', ' before unless at start
            if (before.length > 0 && !before.endsWith(", ")) {
                before = before.replace(/,?\s*$/, ", ");
            }

            // Always ensure ', ' after unless at end
            if (after.length > 0 && !after.startsWith(", ")) {
                after = after.replace(/^,?\s*/, ", ");
            }

            ta.value = before + tagName + after;
            // Place cursor right after the inserted tag and the ", " if present
            let cursorPos = (before + tagName).length;
            if (after.startsWith(", ")) {
                cursorPos += 2;
            }
            ta.selectionStart = ta.selectionEnd = cursorPos;
            ta.dispatchEvent(new Event("input"));
            hide();
            ta.focus();
        }

        function setActive(idx) {
            const items = drop.querySelectorAll(".bt-item");
            items.forEach(el => el.classList.remove("bt-active"));
            if (idx >= 0 && idx < items.length) {
                items[idx].classList.add("bt-active");
                items[idx].scrollIntoView({ block: "nearest" });
            }
            activeIdx = idx;
        }

        ta.addEventListener("input", () => {
            clearTimeout(timer);
            const { word } = getWordBounds(ta);
            if (word.length < MIN_CHARS) { hide(); return; }
            timer = setTimeout(async () => {
                try {
                    const safe = encodeURIComponent(word.replace(/[^\w\s\-]/g, ""));
                    const res = await fetch(`/booru_tags/search?q=${safe}`);
                    render(await res.json());
                } catch (_) { hide(); }
            }, DEBOUNCE);
        });

        ta.addEventListener("keydown", e => {
            if (drop.style.display === "none") return;
            if (e.key === "ArrowDown") {
                e.preventDefault();
                setActive(Math.min(activeIdx + 1, lastResults.length - 1));
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setActive(Math.max(activeIdx - 1, 0));
            } else if (e.key === "Enter" || e.key === "Tab") {
                if (activeIdx >= 0) {
                    e.preventDefault();
                    insert(lastResults[activeIdx].name);
                }
            } else if (e.key === "Escape") {
                hide();
            }
        });

        ta.addEventListener("blur", () => setTimeout(hide, 150));
        window.addEventListener("scroll", reposition, true);
        window.addEventListener("resize", reposition);
    }

    function init() {
        const prompt   = document.getElementById("prompt");
        const negative = document.getElementById("negative_prompt");
        if (prompt)   attachTo(prompt);
        if (negative) attachTo(negative);
    }

    if (document.getElementById("prompt")) init();
    else document.addEventListener("DOMContentLoaded", init);
})();
