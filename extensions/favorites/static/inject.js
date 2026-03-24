(function () {
    // ── Register action bar button ──────────────────────────────────────────
    function _registerButton() {
        if (typeof window.registerExtensionButton === "function") {
            window.registerExtensionButton(
                "Favorites",
                "/favorites/",
                `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512" fill="currentColor"><path d="M316.9 18C311.6 7 300.4 0 288.1 0s-23.4 7-28.8 18L195 150.3 51.4 171.5c-12 1.8-22 10.2-25.7 21.7s-.7 24.2 7.9 32.7L137.8 329 113.2 474.7c-2 12 3 24.2 12.9 31.3s23 8 33.8 2.3l128.3-68.5 128.3 68.5c10.8 5.7 23.9 4.9 33.8-2.3s14.9-19.3 12.9-31.3L438.5 329 542.7 225.9c8.6-8.5 11.7-21.2 7.9-32.7s-13.7-19.9-25.7-21.7L381.2 150.3 316.9 18z"/></svg>`
            );
        }
    }
    if (document.getElementById("ext-buttons")) _registerButton();
    else document.addEventListener("DOMContentLoaded", _registerButton);

    // ── Shared helpers ──────────────────────────────────────────────────────
    const STAR_FILLED = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512" fill="currentColor" width="14" height="14"><path d="M316.9 18C311.6 7 300.4 0 288.1 0s-23.4 7-28.8 18L195 150.3 51.4 171.5c-12 1.8-22 10.2-25.7 21.7s-.7 24.2 7.9 32.7L137.8 329 113.2 474.7c-2 12 3 24.2 12.9 31.3s23 8 33.8 2.3l128.3-68.5 128.3 68.5c10.8 5.7 23.9 4.9 33.8-2.3s14.9-19.3 12.9-31.3L438.5 329 542.7 225.9c8.6-8.5 11.7-21.2 7.9-32.7s-13.7-19.9-25.7-21.7L381.2 150.3 316.9 18z"/></svg>`;
    const STAR_EMPTY  = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512" fill="currentColor" width="14" height="14"><path d="M287.9 0c9.2 0 17.6 5.2 21.6 13.5l68.6 141 153.2 22.6c9 1.3 16.5 7.6 19.3 16.3s.5 18.1-5.9 24.5L433.6 328.4l26.2 155.6c1.5 9-2.2 18.1-9.7 23.5s-17.3 6-25.3 1.7l-137-73.2-137 73.2c-8 4.3-17.9 3.7-25.3-1.7s-11.2-14.5-9.7-23.5l26.2-155.6L31.1 218c-6.5-6.4-8.7-15.9-5.9-24.5s10.3-14.9 19.3-16.3l153.2-22.6L266.3 13.5C270.4 5.2 278.7 0 287.9 0zm0 79L235.4 187.2c-3.5 7.1-10.2 12.1-18.1 13.3L99 217.9 184.9 303c5.5 5.4 8.1 13.2 6.8 20.8L171.4 443.7l105.2-56.2c7.1-3.8 15.6-3.8 22.6 0l105.2 56.2L384.2 323.8c-1.3-7.6 1.2-15.4 6.8-20.8l85.9-85.1-118.3-17.4c-7.8-1.2-14.6-6.1-18.1-13.3L287.9 79z"/></svg>`;

    function makeStarBtn(src, isFavorited) {
        const btn = document.createElement("button");
        btn.className = "fav-star" + (isFavorited ? " fav-star--on" : "");
        btn.title = isFavorited ? "Remove from favorites" : "Add to favorites";
        btn.innerHTML = isFavorited ? STAR_FILLED : STAR_EMPTY;
        btn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const on = btn.classList.contains("fav-star--on");
            const filename = src.split("/").pop().split("?")[0];
            if (on) {
                await fetch("/favorites/remove", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ filename })
                });
                btn.classList.remove("fav-star--on");
                btn.title = "Add to favorites";
                btn.innerHTML = STAR_EMPTY;
            } else {
                await fetch("/favorites/add", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ src })
                });
                btn.classList.add("fav-star--on");
                btn.title = "Remove from favorites";
                btn.innerHTML = STAR_FILLED;
            }
        });
        return btn;
    }

    async function isFavorited(filename) {
        const r = await fetch(`/favorites/check?filename=${encodeURIComponent(filename)}`);
        return (await r.json()).favorited;
    }

    // ── Inject shared CSS once ──────────────────────────────────────────────
    function injectCSS() {
        if (document.getElementById("fav-style")) return;
        const s = document.createElement("style");
        s.id = "fav-style";
        s.textContent = `
            .fav-star {
                padding: 0px !important;
                position: absolute;
                top: 6px; left: 6px;
                background: rgba(0,0,0,0.55);
                border: none;
                border-radius: 6px;
                color: #ccc;
                width: 28px; height: 28px;
                cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                opacity: 0.7;
                transform: scale(1);
                box-shadow: none;
                transition: opacity 0.18s, color 0.18s, transform 0.18s, box-shadow 0.18s;
                z-index: 2;
            }
            .fav-star--on { color: gold; opacity: 1 !important; }
            .image-wrapper:hover .fav-star,
            .card:hover .fav-star {
                opacity: 1;
                transform: scale(1.18);
                background: rgba(0,0,0,0.72);
                box-shadow: 0 2px 8px 0 rgba(0,0,0,0.22);
            }
        `;
        document.head.appendChild(s);
    }

    // ── Main gallery (index page) ───────────────────────────────────────────
    function isIndexPage() {
        return !!document.getElementById("images");
    }

    function attachStarToWrapper(wrapper) {
        if (wrapper.querySelector(".fav-star")) return;
        const img = wrapper.querySelector("img");
        if (!img) return;
        const src = img.src.split("?")[0].replace(window.location.origin, "");
        const filename = src.split("/").pop();
        const btn = makeStarBtn(src, false);
        wrapper.style.position = "relative";
        wrapper.appendChild(btn);
        isFavorited(filename).then(on => {
            if (on) {
                btn.classList.add("fav-star--on");
                btn.title = "Remove from favorites";
                btn.innerHTML = STAR_FILLED;
            }
        });
    }

    function observeGallery() {
        function scanAndAttach() {
            const gallery = document.getElementById("images");
            if (!gallery) return;
            gallery.querySelectorAll(".image-wrapper").forEach(attachStarToWrapper);
        }
        // Initial scan
        scanAndAttach();
        // Watch for new ones
        const gallery = document.getElementById("images");
        if (gallery) {
            new MutationObserver(mutations => {
                mutations.forEach(m => m.addedNodes.forEach(n => {
                    if (n.classList?.contains("image-wrapper")) attachStarToWrapper(n);
                }));
            }).observe(gallery, { childList: true, subtree: true });
        }
        // Fallback: periodic scan in case images are added in bulk or by replacement
        setInterval(scanAndAttach, 1000);
    }

    // ── Image History page ──────────────────────────────────────────────────
    function isHistoryPage() {
        return window.location.pathname.startsWith("/image_history");
    }

    function attachStarToCard(card) {
        if (card.querySelector(".fav-star")) return;
        const img = card.querySelector("img");
        if (!img) return;
        const src = img.src.replace(window.location.origin, "").split("?")[0];
        const filename = src.split("/").pop();
        const btn = makeStarBtn(src, false);
        card.appendChild(btn);
        isFavorited(filename).then(on => {
            if (on) {
                btn.classList.add("fav-star--on");
                btn.title = "Remove from favorites";
                btn.innerHTML = STAR_FILLED;
            }
        });
    }

    function observeHistoryGallery() {
        const gallery = document.getElementById("gallery");
        if (!gallery) return;
        gallery.querySelectorAll(".card").forEach(attachStarToCard);
        new MutationObserver(mutations => {
            mutations.forEach(m => m.addedNodes.forEach(n => {
                if (n.classList?.contains("card")) attachStarToCard(n);
            }));
        }).observe(gallery, { childList: true });
    }

    // ── Boot ────────────────────────────────────────────────────────────────
    function init() {
        injectCSS();
        if (isIndexPage())    observeGallery();
        if (isHistoryPage())  observeHistoryGallery();
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})();
