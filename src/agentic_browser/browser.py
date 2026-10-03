"""Browser layer: a real Chromium driven through Playwright.

This is the v0 answer to Polar's key architectural lesson (Composer/Electron ->
Chromium fork): an agent needs a *real* browser engine with real sessions,
cookies, and extension-capable plumbing. We don't fork Chromium in v0 — we drive
upstream Chromium via Playwright, which gives us CDP-level control, and we keep
a persistent profile directory so logins survive restarts (the "logged in as
them" property).

Perception: screenshot + a compact, ref-tagged interactive-element tree built
by injected JS. Actions address elements by ref (e.g. click "e12"), which avoids
the brittleness of coordinate prediction in v0. Coordinate grounding is a
planned upgrade (see README milestones).
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

_GROUND_JS = """
() => {
  const els = [];
  const selector = 'a, button, input, select, textarea, [role=button], [role=link], [role=checkbox], [role=radio], [role=switch], [role=tab], [role=menuitem], [contenteditable=true]';
  let n = 0;
  const seen = new Set();
  for (const el of document.querySelectorAll(selector)) {
    if (seen.has(el)) continue;
    const r = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    if (r.width < 2 || r.height < 2) continue;
    if (style.visibility === 'hidden' || style.display === 'none') continue;
    const ref = 'e' + (++n);
    el.setAttribute('data-agent-ref', ref);
    seen.add(el);
    let name = (el.getAttribute('aria-label') || '').trim();
    if (!name) {
      const text = (el.innerText || el.value || '').trim().replace(/\\s+/g, ' ');
      name = text.slice(0, 80);
    }
    if (!name) name = (el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('name') || '').trim().slice(0, 80);
    let role = el.getAttribute('role') || el.tagName.toLowerCase();
    if (role === 'input') role = (el.getAttribute('type') || 'text') + ' input';
    els.push({ref, role, name: name || '(no label)', x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2)});
    if (n >= 250) break;
  }
  return els;
}
"""


def _proxy_from_env() -> dict | None:
    """Pick up an egress proxy (with auth) from the environment, if set.

    Playwright does not inherit proxy env vars on its own, so we parse them
    here. Credentials stay in the environment — never in source.
    Returns {"server", "host", "port", "username"(?), "password"(?)}.
    """
    for var in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        parts = urlparse(raw)
        if not parts.hostname:
            continue
        proxy: dict = {
            "server": f"{parts.scheme or 'http'}://{parts.hostname}:{parts.port or 3128}",
            "host": parts.hostname,
            "port": parts.port or 3128,
        }
        if parts.username:
            proxy["username"] = parts.username
        if parts.password:
            proxy["password"] = parts.password
        return proxy
    return None


def _start_local_relay(upstream_host: str, upstream_port: int) -> int:
    """Start a localhost TCP relay to the upstream proxy, in daemon threads.

    Needed in locked-down environments (CI sandboxes, filtered egress) where
    the Chromium binary's own TCP connections are interfered with but a local
    Python process may relay freely. Protocol-agnostic: handles both plain
    HTTP proxying and CONNECT tunnels. Returns the local port.
    """
    import socket as _socket
    import threading as _threading

    srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(50)
    port = srv.getsockname()[1]

    def _fwd(a: "_socket.socket", b: "_socket.socket") -> None:
        try:
            while True:
                d = a.recv(65536)
                if not d:
                    break
                b.sendall(d)
        except OSError:
            pass

    def _handle(client: "_socket.socket") -> None:
        try:
            up = _socket.create_connection((upstream_host, upstream_port), timeout=15)
        except OSError:
            client.close()
            return
        t1 = _threading.Thread(target=_fwd, args=(client, up), daemon=True)
        t2 = _threading.Thread(target=_fwd, args=(up, client), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        client.close()
        up.close()

    def _accept() -> None:
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                break
            _threading.Thread(target=_handle, args=(c,), daemon=True).start()

    _threading.Thread(target=_accept, daemon=True).start()
    return port


@dataclass
class Observation:
    url: str
    title: str
    elements: list[dict]
    screenshot_b64: str | None

    def to_text(self) -> str:
        lines = [f"URL: {self.url}", f"Title: {self.title}", "Interactive elements:"]
        for el in self.elements:
            lines.append(f"  [{el['ref']}] {el['role']}: {el['name']}")
        if not self.elements:
            lines.append("  (none found — page may still be loading; try snapshot again)")
        return "\n".join(lines)


class BrowserSession:
    """One agent's browser: one Chromium context, one page."""

    def __init__(
        self,
        headless: bool = True,
        profile_dir: str | Path | None = None,
        viewport: tuple[int, int] = (1280, 800),
    ) -> None:
        self.headless = headless
        self.profile_dir = Path(profile_dir) if profile_dir else None
        self.viewport = viewport
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._attached = False  # True when driving someone else's browser over CDP

    def start(self) -> "BrowserSession":
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        viewport = {"width": self.viewport[0], "height": self.viewport[1]}
        common: dict = {
            "headless": self.headless,
            "viewport": viewport,
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
        }
        proxy = _proxy_from_env()
        if proxy:
            if os.environ.get("AGENTIC_RELAY_PROXY") == "1":
                # Sandbox/CI mode: Chromium's own egress is filtered; relay
                # through a localhost forwarder running in this process.
                # The sandbox egress proxy also TLS-intercepts, so cert
                # validation is disabled IN THIS MODE ONLY. Never enable
                # outside a trusted sandbox.
                relay_port = _start_local_relay(proxy["host"], proxy["port"])
                proxy = {"server": f"http://127.0.0.1:{relay_port}"}
                common["args"].append("--ignore-certificate-errors")
            # Playwright's proxy option only takes server/username/password.
            common["proxy"] = {
                k: proxy[k] for k in ("server", "username", "password") if k in proxy
            }
        if self.profile_dir:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            self._context = self._pw.chromium.launch_persistent_context(
                str(self.profile_dir), **common
            )
            self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        else:
            launch_kwargs = {k: v for k, v in common.items() if k != "viewport"}
            self._browser = self._pw.chromium.launch(**launch_kwargs)
            self._context = self._browser.new_context(
                viewport=viewport, proxy=common.get("proxy")
            )
            self._page = self._context.new_page()
        return self

    def attach_cdp(self, cdp_url: str, url_match: str | None = None) -> "BrowserSession":
        """Attach to a running Chromium (e.g. the Frontier browser app) over CDP
        and take over one of its tabs.

        Selection order: exact URL match, substring match, first http(s) page.
        The host app's own chrome (file://…/renderer/…) is never selected.
        """
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.connect_over_cdp(cdp_url)
        pages = [p for ctx in self._browser.contexts for p in ctx.pages]

        def eligible(p) -> bool:
            return "/renderer/" not in p.url  # never drive the app's own UI

        cands = [p for p in pages if eligible(p)]
        page = None
        if url_match:
            page = next((p for p in cands if p.url == url_match), None)
            if page is None:
                page = next((p for p in cands if url_match in p.url), None)
        if page is None:
            page = next((p for p in cands if p.url.startswith("http")), None)
        if page is None:
            raise RuntimeError(f"No attachable web tab found via {cdp_url}")
        self._page = page
        self._context = page.context
        self._attached = True
        return self

    def close(self) -> None:
        try:
            if self._attached:
                # Driving someone else's browser: never close their tabs or
                # contexts — just drop our CDP connection.
                if self._browser:
                    try:
                        self._browser.close()
                    except Exception:
                        pass
            else:
                if self._context:
                    self._context.close()
                if self._browser:
                    self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()
            self._pw = self._browser = self._context = self._page = None
            self._attached = False

    # -- observation -----------------------------------------------------
    def observe(self, with_screenshot: bool = True) -> Observation:
        elements = self._page.evaluate(_GROUND_JS)
        shot = None
        if with_screenshot:
            shot = base64.b64encode(self._page.screenshot()).decode()
        return Observation(
            url=self._page.url,
            title=self._page.title(),
            elements=elements,
            screenshot_b64=shot,
        )

    def page_text(self, max_chars: int = 8000) -> str:
        text = self._page.evaluate("() => document.body ? document.body.innerText : ''")
        return text[:max_chars]

    # -- actions ----------------------------------------------------------
    def _loc(self, ref: str):
        return self._page.locator(f'[data-agent-ref="{ref}"]')

    def new_tab(self, url: str | None = None) -> str:
        """Open a new tab and make it the driven page. The old tab stays
        open — the session just follows the new one."""
        if self._context is None:
            raise RuntimeError("browser not started")
        self._page = self._context.new_page()
        if url:
            return self.navigate(url)
        return f"New tab opened: {self._page.url}"

    def navigate(self, url: str) -> str:
        if "://" not in url:
            url = "https://" + url
        self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
        try:
            self._page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        return f"Navigated to {self._page.url} — title: {self._page.title()!r}"

    def click(self, ref: str) -> str:
        loc = self._loc(ref)
        loc.scroll_into_view_if_needed(timeout=5000)
        loc.click(timeout=10000)
        self._page.wait_for_timeout(800)
        return f"Clicked [{ref}]. URL now: {self._page.url}"

    def fill(self, ref: str, text: str, submit: bool = False) -> str:
        loc = self._loc(ref)
        loc.scroll_into_view_if_needed(timeout=5000)
        loc.click(timeout=5000)
        loc.fill(text, timeout=10000)
        if submit:
            self._page.keyboard.press("Enter")
            self._page.wait_for_timeout(1200)
        return f"Filled [{ref}] with {text!r}{' and submitted' if submit else ''}."

    def press(self, key: str) -> str:
        self._page.keyboard.press(key, timeout=5000)
        self._page.wait_for_timeout(600)
        return f"Pressed {key!r}. URL now: {self._page.url}"

    def scroll(self, direction: str = "down", pixels: int = 600) -> str:
        dy = pixels if direction == "down" else -pixels
        self._page.mouse.wheel(0, dy)
        self._page.wait_for_timeout(500)
        return f"Scrolled {direction} {pixels}px."

    def hover(self, ref: str) -> str:
        self._loc(ref).hover(timeout=5000)
        return f"Hovered [{ref}]."

    def select_option(self, ref: str, value: str) -> str:
        self._loc(ref).select_option(value, timeout=8000)
        return f"Selected {value!r} in [{ref}]."

    def go_back(self) -> str:
        self._page.go_back(wait_until="domcontentloaded", timeout=15000)
        return f"Went back. URL now: {self._page.url}"

    def go_forward(self) -> str:
        self._page.go_forward(wait_until="domcontentloaded", timeout=15000)
        return f"Went forward. URL now: {self._page.url}"

    def save_screenshot(self, path: str | Path) -> str:
        self._page.screenshot(path=str(path))
        return f"Screenshot saved to {path}"
