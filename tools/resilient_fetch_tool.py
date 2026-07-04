#!/usr/bin/env python3
"""Resilient web fetch — a scrappy, self-hosted anti-bot fetch path.

Why this exists
---------------
The model kept hand-rolling raw HTTP requests in ``execute_code`` to read web
pages. Raw ``requests``/``curl`` send a non-browser TLS + HTTP fingerprint, so
Cloudflare/Akamai/DataDome flag them as bots and return 403/429/CAPTCHA or soft
404 pages — which sent the agent into retry loops (e.g. levels.fyi).

This tool gives the agent one call that tries progressively harder to look like
a real browser, instead of looping on a blocked raw fetch:

  Tier 1 — ``curl_cffi`` with ``impersonate="chrome"``: matches a real Chrome
           TLS/JA3 + HTTP-2 fingerprint and realistic headers. Cheap, fast,
           defeats the large *fingerprint-based* tier of bot blocking.
  Tier 2 — if Tier 1 still looks blocked (or ``render=True``), fall back to the
           real Chromium the harness already drives over CDP, which executes the
           JavaScript challenge. We capture the resulting ``cf_clearance`` cookie
           and cache it per-domain so subsequent Tier-1 calls reuse it cheaply.

Honest limits: without residential proxies this will not reliably beat the
hardest targets (top-tier Cloudflare/DataDome/PerimeterX). When both tiers are
blocked the tool says so plainly so the agent switches to a sanctioned source
(official API / public dataset) rather than looping.

All third-party imports are lazy so tool-discovery never fails when an optional
dependency (curl_cffi) is absent.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Stealth evasions injected into every new document before page scripts run
# (Tier 2). Mirrors the well-known puppeteer-extra-stealth / rebrowser tricks:
# hide the automation tells (navigator.webdriver, missing window.chrome, empty
# plugins, headless WebGL vendor) that Cloudflare/DataDome JS challenges probe.
# Honest limit: this does NOT fix the CDP Runtime.enable leak that the newest
# anti-bot stacks check — for those you still need residential proxies or a
# sanctioned source. It meaningfully raises the pass rate on the common tier.
_STEALTH_JS = r"""
(() => {
  if (window.__hermesStealth) return;
  window.__hermesStealth = true;
  const def = (obj, prop, val) => {
    try { Object.defineProperty(obj, prop, { get: () => val, configurable: true }); }
    catch (e) {}
  };
  // 1. navigator.webdriver — the single biggest tell.
  def(Object.getPrototypeOf(navigator), 'webdriver', undefined);
  try { delete Navigator.prototype.webdriver; } catch (e) {}
  // 2. window.chrome stub (headless Chromium lacks it).
  if (!window.chrome) { window.chrome = {}; }
  if (!window.chrome.runtime) { window.chrome.runtime = {}; }
  // 3. Plausible languages.
  def(navigator, 'languages', ['en-US', 'en']);
  // 4. Non-empty plugins / mimeTypes.
  def(navigator, 'plugins', [1, 2, 3, 4, 5]);
  def(navigator, 'mimeTypes', [1, 2]);
  // 5. Hardware that matches a real laptop, not a stripped VM.
  def(navigator, 'hardwareConcurrency', 8);
  def(navigator, 'deviceMemory', 8);
  // 6. permissions.query for notifications (headless returns 'denied' oddly).
  try {
    const orig = window.navigator.permissions.query.bind(window.navigator.permissions);
    window.navigator.permissions.query = (p) =>
      p && p.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : orig(p);
  } catch (e) {}
  // 7. WebGL vendor/renderer — headless reports 'Google SwiftShader'.
  try {
    const getParam = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function (p) {
      if (p === 37445) return 'Intel Inc.';            // UNMASKED_VENDOR_WEBGL
      if (p === 37446) return 'Intel Iris OpenGL Engine'; // UNMASKED_RENDERER_WEBGL
      return getParam.call(this, p);
    };
  } catch (e) {}
})();
"""

# Per-domain cookie cache (persists a solved cf_clearance across calls).
_COOKIE_DIR = os.path.join(
    os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes")),
    "cache",
    "resilient_fetch",
)

# Markers that indicate an anti-bot interstitial rather than real content.
_CHALLENGE_MARKERS = (
    "just a moment",
    "checking your browser",
    "attention required",
    "cf-browser-verification",
    "cf-challenge",
    "_cf_chl_opt",
    "challenge-platform",
    "/cdn-cgi/challenge-platform",
    "please enable javascript and cookies",
    "verifying you are human",
    "ddos-guard",
    "px-captcha",
    "perimeterx",
    "access denied",
    "request unsuccessful. incapsula",
)

_BLOCK_STATUSES = {401, 403, 406, 409, 429, 503}


# ──────────────────────────── helpers ────────────────────────────


def _curl_cffi_available() -> bool:
    try:
        import curl_cffi  # noqa: F401
        return True
    except Exception:
        return False


def _domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _cookie_path(domain: str) -> str:
    safe = re.sub(r"[^a-z0-9._-]", "_", domain)
    return os.path.join(_COOKIE_DIR, f"{safe}.json")


def _load_cookies(domain: str) -> Dict[str, str]:
    try:
        with open(_cookie_path(domain), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_cookies(domain: str, cookies: Dict[str, str]) -> None:
    if not cookies:
        return
    try:
        os.makedirs(_COOKIE_DIR, exist_ok=True)
        merged = _load_cookies(domain)
        merged.update({str(k): str(v) for k, v in cookies.items()})
        with open(_cookie_path(domain), "w", encoding="utf-8") as fh:
            json.dump(merged, fh)
    except Exception as exc:  # pragma: no cover — best-effort cache
        logger.debug("resilient_fetch: cookie save failed for %s: %s", domain, exc)


def _proxy_pool() -> List[str]:
    """Residential/rotating proxies, in precedence order.

    Source: ``HERMES_FETCH_PROXY`` (one or a comma-separated list — set it in
    the profile ``.env``), falling back to the standard ``HTTPS_PROXY`` /
    ``HTTP_PROXY`` environment variables. Each entry is a full proxy URL, e.g.
    ``http://user:pass@gate.smartproxy.com:7000``. A list is rotated per call so
    each request can exit from a different residential IP — the piece DIY
    fingerprinting can't otherwise replicate.
    """
    raw = os.environ.get("HERMES_FETCH_PROXY", "").strip()
    if raw:
        pool = [p.strip() for p in raw.split(",") if p.strip()]
        if pool:
            return pool
    for env in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.environ.get(env, "").strip()
        if val:
            return [val]
    return []


def _resolve_proxy(explicit: Optional[str]) -> str:
    """Pick a proxy URL for this call (explicit arg wins; else rotate the pool)."""
    if explicit and explicit.strip():
        return explicit.strip()
    pool = _proxy_pool()
    if not pool:
        return ""
    return random.choice(pool)


def looks_blocked(status: int, body: str) -> Tuple[bool, str]:
    """Heuristic: does this response look like an anti-bot interstitial?

    Returns ``(blocked, reason)``. Kept pure so it is unit-testable without
    any network access.
    """
    if status in _BLOCK_STATUSES:
        return True, f"http_{status}"
    head = (body or "")[:4000].lower()
    for marker in _CHALLENGE_MARKERS:
        if marker in head:
            return True, f"challenge_marker:{marker}"
    # A tiny body on a 200 is often a JS-challenge shell.
    if status == 200 and len(body or "") < 512 and "cf-" in head:
        return True, "tiny_cf_shell"
    return False, ""


def _strip_to_text(html: str, limit: int) -> str:
    """Very light HTML→text so the model gets readable content, not markup."""
    text = re.sub(r"(?is)<(script|style|noscript|template)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|li|h[1-6]|tr)>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()[:limit]


def _title(html: str) -> str:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html or "")
    return re.sub(r"\s+", " ", m.group(1)).strip()[:200] if m else ""


# ──────────────────────────── tiers ────────────────────────────


def _tier1_curl_cffi(url: str, timeout: float, proxy: str = "") -> Dict[str, Any]:
    from curl_cffi import requests as creq  # lazy

    domain = _domain(url)
    cached = _load_cookies(domain)
    headers = {
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Upgrade-Insecure-Requests": "1",
    }
    kwargs: Dict[str, Any] = {}
    if proxy:
        kwargs["proxies"] = {"http": proxy, "https": proxy}
    resp = creq.get(
        url,
        headers=headers,
        cookies=cached or None,
        impersonate="chrome",
        timeout=timeout,
        allow_redirects=True,
        **kwargs,
    )
    body = resp.text or ""
    # Persist any fresh cookies (incl. cf_clearance) the server handed back.
    try:
        _save_cookies(domain, {c.name: c.value for c in resp.cookies.jar})
    except Exception:
        try:
            _save_cookies(domain, dict(resp.cookies))
        except Exception:
            pass
    blocked, reason = looks_blocked(resp.status_code, body)
    return {
        "status": resp.status_code,
        "final_url": str(resp.url),
        "body": body,
        "blocked": blocked,
        "reason": reason,
    }


def _tier2_browser(url: str, timeout: float) -> Optional[Dict[str, Any]]:
    """Render via the harness's real Chromium over CDP, if one is reachable.

    Returns a result dict, or ``None`` when no browser/CDP endpoint is
    available so the caller degrades gracefully instead of crashing.
    """
    try:
        from tools.browser_cdp_tool import _browser_cdp_check, browser_cdp
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("resilient_fetch: browser_cdp import failed: %s", exc)
        return None
    try:
        if not _browser_cdp_check():
            return None
    except Exception:
        return None

    target_id = None
    try:
        # Open a blank tab first so we can install the stealth evasions BEFORE
        # any page script runs, then navigate. addScriptToEvaluateOnNewDocument
        # applies to the next document load (the navigate below).
        created = json.loads(
            browser_cdp("Target.createTarget", {"url": "about:blank"}, timeout=timeout)
        )
        target_id = (created.get("result") or {}).get("targetId")
        if not target_id:
            return None
        browser_cdp("Page.enable", {}, target_id=target_id, timeout=timeout)
        browser_cdp(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": _STEALTH_JS},
            target_id=target_id,
            timeout=timeout,
        )
        browser_cdp("Page.navigate", {"url": url}, target_id=target_id, timeout=timeout)
        # Give the challenge time to run and set cf_clearance.
        time.sleep(min(8.0, max(3.0, timeout / 4)))
        evaled = json.loads(
            browser_cdp(
                "Runtime.evaluate",
                {"expression": "document.documentElement.outerHTML", "returnByValue": True},
                target_id=target_id,
                timeout=timeout,
            )
        )
        html = (((evaled.get("result") or {}).get("result")) or {}).get("value") or ""
        cookies_raw = json.loads(browser_cdp("Network.getAllCookies", {}, timeout=timeout))
        cookies = {
            c["name"]: c["value"]
            for c in ((cookies_raw.get("result") or {}).get("cookies") or [])
            if _domain(url).endswith(str(c.get("domain", "")).lstrip("."))
            or str(c.get("domain", "")).lstrip(".") in _domain(url)
        }
        if cookies:
            _save_cookies(_domain(url), cookies)
        blocked, reason = looks_blocked(200, html)
        return {"status": 200, "final_url": url, "body": html, "blocked": blocked, "reason": reason}
    except Exception as exc:
        logger.debug("resilient_fetch: tier2 failed: %s", exc)
        return None
    finally:
        if target_id:
            try:
                browser_cdp("Target.closeTarget", {"targetId": target_id}, timeout=5.0)
            except Exception:
                pass


# ──────────────────────────── entry point ────────────────────────────


def resilient_fetch(
    url: str,
    render: bool = False,
    timeout: float = 30.0,
    max_chars: int = 20000,
    raw: bool = False,
    proxy: Optional[str] = None,
) -> str:
    """Fetch ``url`` through the tiered anti-bot path. Returns a JSON string.

    Args:
        url: absolute http(s) URL.
        render: skip straight to the real-browser tier (for JS-heavy pages).
        timeout: per-request timeout seconds.
        max_chars: cap on returned text/html.
        raw: include truncated raw HTML in addition to extracted text.
        proxy: explicit proxy URL; defaults to the HERMES_FETCH_PROXY pool.
    """
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        return tool_error("'url' must be an absolute http(s) URL.")
    try:
        timeout = max(1.0, min(float(timeout), 120.0))
    except (TypeError, ValueError):
        timeout = 30.0
    try:
        max_chars = max(500, min(int(max_chars), 200000))
    except (TypeError, ValueError):
        max_chars = 20000

    if not _curl_cffi_available():
        return tool_error(
            "curl_cffi is not installed. Install it into the Hermes venv: "
            "uv pip install --python <hermes-venv>/bin/python curl_cffi"
        )

    tiers_tried = []
    result: Optional[Dict[str, Any]] = None
    proxy_url = _resolve_proxy(proxy)

    if not render:
        tiers_tried.append("curl_cffi+proxy" if proxy_url else "curl_cffi")
        try:
            result = _tier1_curl_cffi(url, timeout, proxy=proxy_url)
        except Exception as exc:
            logger.debug("resilient_fetch: tier1 error: %s", exc)
            result = None

    # Escalate to the real browser when Tier 1 was skipped, errored, or blocked.
    if render or result is None or result.get("blocked"):
        t2 = _tier2_browser(url, timeout)
        tiers_tried.append("browser_cdp")
        if t2 is not None:
            result = t2

    if result is None:
        return json.dumps(
            {
                "ok": False,
                "url": url,
                "tiers_tried": tiers_tried,
                "blocked": True,
                "error": "All fetch tiers failed (Tier 1 errored; no browser/CDP "
                "endpoint for Tier 2). Switch to an official API or public dataset, "
                "or run '/browser connect' to enable the browser tier.",
            },
            ensure_ascii=False,
        )

    html = result.get("body") or ""
    text = _strip_to_text(html, max_chars)
    payload: Dict[str, Any] = {
        "ok": not result.get("blocked"),
        "url": url,
        "final_url": result.get("final_url"),
        "status": result.get("status"),
        "tiers_tried": tiers_tried,
        "tier_used": tiers_tried[-1] if tiers_tried else None,
        "blocked": bool(result.get("blocked")),
        "title": _title(html),
        "text": text,
        "text_truncated": len(_strip_to_text(html, max_chars + 1)) > len(text),
    }
    if result.get("blocked"):
        payload["note"] = (
            f"Still blocked after {('+'.join(tiers_tried)) or 'all tiers'} "
            f"({result.get('reason')}). Do not retry this URL — switch to an "
            "official API or public dataset, or report the blocker to the user."
        )
    if raw:
        payload["raw_html"] = html[:max_chars]
    return json.dumps(payload, ensure_ascii=False)


# ──────────────────────────── registration ────────────────────────────

RESILIENT_FETCH_SCHEMA: Dict[str, Any] = {
    "name": "fetch_resilient",
    "description": (
        "Fetch a web page through a tiered anti-bot path that mimics a real "
        "browser, instead of a raw HTTP request that gets flagged as a bot. "
        "Use this for any single-page fetch — especially when a normal fetch "
        "returned 403/429/CAPTCHA or a Cloudflare 'Just a moment' / soft-404 "
        "page.\n\n"
        "Tier 1 uses a real Chrome TLS fingerprint (curl_cffi), optionally "
        "through a rotating residential proxy; if still blocked it escalates to "
        "the real browser (with anti-bot stealth evasions injected) when one is "
        "connected, and caches the cleared cookie for next time. If BOTH tiers "
        "are blocked it returns blocked=true with a note — do NOT retry the "
        "same URL; switch to an official API or public dataset.\n\n"
        "Returns JSON: {ok, status, blocked, title, text, tier_used, note}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL to fetch."},
            "render": {
                "type": "boolean",
                "description": "Skip straight to the real-browser tier for JS-heavy pages.",
            },
            "timeout": {"type": "number", "description": "Per-request timeout seconds (default 30)."},
            "max_chars": {"type": "integer", "description": "Cap on returned text (default 20000)."},
            "raw": {"type": "boolean", "description": "Also include truncated raw HTML."},
            "proxy": {
                "type": "string",
                "description": "Explicit proxy URL; defaults to the HERMES_FETCH_PROXY pool.",
            },
        },
        "required": ["url"],
    },
}


registry.register(
    name="fetch_resilient",
    toolset="web",
    schema=RESILIENT_FETCH_SCHEMA,
    handler=lambda args, **kw: resilient_fetch(
        url=args.get("url", ""),
        render=bool(args.get("render", False)),
        timeout=args.get("timeout", 30.0),
        max_chars=args.get("max_chars", 20000),
        raw=bool(args.get("raw", False)),
        proxy=args.get("proxy"),
    ),
    check_fn=_curl_cffi_available,
    emoji="🛡️",
    max_result_size_chars=200000,
)


if __name__ == "__main__":  # manual smoke: python -m tools.resilient_fetch_tool <url>
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print(resilient_fetch(target, render="--render" in sys.argv))
