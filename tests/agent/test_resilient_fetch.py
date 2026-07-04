"""Unit tests for the resilient anti-bot fetch tool (pure, no network)."""

import json
import os
import tempfile

import tools.resilient_fetch_tool as rf


def test_looks_blocked_flags_block_statuses():
    assert rf.looks_blocked(403, "<html>ok</html>")[0] is True
    assert rf.looks_blocked(429, "")[0] is True
    assert rf.looks_blocked(503, "")[0] is True


def test_looks_blocked_flags_challenge_markers():
    cf = "<html><head><title>Just a moment...</title></head><body>cf-challenge</body></html>"
    blocked, reason = rf.looks_blocked(200, cf)
    assert blocked is True
    assert "challenge_marker" in reason


def test_looks_blocked_passes_real_content():
    page = "<html><head><title>Salaries</title></head><body>" + "data " * 200 + "</body></html>"
    assert rf.looks_blocked(200, page) == (False, "")


def test_strip_to_text_removes_markup_and_scripts():
    html = "<html><body><script>evil()</script><h1>Hi</h1><p>Body text</p></body></html>"
    text = rf._strip_to_text(html, 1000)
    assert "evil" not in text
    assert "Hi" in text and "Body text" in text
    assert "<" not in text


def test_title_extraction():
    assert rf._title("<title> My  Page </title>") == "My Page"
    assert rf._title("<html>no title</html>") == ""


def test_cookie_cache_roundtrip(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(rf, "_COOKIE_DIR", tmp)
        rf._save_cookies("www.levels.fyi", {"cf_clearance": "abc123"})
        rf._save_cookies("www.levels.fyi", {"session": "xyz"})  # merge, not overwrite
        loaded = rf._load_cookies("www.levels.fyi")
        assert loaded == {"cf_clearance": "abc123", "session": "xyz"}
        assert rf._load_cookies("unknown.com") == {}


def test_resilient_fetch_rejects_non_http():
    out = json.loads(rf.resilient_fetch("ftp://x"))
    assert "error" in out


def test_resilient_fetch_tier1_blocked_then_no_browser_degrades(monkeypatch):
    """Tier 1 reports blocked, Tier 2 has no browser -> graceful blocked result."""
    monkeypatch.setattr(rf, "_curl_cffi_available", lambda: True)
    monkeypatch.setattr(
        rf, "_tier1_curl_cffi",
        lambda url, timeout, proxy='': {
            "status": 403, "final_url": url, "body": "Attention Required! Cloudflare",
            "blocked": True, "reason": "http_403",
        },
    )
    monkeypatch.setattr(rf, "_tier2_browser", lambda url, timeout: None)  # no browser
    out = json.loads(rf.resilient_fetch("https://www.levels.fyi/x"))
    assert out["blocked"] is True
    assert out["tiers_tried"] == ["curl_cffi", "browser_cdp"]
    assert "note" in out and "do not retry" in out["note"].lower()


def test_resilient_fetch_tier1_success_skips_browser(monkeypatch):
    monkeypatch.setattr(rf, "_curl_cffi_available", lambda: True)
    page = "<title>OK</title>" + "content " * 100
    monkeypatch.setattr(
        rf, "_tier1_curl_cffi",
        lambda url, timeout, proxy='': {
            "status": 200, "final_url": url, "body": page, "blocked": False, "reason": "",
        },
    )
    called = {"browser": False}
    def _b(url, timeout):
        called["browser"] = True
        return None
    monkeypatch.setattr(rf, "_tier2_browser", _b)
    out = json.loads(rf.resilient_fetch("https://example.com"))
    assert out["ok"] is True
    assert out["tier_used"] == "curl_cffi"
    assert called["browser"] is False  # no escalation on success
    assert out["title"] == "OK"


# ─────────────── proxy support (Tier 1) ───────────────


def test_proxy_pool_reads_hermes_env_list(monkeypatch):
    monkeypatch.setenv("HERMES_FETCH_PROXY", "http://a:1 , http://b:2,http://c:3")
    assert rf._proxy_pool() == ["http://a:1", "http://b:2", "http://c:3"]


def test_proxy_pool_falls_back_to_standard_env(monkeypatch):
    monkeypatch.delenv("HERMES_FETCH_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://corp:8080")
    assert rf._proxy_pool() == ["http://corp:8080"]


def test_resolve_proxy_explicit_wins(monkeypatch):
    monkeypatch.setenv("HERMES_FETCH_PROXY", "http://pool:1")
    assert rf._resolve_proxy("http://explicit:9") == "http://explicit:9"


def test_resolve_proxy_rotates_pool(monkeypatch):
    monkeypatch.setenv("HERMES_FETCH_PROXY", "http://a:1,http://b:2,http://c:3")
    seen = {rf._resolve_proxy(None) for _ in range(50)}
    assert seen <= {"http://a:1", "http://b:2", "http://c:3"} and len(seen) >= 2


def test_resolve_proxy_empty_when_unset(monkeypatch):
    monkeypatch.delenv("HERMES_FETCH_PROXY", raising=False)
    for e in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(e, raising=False)
    assert rf._resolve_proxy(None) == ""


def test_tier1_passes_proxy_to_curl_cffi(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        url = "https://x"
        text = "<title>ok</title>" + "z" * 600

        class cookies:
            jar = []

    def _fake_get(url, **kwargs):
        captured.update(kwargs)
        return _Resp()

    # _tier1 does `from curl_cffi import requests as creq; creq.get(...)`.
    # Patch the real attribute in place — do NOT swap sys.modules (that breaks
    # curl_cffi's package import for sibling tests).
    import curl_cffi.requests as creq_mod
    monkeypatch.setattr(creq_mod, "get", _fake_get)
    rf._tier1_curl_cffi("https://x", 10.0, proxy="http://res:7000")
    assert captured.get("proxies") == {"http": "http://res:7000", "https": "http://res:7000"}


# ─────────────── stealth injection (Tier 2) ───────────────


def test_stealth_js_covers_key_evasions():
    js = rf._STEALTH_JS
    for tell in ("webdriver", "window.chrome", "languages", "plugins",
                 "permissions", "getParameter", "hardwareConcurrency"):
        assert tell in js, f"stealth script missing {tell}"


def test_tier2_injects_stealth_before_navigate(monkeypatch):
    calls = []

    def _fake_cdp(method, params=None, target_id=None, frame_id=None, timeout=30.0, task_id=None):
        calls.append(method)
        if method == "Target.createTarget":
            return json.dumps({"result": {"targetId": "T1"}})
        if method == "Runtime.evaluate":
            return json.dumps({"result": {"result": {"value": "<title>cleared</title>" + "y" * 600}}})
        if method == "Network.getAllCookies":
            return json.dumps({"result": {"cookies": [
                {"name": "cf_clearance", "value": "tok", "domain": ".levels.fyi"}]}})
        return json.dumps({"result": {}})

    import types
    fake_mod = types.SimpleNamespace(_browser_cdp_check=lambda: True, browser_cdp=_fake_cdp)
    monkeypatch.setitem(__import__("sys").modules, "tools.browser_cdp_tool", fake_mod)

    out = rf._tier2_browser("https://www.levels.fyi/x", 12.0)
    assert out is not None and out["status"] == 200
    # Stealth must be installed before the navigate to the real URL.
    assert "Page.addScriptToEvaluateOnNewDocument" in calls
    assert calls.index("Page.addScriptToEvaluateOnNewDocument") < calls.index("Page.navigate")
    # First target is blank, real nav happens via Page.navigate.
    assert calls[0] == "Target.createTarget"
