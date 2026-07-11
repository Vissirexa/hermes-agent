"""Tests for approvals.cron_mode — configurable approval behavior for cron jobs."""

import contextvars
import os

import pytest

import tools.approval as approval_module
from tools.approval import (
    _get_cron_approval_mode,
    _is_cron_session,
    _is_gateway_approval_context,
    check_all_command_guards,
    check_dangerous_command,
    check_execute_code_guard,
    detect_dangerous_command,
)
from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    clear_session_vars,
    reset_session_vars,
    set_session_vars,
)


@pytest.fixture(autouse=True)
def _clear_approval_state():
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("test-session")
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("test-session")


# ---------------------------------------------------------------------------
# _get_cron_approval_mode() config parsing
# ---------------------------------------------------------------------------

class TestCronApprovalModeParsing:
    def test_default_is_deny(self):
        """When no config is set, cron_mode defaults to 'deny'."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {}}):
            assert _get_cron_approval_mode() == "deny"

    def test_explicit_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"cron_mode": "deny"}}):
            assert _get_cron_approval_mode() == "deny"

    def test_explicit_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"cron_mode": "approve"}}):
            assert _get_cron_approval_mode() == "approve"

    def test_off_maps_to_approve(self):
        """'off' is an alias for 'approve' (matches --yolo semantics)."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"cron_mode": "off"}}):
            assert _get_cron_approval_mode() == "approve"

    def test_allow_maps_to_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"cron_mode": "allow"}}):
            assert _get_cron_approval_mode() == "approve"

    def test_yes_maps_to_approve(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"cron_mode": "yes"}}):
            assert _get_cron_approval_mode() == "approve"

    def test_case_insensitive(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"cron_mode": "APPROVE"}}):
            assert _get_cron_approval_mode() == "approve"

    def test_unknown_value_defaults_to_deny(self):
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"cron_mode": "maybe"}}):
            assert _get_cron_approval_mode() == "deny"

    def test_config_load_failure_defaults_to_deny(self):
        """If config loading fails entirely, default to deny (safe)."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", side_effect=RuntimeError("config broken")):
            assert _get_cron_approval_mode() == "deny"

    def test_yaml_boolean_false_maps_to_deny(self):
        """YAML 1.1 parses bare 'off' as False. Ensure it maps to deny."""
        from unittest.mock import patch as mock_patch
        with mock_patch("hermes_cli.config.load_config", return_value={"approvals": {"cron_mode": False}}):
            # str(False) = "False", which is not in the approve set, so deny
            assert _get_cron_approval_mode() == "deny"


# ---------------------------------------------------------------------------
# check_dangerous_command() with cron session
# ---------------------------------------------------------------------------

class TestCronDenyMode:
    """When HERMES_CRON_SESSION is set and cron_mode=deny, dangerous commands are blocked."""

    def test_dangerous_command_blocked_in_cron_deny_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]
            assert "cron_mode" in result["message"]

    def test_safe_command_allowed_in_cron_deny_mode(self, monkeypatch):
        """Non-dangerous commands still work even with cron_mode=deny."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_dangerous_command("ls -la", "local")
            assert result["approved"]

    def test_multiple_dangerous_patterns_blocked(self, monkeypatch):
        """All dangerous patterns are blocked, not just rm."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        dangerous_commands = [
            "rm -rf /",
            "chmod 777 /etc/passwd",
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/sda",
        ]

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            for cmd in dangerous_commands:
                is_dangerous, _, _ = detect_dangerous_command(cmd)
                if is_dangerous:
                    result = check_dangerous_command(cmd, "local")
                    assert not result["approved"], f"Should be blocked: {cmd}"
                    assert "BLOCKED" in result["message"]

    def test_block_message_includes_description(self, monkeypatch):
        """The block message should mention what pattern was matched."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            # Should contain the description of what was flagged
            assert "dangerous" in result["message"].lower() or "delete" in result["message"].lower()


class TestCronApproveMode:
    """When HERMES_CRON_SESSION is set and cron_mode=approve, dangerous commands pass through."""

    def test_dangerous_command_allowed_in_cron_approve_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="approve"):
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert result["approved"]


# ---------------------------------------------------------------------------
# check_all_command_guards() with cron session
# ---------------------------------------------------------------------------

class TestCronDenyModeAllGuards:
    """The combined guard function also respects cron_mode."""

    def test_dangerous_command_blocked_in_combined_guard(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]

    def test_safe_command_allowed_in_combined_guard(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_all_command_guards("echo hello", "local")
            assert result["approved"]

    def test_combined_guard_approve_mode(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="approve"):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
            assert result["approved"]

    def test_tirith_content_threat_blocked_in_cron_deny(self, monkeypatch):
        """Content-level threats caught only by tirith (not the regex patterns)
        are blocked in cron-deny mode. Regression for #22070: previously the
        cron-deny early return ran only detect_dangerous_command and returned
        before reaching the tirith check, so these were silently approved."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        # A tirith "block" result while detect_dangerous_command reports safe:
        # proves the block comes from the tirith path, not the regex path.
        fake_tirith = {
            "action": "block",
            "findings": [{"severity": "HIGH", "title": "Homograph URL",
                          "description": "URL contains Cyrillic lookalike chars"}],
            "summary": "homograph url",
        }
        with (
            mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"),
            mock_patch("tools.approval.detect_dangerous_command",
                       return_value=(False, None, None)),
            mock_patch("tools.tirith_security.check_command_security",
                       return_value=fake_tirith),
        ):
            result = check_all_command_guards("curl http://xn--e1afmkfd.example/x", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]

    def test_tirith_import_error_fail_closed_blocks_in_cron_deny(self, monkeypatch):
        """When tirith is unavailable and security.tirith_fail_open is false,
        cron-deny mode blocks rather than silently allowing (a cron session has
        no user to approve). Mirrors the fail-closed handling in the main flow."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        import builtins
        _real_import = builtins.__import__

        def _blocked_import(name, *a, **k):
            if name.endswith("tirith_security"):
                raise ImportError("simulated missing tirith")
            return _real_import(name, *a, **k)

        with (
            mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"),
            mock_patch("tools.approval.detect_dangerous_command",
                       return_value=(False, None, None)),
            mock_patch("hermes_cli.config.load_config",
                       return_value={"security": {"tirith_enabled": True,
                                                   "tirith_fail_open": False}}),
            mock_patch.object(builtins, "__import__", _blocked_import),
        ):
            result = check_all_command_guards("echo hi", "local")
            assert not result["approved"]
            assert "tirith_fail_open" in result["message"]

    def test_tirith_import_error_fail_open_allows_in_cron_deny(self, monkeypatch):
        """When tirith is unavailable and tirith_fail_open is true (default),
        cron-deny mode allows safe commands — preserving pre-#22070 behavior."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        import builtins
        _real_import = builtins.__import__

        def _blocked_import(name, *a, **k):
            if name.endswith("tirith_security"):
                raise ImportError("simulated missing tirith")
            return _real_import(name, *a, **k)

        with (
            mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"),
            mock_patch("tools.approval.detect_dangerous_command",
                       return_value=(False, None, None)),
            mock_patch("hermes_cli.config.load_config",
                       return_value={"security": {"tirith_enabled": True,
                                                   "tirith_fail_open": True}}),
            mock_patch.object(builtins, "__import__", _blocked_import),
        ):
            result = check_all_command_guards("echo hi", "local")
            assert result["approved"]


# ---------------------------------------------------------------------------
# Edge cases: cron mode interaction with other approval mechanisms
# ---------------------------------------------------------------------------

class TestCronModeInteractions:
    """Cron mode should NOT interfere with other approval bypass mechanisms."""

    def test_container_env_still_auto_approves(self, monkeypatch):
        """Docker/sandbox environments bypass approvals regardless of cron_mode."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_dangerous_command("rm -rf /", "docker")
            assert result["approved"]

    def test_yolo_overrides_cron_deny(self, monkeypatch):
        """--yolo still bypasses cron_mode=deny for dangerous (non-hardline) commands."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("HERMES_YOLO_MODE", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)

        # _YOLO_MODE_FROZEN is frozen at module import time (security: prevents
        # prompt injection from runtime-setting HERMES_YOLO_MODE). When the
        # test process imports tools.approval BEFORE this test sets the env,
        # the frozen value is False and yolo-bypass paths don't activate.
        # Patch the module attribute directly to simulate process-startup
        # with HERMES_YOLO_MODE=1.
        from unittest.mock import patch as mock_patch
        import tools.approval
        with (
            mock_patch.object(tools.approval, "_YOLO_MODE_FROZEN", True),
            mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"),
        ):
            # Use a dangerous-but-not-hardline command — `rm -rf /` is now
            # hardline-blocked regardless of yolo (see test_hardline_blocklist.py).
            result = check_dangerous_command("rm -rf /tmp/stuff", "local")
            assert result["approved"]

    def test_non_cron_non_interactive_still_auto_approves(self, monkeypatch):
        """Non-cron, non-interactive sessions (e.g. scripted usage) still auto-approve."""
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        result = check_dangerous_command("rm -rf /tmp/stuff", "local")
        assert result["approved"]


class TestCronWithGatewayOrigin:
    """Cron jobs originating from a gateway platform must NOT be treated as gateway.

    cron/scheduler.py binds HERMES_SESSION_PLATFORM via contextvars for
    delivery routing (so cron output lands back in the origin chat). The
    API-server approvals work (PR #20311) made check_dangerous_command treat
    any contextvar-bound platform as a gateway session. That would route
    cron-from-telegram/discord/etc. through submit_pending with no listener,
    hanging the job instead of respecting approvals.cron_mode.
    """

    def test_cron_with_telegram_origin_uses_cron_mode_not_gateway(self, monkeypatch):
        """Cron + contextvar platform=telegram + cron_mode=deny → BLOCKED, not pending."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)

        from gateway.session_context import set_session_vars, clear_session_vars
        tokens = set_session_vars(platform="telegram", chat_id="123")
        try:
            from unittest.mock import patch as mock_patch
            with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
                result = check_dangerous_command("rm -rf /tmp/stuff", "local")
                # Cron-mode path: BLOCKED message, NOT pending/approval_required.
                assert not result["approved"]
                assert "BLOCKED" in result["message"]
                assert "cron_mode" in result["message"]
                assert result.get("status") != "approval_required"
        finally:
            clear_session_vars(tokens)

    def test_cron_with_telegram_origin_approve_mode_allows(self, monkeypatch):
        """Cron + contextvar platform=telegram + cron_mode=approve → allowed via cron path."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)

        from gateway.session_context import set_session_vars, clear_session_vars
        tokens = set_session_vars(platform="discord", chat_id="456")
        try:
            from unittest.mock import patch as mock_patch
            with mock_patch("tools.approval._get_cron_approval_mode", return_value="approve"):
                result = check_dangerous_command("rm -rf /tmp/stuff", "local")
                assert result["approved"]
                # Should NOT be a gateway-approval response.
                assert result.get("status") != "approval_required"
        finally:
            clear_session_vars(tokens)

    def test_cron_with_telegram_origin_combined_guard_uses_cron_mode(self, monkeypatch):
        """check_all_command_guards must also honor cron_mode over gateway classification."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)

        from gateway.session_context import set_session_vars, clear_session_vars
        tokens = set_session_vars(platform="telegram", chat_id="789")
        try:
            from unittest.mock import patch as mock_patch
            with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
                result = check_all_command_guards("rm -rf /tmp/stuff", "local")
                assert not result["approved"]
                assert "BLOCKED" in result["message"]
                assert result.get("status") != "approval_required"
        finally:
            clear_session_vars(tokens)


# ---------------------------------------------------------------------------
# HERMES_CRON_SESSION: ContextVar (task-local) vs. the old os.environ leak
# ---------------------------------------------------------------------------
#
# Regression coverage for the fix: run_job() (cron/scheduler.py) used to set
# os.environ["HERMES_CRON_SESSION"] = "1" process-wide and never clear it.
# Because the in-process cron scheduler shares its process with the live
# gateway (gateway/run.py::_start_cron_ticker -> InProcessCronScheduler),
# every interactive session (e.g. Telegram) after the FIRST cron tick was
# permanently misclassified as a cron session — approvals took the
# ``approvals.cron_mode`` branch (hard-deny, or silent auto-approve) instead
# of the interactive gateway flow.
#
# The fix: gateway/session_context._CRON_SESSION is a task-local ContextVar,
# set per-job by run_job() via ``_VAR_MAP["HERMES_CRON_SESSION"].set("1")``
# instead of mutating os.environ. tools.approval._is_cron_session() reads it
# through get_session_env (contextvar-first, os.environ fallback preserved
# for dedicated cron worker processes / legacy tests that never touch the
# contextvar at all).

class TestCronSessionContextVar:
    """Task-local HERMES_CRON_SESSION contextvar replaces the process-wide env leak."""

    @pytest.fixture(autouse=True)
    def _reset_cron_session_contextvar(self, monkeypatch):
        # ContextVar.set() calls made directly in a test function (i.e. NOT
        # inside a contextvars.copy_context().run(...) call) mutate the
        # thread's ambient context and persist into whatever test runs next
        # in the same thread. Reset to the "never set" sentinel before and
        # after every test in this class, and make sure the legacy env var
        # doesn't leak in either direction.
        _VAR_MAP["HERMES_CRON_SESSION"].set(_UNSET)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        yield
        _VAR_MAP["HERMES_CRON_SESSION"].set(_UNSET)

    def test_contextvar_set_makes_is_cron_session_true(self):
        """What run_job() now does: bind the contextvar, task-local, no env write."""
        _VAR_MAP["HERMES_CRON_SESSION"].set("1")
        assert _is_cron_session() is True
        # No process-global mutation accompanies the contextvar bind.
        assert "HERMES_CRON_SESSION" not in os.environ

    def test_contextvar_set_drives_cron_deny_branch_for_dangerous_command(self):
        _VAR_MAP["HERMES_CRON_SESSION"].set("1")

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_all_command_guards("rm -rf /tmp/stuff", "local")
            assert not result["approved"]
            assert "BLOCKED" in result["message"]
            assert "cron_mode" in result["message"]

    def test_contextvar_set_drives_cron_deny_branch_for_execute_code(self):
        """check_execute_code_guard's cron branch also reads the contextvar
        (the fourth of the four former env_var_enabled call sites)."""
        _VAR_MAP["HERMES_CRON_SESSION"].set("1")

        from unittest.mock import patch as mock_patch
        with mock_patch("tools.approval._get_cron_approval_mode", return_value="deny"):
            result = check_execute_code_guard(
                "import os\nos.system('rm -rf /tmp/stuff')", "local"
            )
            assert not result["approved"]
            assert "BLOCKED" in result["message"]
            assert result.get("pattern_key") == "execute_code"

    def test_env_fallback_preserved_when_contextvar_never_set(self, monkeypatch):
        """Dedicated cron worker processes / legacy tests that never touch the
        contextvar at all still classify via the os.environ fallback."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        assert _is_cron_session() is True

    def test_fallback_reads_process_env_without_recursion(self, monkeypatch):
        """If the session_context read fails, _is_cron_session() must consult
        os.environ directly. An earlier version of the helper recursed into
        itself in the except block, turning any import/read failure into a
        RecursionError instead of the documented legacy fallback."""
        from unittest.mock import patch as mock_patch

        with mock_patch(
            "gateway.session_context.get_session_env",
            side_effect=RuntimeError("session context unavailable"),
        ):
            assert _is_cron_session() is False
            monkeypatch.setenv("HERMES_CRON_SESSION", "1")
            assert _is_cron_session() is True

    def test_reset_session_vars_returns_to_unset_then_env_fallback_reapplies(self, monkeypatch):
        _VAR_MAP["HERMES_CRON_SESSION"].set("1")
        assert _is_cron_session() is True

        reset_session_vars()
        assert _VAR_MAP["HERMES_CRON_SESSION"].get() is _UNSET

        # Contextvar back to "never set in this context" -> os.environ fallback.
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        assert _is_cron_session() is True
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        assert _is_cron_session() is False

    def test_cron_context_does_not_leak_into_fresh_interactive_context(self):
        """THE CORE REGRESSION.

        Before the fix, a cron tick's os.environ["HERMES_CRON_SESSION"] = "1"
        was process-wide and permanent, so every interactive session running
        afterwards in the SAME PROCESS was misclassified as cron. Simulate the
        cron job the way run_job() actually executes — in its own task, i.e.
        a copied contextvars.Context — and prove:

          1. os.environ is never written to (the leak vector is gone).
          2. A separately-copied context that binds a real interactive gateway
             session (set_session_vars(platform="telegram", ...)) sees NO
             cron classification and IS treated as a gateway approval context.
        """
        assert "HERMES_CRON_SESSION" not in os.environ

        def _cron_job_body():
            # Task-local bind, exactly like run_job() in cron/scheduler.py.
            _VAR_MAP["HERMES_CRON_SESSION"].set("1")
            assert _is_cron_session() is True

        cron_ctx = contextvars.copy_context()
        cron_ctx.run(_cron_job_body)

        # The cron job's bind must not have escaped into os.environ or into
        # this (the spawning) context.
        assert "HERMES_CRON_SESSION" not in os.environ
        assert _VAR_MAP["HERMES_CRON_SESSION"].get() is _UNSET

        def _interactive_body():
            tokens = set_session_vars(platform="telegram", chat_id="123", user_id="u1")
            try:
                assert _is_cron_session() is False
                assert _is_gateway_approval_context() is True
            finally:
                clear_session_vars(tokens)

        interactive_ctx = contextvars.copy_context()
        interactive_ctx.run(_interactive_body)


class TestRunJobCronMarkerLifecycle:
    """Real run_job() lifecycle for the HERMES_CRON_SESSION marker.

    The marker must be bound for the duration of the job (so cron_mode
    applies inside it) and token-reset in run_job's finally: leaving it set
    would misclassify later work on the scheduler thread's ambient context,
    while resetting via ``.set("")`` would be just as wrong — an explicit
    empty value is authoritative for get_session_env() and would suppress
    the os.environ fallback that dedicated cron worker processes rely on.
    Mock harness mirrors tests/cron/test_cron_provider_pin.py.
    """

    @pytest.fixture(autouse=True)
    def _clean_marker_state(self, monkeypatch):
        _VAR_MAP["HERMES_CRON_SESSION"].set(_UNSET)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        yield
        _VAR_MAP["HERMES_CRON_SESSION"].set(_UNSET)

    def test_marker_bound_during_job_and_token_reset_after(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch as mock_patch

        from cron.scheduler import run_job

        seen = {}

        def _record_marker(*args, **kwargs):
            # Runs inside run_job, standing in for the agent turn: the job
            # itself must classify as cron so cron_mode applies to it.
            seen["is_cron"] = _is_cron_session()
            seen["env_written"] = "HERMES_CRON_SESSION" in os.environ
            return {"final_response": "ok"}

        job = {
            "id": "marker-lifecycle-test",
            "name": "marker lifecycle test",
            "prompt": "hello",
            "model": "test-model",
            "provider": None,
            "provider_snapshot": "openrouter",
            "base_url": None,
        }

        def _job_body():
            # run_job also clear_session_vars()-clears the HERMES_SESSION_*
            # contextvars to explicit "" on exit (documented semantics, not
            # under test here), so drive it in a copied Context — like the
            # scheduler's own thread — to keep this test's ambient context
            # clean for whatever test runs next.
            success, _output, _final, error = run_job(job)
            seen["success"] = success
            seen["error"] = error
            # Still inside the job's context, after run_job returned: the
            # marker must be token-reset back to the "never set" sentinel —
            # not left at "1", and not overwritten with an authoritative "".
            seen["marker_after"] = _VAR_MAP["HERMES_CRON_SESSION"].get()

        fake_db = MagicMock()
        with mock_patch("cron.scheduler._hermes_home", tmp_path), \
             mock_patch("cron.scheduler._resolve_origin", return_value=None), \
             mock_patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             mock_patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             mock_patch("hermes_state.SessionDB", return_value=fake_db), \
             mock_patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "test-key",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             mock_patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.side_effect = _record_marker
            mock_agent_cls.return_value = mock_agent

            contextvars.copy_context().run(_job_body)

        assert seen["success"] is True
        assert seen["error"] is None

        # Inside the job: classified as cron, without touching os.environ.
        assert seen["is_cron"] is True
        assert seen["env_written"] is False

        # After completion, in the job's own context: reset to _UNSET.
        assert seen["marker_after"] is _UNSET
        assert "HERMES_CRON_SESSION" not in os.environ

        # And in the spawning context the os.environ fallback still works —
        # a ".set(\"\")"-style reset inside run_job's context could never
        # shadow it out here, and the marker itself never escaped.
        assert _VAR_MAP["HERMES_CRON_SESSION"].get() is _UNSET
        assert _is_cron_session() is False
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        assert _is_cron_session() is True
