"""Tests for the Telegram persistent quick keyboard.

Reply-keyboard buttons send their literal label text as an ordinary message;
the adapter intercepts exact label matches post-auth and dispatches them
deterministically through the shared control-surface path (command → synthetic
COMMAND event, prompt → fresh TEXT turn). Attach/remove ride /keyboard;
auto-attach fires once per chat per run when the configured layout fingerprint
changed. Disabled by default; fail-closed on identity.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType


KB_EXTRA = {
    "quick_keyboard": {
        "enabled": True,
        "buttons": [
            [
                {"label": "⏹ Stop", "action": "command", "command": "/stop"},
                {"label": "📊 Status", "action": "command", "command": "/status"},
            ],
            [
                {"label": "📝 Summarize", "action": "prompt",
                 "prompt": "Summarize the current state of this task briefly."},
            ],
        ],
    }
}


def _make_adapter(extra=None):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    if extra:
        config.extra.update(extra)
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter.platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._message_handler = None
    adapter._bot = SimpleNamespace()
    adapter.handle_message = AsyncMock()
    adapter._send_message_with_thread_fallback = AsyncMock(
        return_value=SimpleNamespace(message_id=1)
    )
    return adapter


def _msg(text, chat_type="private", user_id=12345, chat_id=777, message_id=555):
    return SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=chat_id, type=chat_type, title=None, full_name="Sid"),
        from_user=SimpleNamespace(id=user_id, is_bot=False, full_name="Sid", username="sid"),
        message_id=message_id,
        date=None,
        is_topic_message=False,
        message_thread_id=None,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "TELEGRAM_QUICK_KEYBOARD",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def _tmp_store(monkeypatch, tmp_path):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    path = str(tmp_path / "telegram_quick_keyboard.json")
    monkeypatch.setattr(
        TelegramAdapter, "_quick_keyboard_store_path", staticmethod(lambda: path)
    )
    return path


class TestConfig:
    def test_disabled_by_default(self):
        adapter = _make_adapter()
        cfg = adapter._quick_keyboard_config()
        assert cfg["enabled"] is False
        assert cfg["rows"] == []

    def test_rows_and_labels_parsed(self):
        adapter = _make_adapter(extra=KB_EXTRA)
        cfg = adapter._quick_keyboard_config()
        assert cfg["enabled"] is True
        assert [[b["label"] for b in row] for row in cfg["rows"]] == [
            ["⏹ Stop", "📊 Status"], ["📝 Summarize"],
        ]
        assert cfg["labels"]["⏹ Stop"]["command"] == "/stop"
        assert cfg["labels"]["📝 Summarize"]["action"] == "prompt"

    def test_flat_button_list_becomes_one_row_each(self):
        adapter = _make_adapter(extra={"quick_keyboard": {
            "enabled": True,
            "buttons": [
                {"label": "A", "action": "command", "command": "/a"},
                {"label": "B", "action": "command", "command": "/b"},
            ],
        }})
        cfg = adapter._quick_keyboard_config()
        assert [[b["label"] for b in row] for row in cfg["rows"]] == [["A"], ["B"]]

    def test_malformed_and_duplicate_buttons_dropped(self):
        adapter = _make_adapter(extra={"quick_keyboard": {
            "enabled": True,
            "buttons": [[
                {"label": "A", "action": "command"},            # missing command
                {"label": "", "action": "command", "command": "/x"},  # no label
                {"label": "B", "action": "command", "command": "/b"},
                {"label": "B", "action": "command", "command": "/other"},  # dup
                "not-a-dict",
            ]],
        }})
        cfg = adapter._quick_keyboard_config()
        assert list(cfg["labels"]) == ["B"]
        assert cfg["labels"]["B"]["command"] == "/b"

    def test_env_false_overrides_config_enabled(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_QUICK_KEYBOARD", "false")
        adapter = _make_adapter(extra=KB_EXTRA)
        assert adapter._quick_keyboard_config()["enabled"] is False

    def test_env_true_enables_without_config_flag(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_QUICK_KEYBOARD", "1")
        extra = {"quick_keyboard": dict(KB_EXTRA["quick_keyboard"], enabled=False)}
        adapter = _make_adapter(extra=extra)
        assert adapter._quick_keyboard_config()["enabled"] is True


class TestMarkup:
    def test_markup_is_persistent_and_resized(self):
        adapter = _make_adapter(extra=KB_EXTRA)
        markup = adapter._quick_keyboard_markup(
            adapter._quick_keyboard_config()["rows"]
        )
        assert markup.is_persistent is True
        assert markup.resize_keyboard is True
        assert [btn.text for btn in markup.keyboard[0]] == ["⏹ Stop", "📊 Status"]


class TestAttachRemove:
    @pytest.mark.asyncio
    async def test_attach_sends_reply_keyboard_and_records_fp(self, _tmp_store):
        adapter = _make_adapter(extra=KB_EXTRA)
        assert await adapter._attach_quick_keyboard("777") is True
        kwargs = adapter._send_message_with_thread_fallback.await_args.kwargs
        assert kwargs["reply_markup"].is_persistent is True
        with open(_tmp_store, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["777"] == adapter._quick_keyboard_fingerprint(
            adapter._quick_keyboard_config()["rows"]
        )

    @pytest.mark.asyncio
    async def test_remove_sends_keyboard_remove_and_clears_fp(self, _tmp_store):
        from telegram import ReplyKeyboardRemove

        adapter = _make_adapter(extra=KB_EXTRA)
        await adapter._attach_quick_keyboard("777")
        assert await adapter._remove_quick_keyboard("777") is True
        kwargs = adapter._send_message_with_thread_fallback.await_args.kwargs
        assert isinstance(kwargs["reply_markup"], ReplyKeyboardRemove)
        with open(_tmp_store, encoding="utf-8") as fh:
            assert "777" not in json.load(fh)

    @pytest.mark.asyncio
    async def test_attach_without_buttons_is_noop(self, _tmp_store):
        adapter = _make_adapter()
        assert await adapter._attach_quick_keyboard("777") is False
        adapter._send_message_with_thread_fallback.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_failure_does_not_record_fp(self, _tmp_store):
        adapter = _make_adapter(extra=KB_EXTRA)
        adapter._send_message_with_thread_fallback = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        assert await adapter._attach_quick_keyboard("777") is False
        assert adapter._quick_keyboard_stored_fp("777") is None


class TestAutoAttach:
    @pytest.mark.asyncio
    async def test_first_dm_message_attaches_when_fp_changed(self, _tmp_store):
        adapter = _make_adapter(extra=KB_EXTRA)
        await adapter._maybe_auto_attach_quick_keyboard(_msg("hi"))
        adapter._send_message_with_thread_fallback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unchanged_fp_skips_attach(self, _tmp_store):
        adapter = _make_adapter(extra=KB_EXTRA)
        await adapter._attach_quick_keyboard("777")
        adapter._send_message_with_thread_fallback.reset_mock()
        await adapter._maybe_auto_attach_quick_keyboard(_msg("hi"))
        adapter._send_message_with_thread_fallback.assert_not_called()

    @pytest.mark.asyncio
    async def test_checked_once_per_chat_per_run(self, _tmp_store):
        adapter = _make_adapter(extra=KB_EXTRA)
        # First check attaches; a config change mid-run isn't re-checked.
        await adapter._maybe_auto_attach_quick_keyboard(_msg("hi"))
        adapter._send_message_with_thread_fallback.reset_mock()
        adapter._quick_keyboard_record_fp("777", None)  # pretend fp changed
        await adapter._maybe_auto_attach_quick_keyboard(_msg("again"))
        adapter._send_message_with_thread_fallback.assert_not_called()

    @pytest.mark.asyncio
    async def test_group_chats_never_auto_attach(self, _tmp_store):
        adapter = _make_adapter(extra=KB_EXTRA)
        await adapter._maybe_auto_attach_quick_keyboard(
            _msg("hi", chat_type="supergroup")
        )
        adapter._send_message_with_thread_fallback.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_never_auto_attaches(self, _tmp_store):
        adapter = _make_adapter()
        await adapter._maybe_auto_attach_quick_keyboard(_msg("hi"))
        adapter._send_message_with_thread_fallback.assert_not_called()


class TestLabelDispatch:
    @pytest.mark.asyncio
    async def test_command_label_dispatches_command_event(self):
        adapter = _make_adapter(extra=KB_EXTRA)
        assert await adapter._maybe_dispatch_quick_keyboard_label(
            _msg("⏹ Stop"), update_id=42
        ) is True
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/stop"
        assert event.message_type == MessageType.COMMAND
        assert event.source.chat_id == "777"
        assert event.source.user_id == "12345"

    @pytest.mark.asyncio
    async def test_prompt_label_dispatches_fresh_text_turn(self):
        adapter = _make_adapter(extra=KB_EXTRA)
        assert await adapter._maybe_dispatch_quick_keyboard_label(
            _msg("📝 Summarize")
        ) is True
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.TEXT
        assert event.text == "Summarize the current state of this task briefly."
        assert event.reply_to_message_id is None  # fresh turn, no anchor

    @pytest.mark.asyncio
    async def test_surrounding_whitespace_still_matches(self):
        adapter = _make_adapter(extra=KB_EXTRA)
        assert await adapter._maybe_dispatch_quick_keyboard_label(
            _msg("  ⏹ Stop \n")
        ) is True

    @pytest.mark.asyncio
    async def test_non_label_text_passes_through(self):
        adapter = _make_adapter(extra=KB_EXTRA)
        assert await adapter._maybe_dispatch_quick_keyboard_label(
            _msg("stop the run please")
        ) is False
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_never_intercepts(self):
        extra = {"quick_keyboard": dict(KB_EXTRA["quick_keyboard"], enabled=False)}
        adapter = _make_adapter(extra=extra)
        assert await adapter._maybe_dispatch_quick_keyboard_label(
            _msg("⏹ Stop")
        ) is False
        adapter.handle_message.assert_not_called()


def _wire_text_pipeline(adapter):
    """Stub the downstream text pipeline so _handle_text_message is drivable."""
    adapter._effective_update_message = lambda update: update.message
    adapter._should_process_message = lambda msg, **kw: True
    adapter._should_observe_unmentioned_group_message = lambda msg: False
    adapter._ensure_forum_commands = AsyncMock()
    adapter._build_message_event = Mock(return_value=SimpleNamespace(text="x"))
    adapter._clean_bot_trigger_text = lambda t: t
    adapter._cache_replied_media = AsyncMock()
    adapter._apply_telegram_group_observe_attribution = lambda e: e
    adapter._enqueue_text_event = Mock()


class TestTextHandlerIntegration:
    @pytest.mark.asyncio
    async def test_label_intercepted_before_batching(self, monkeypatch, _tmp_store):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
        adapter = _make_adapter(extra=KB_EXTRA)
        _wire_text_pipeline(adapter)
        # Pre-mark the auto-attach check so it doesn't fire a keyboard send.
        adapter._quick_keyboard_auto_checked = {"777"}
        update = SimpleNamespace(message=_msg("⏹ Stop"), update_id=42)
        await adapter._handle_text_message(update, None)
        adapter.handle_message.assert_awaited_once()
        adapter._enqueue_text_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_label_dropped_before_intercept(self, monkeypatch, _tmp_store):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "99999")
        adapter = _make_adapter(extra=KB_EXTRA)
        _wire_text_pipeline(adapter)
        update = SimpleNamespace(message=_msg("⏹ Stop", user_id=12345), update_id=42)
        await adapter._handle_text_message(update, None)
        adapter.handle_message.assert_not_called()
        adapter._enqueue_text_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_text_still_batches(self, monkeypatch, _tmp_store):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
        adapter = _make_adapter(extra=KB_EXTRA)
        _wire_text_pipeline(adapter)
        adapter._quick_keyboard_auto_checked = {"777"}
        update = SimpleNamespace(message=_msg("hello there"), update_id=42)
        await adapter._handle_text_message(update, None)
        adapter.handle_message.assert_not_called()
        adapter._enqueue_text_event.assert_called_once()


class TestKeyboardCommand:
    def _wire_command_pipeline(self, adapter):
        adapter._effective_update_message = lambda update: update.message
        adapter._should_process_message = lambda msg, **kw: True
        adapter._ensure_forum_commands = AsyncMock()
        adapter._build_message_event = Mock(return_value=SimpleNamespace(text="x"))
        adapter._clean_bot_trigger_text = lambda t: t
        adapter._cache_replied_media = AsyncMock()
        adapter._apply_telegram_group_observe_attribution = lambda e: e

    @pytest.mark.asyncio
    async def test_keyboard_command_attaches_and_skips_gateway(
        self, monkeypatch, _tmp_store
    ):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
        adapter = _make_adapter(extra=KB_EXTRA)
        self._wire_command_pipeline(adapter)
        adapter._quick_keyboard_auto_checked = {"777"}
        update = SimpleNamespace(message=_msg("/keyboard"), update_id=42)
        await adapter._handle_command(update, None)
        adapter._send_message_with_thread_fallback.assert_awaited_once()
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_keyboard_off_removes(self, monkeypatch, _tmp_store):
        from telegram import ReplyKeyboardRemove

        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
        adapter = _make_adapter(extra=KB_EXTRA)
        self._wire_command_pipeline(adapter)
        adapter._quick_keyboard_auto_checked = {"777"}
        update = SimpleNamespace(message=_msg("/keyboard off"), update_id=42)
        await adapter._handle_command(update, None)
        kwargs = adapter._send_message_with_thread_fallback.await_args.kwargs
        assert isinstance(kwargs["reply_markup"], ReplyKeyboardRemove)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_botname_suffix_recognized(self, monkeypatch, _tmp_store):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
        adapter = _make_adapter(extra=KB_EXTRA)
        self._wire_command_pipeline(adapter)
        adapter._quick_keyboard_auto_checked = {"777"}
        update = SimpleNamespace(message=_msg("/keyboard@HermesBot"), update_id=42)
        await adapter._handle_command(update, None)
        adapter._send_message_with_thread_fallback.assert_awaited_once()
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_other_commands_still_reach_gateway(self, monkeypatch, _tmp_store):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
        adapter = _make_adapter(extra=KB_EXTRA)
        self._wire_command_pipeline(adapter)
        adapter._quick_keyboard_auto_checked = {"777"}
        update = SimpleNamespace(message=_msg("/status"), update_id=42)
        await adapter._handle_command(update, None)
        adapter.handle_message.assert_awaited_once()
