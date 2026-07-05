"""Tests for Telegram response quick-action buttons.

An inline button row is attached to the final response message *after*
delivery (on_processing_complete → edit_message_reply_markup), so it can't
race streaming edits. qa:<key>:<message_id> callbacks dispatch through the
shared control-surface path: command → synthetic COMMAND event, prompt →
TEXT turn anchored to the tapped message. Disabled by default; fail-closed
on identity; stale keys toast without dispatch.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType, ProcessingOutcome


QA_EXTRA = {
    "quick_actions": {
        "enabled": True,
        "buttons": [
            {"key": "retry", "label": "🔁 Retry", "action": "prompt",
             "prompt": "That wasn't right — try again with a different approach."},
            {"key": "stop", "label": "⏹ Stop", "action": "command", "command": "/stop"},
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
    adapter._bot = SimpleNamespace(edit_message_reply_markup=AsyncMock())
    adapter.handle_message = AsyncMock()
    return adapter


def _event(chat_id="777", message_type=MessageType.TEXT):
    return SimpleNamespace(
        source=SimpleNamespace(chat_id=chat_id),
        message_type=message_type,
        message_id="111",
    )


def _query(data="qa:stop:888", user_id=12345, with_message=True):
    message = None
    if with_message:
        message = SimpleNamespace(
            chat=SimpleNamespace(id=777, type="private", title=None, full_name="Sid"),
            chat_id=777,
            message_thread_id=None,
            date=None,
        )
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(
            id=user_id, is_bot=False, first_name="Sid", full_name="Sid"
        ),
        message=message,
        answer=AsyncMock(),
    )


async def _dispatch(adapter, query):
    await adapter._handle_quick_action_callback(
        query,
        query.data,
        query_chat_id=getattr(query.message, "chat_id", None),
        query_chat_type="private",
        query_thread_id=None,
        query_user_name="Sid",
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "TELEGRAM_QUICK_ACTIONS",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")


@pytest.fixture(autouse=True)
def _real_inline_keyboard(monkeypatch):
    """Inspectable inline-keyboard classes (the conftest telegram mock keeps
    InlineKeyboard* as MagicMock for older tests' call-behavior asserts)."""
    from plugins.platforms.telegram import adapter as adapter_mod

    class _Btn:
        def __init__(self, text, callback_data=None, **kwargs):
            self.text = text
            self.callback_data = callback_data

    class _Markup:
        def __init__(self, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    monkeypatch.setattr(adapter_mod, "InlineKeyboardButton", _Btn, raising=False)
    monkeypatch.setattr(adapter_mod, "InlineKeyboardMarkup", _Markup, raising=False)


class TestConfig:
    def test_disabled_by_default(self, monkeypatch):
        adapter = _make_adapter()
        cfg = adapter._quick_actions_config()
        assert cfg["enabled"] is False
        assert cfg["buttons"] == []

    def test_buttons_and_keys_parsed(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        cfg = adapter._quick_actions_config()
        assert cfg["enabled"] is True
        assert [b["key"] for b in cfg["buttons"]] == ["retry", "stop"]
        assert cfg["keys"]["stop"]["command"] == "/stop"

    def test_invalid_keys_and_duplicates_dropped(self):
        adapter = _make_adapter(extra={"quick_actions": {
            "enabled": True,
            "buttons": [
                {"key": "OK!", "label": "A", "action": "command", "command": "/a"},
                {"key": "x" * 40, "label": "B", "action": "command", "command": "/b"},
                {"key": "", "label": "C", "action": "command", "command": "/c"},
                {"key": "good", "label": "", "action": "command", "command": "/d"},
                {"key": "good", "label": "E", "action": "command", "command": "/e"},
                {"key": "good", "label": "F", "action": "command", "command": "/f"},
            ],
        }})
        cfg = adapter._quick_actions_config()
        assert list(cfg["keys"]) == ["good"]
        assert cfg["keys"]["good"]["command"] == "/e"

    def test_env_false_overrides_config_enabled(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_QUICK_ACTIONS", "0")
        adapter = _make_adapter(extra=QA_EXTRA)
        assert adapter._quick_actions_config()["enabled"] is False


class TestAttach:
    @pytest.mark.asyncio
    async def test_attaches_to_tracked_final_message(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        adapter._track_outbound_for_quick_actions("777", 999)
        await adapter._maybe_attach_quick_actions(
            _event(), ProcessingOutcome.SUCCESS
        )
        adapter._bot.edit_message_reply_markup.assert_awaited_once()
        kwargs = adapter._bot.edit_message_reply_markup.await_args.kwargs
        assert kwargs["message_id"] == 999
        row = kwargs["reply_markup"].inline_keyboard[0]
        assert [b.callback_data for b in row] == ["qa:retry:999", "qa:stop:999"]

    @pytest.mark.asyncio
    async def test_tracked_id_is_one_shot(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        adapter._track_outbound_for_quick_actions("777", 999)
        await adapter._maybe_attach_quick_actions(_event(), ProcessingOutcome.SUCCESS)
        adapter._bot.edit_message_reply_markup.reset_mock()
        await adapter._maybe_attach_quick_actions(_event(), ProcessingOutcome.SUCCESS)
        adapter._bot.edit_message_reply_markup.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_attach_on_failure_or_cancel(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        adapter._track_outbound_for_quick_actions("777", 999)
        await adapter._maybe_attach_quick_actions(_event(), ProcessingOutcome.FAILURE)
        await adapter._maybe_attach_quick_actions(_event(), ProcessingOutcome.CANCELLED)
        adapter._bot.edit_message_reply_markup.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_attach_for_command_events(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        adapter._track_outbound_for_quick_actions("777", 999)
        await adapter._maybe_attach_quick_actions(
            _event(message_type=MessageType.COMMAND), ProcessingOutcome.SUCCESS
        )
        adapter._bot.edit_message_reply_markup.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_no_attach(self):
        adapter = _make_adapter()
        adapter._track_outbound_for_quick_actions("777", 999)
        await adapter._maybe_attach_quick_actions(_event(), ProcessingOutcome.SUCCESS)
        adapter._bot.edit_message_reply_markup.assert_not_called()

    @pytest.mark.asyncio
    async def test_edit_failure_is_swallowed(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        adapter._bot.edit_message_reply_markup = AsyncMock(
            side_effect=RuntimeError("message to edit not found")
        )
        adapter._track_outbound_for_quick_actions("777", 999)
        # Must not raise — a failed attach can't disturb the completed turn.
        await adapter._maybe_attach_quick_actions(_event(), ProcessingOutcome.SUCCESS)

    @pytest.mark.asyncio
    async def test_on_processing_complete_attaches_without_reactions(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        adapter._reactions_enabled = lambda: False
        adapter._track_outbound_for_quick_actions("777", 999)
        await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
        adapter._bot.edit_message_reply_markup.assert_awaited_once()


class TestTracking:
    def test_last_write_wins_per_chat(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        adapter._track_outbound_for_quick_actions("777", 1)
        adapter._track_outbound_for_quick_actions("777", 2)
        adapter._track_outbound_for_quick_actions("888", 3)
        assert adapter._qa_last_outbound == {"777": "2", "888": "3"}

    def test_missing_ids_ignored(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        adapter._track_outbound_for_quick_actions(None, 1)
        adapter._track_outbound_for_quick_actions("777", None)
        assert getattr(adapter, "_qa_last_outbound", None) in (None, {})


class TestCallbackDispatch:
    @pytest.mark.asyncio
    async def test_command_button_dispatches_command_event(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        query = _query("qa:stop:888")
        await _dispatch(adapter, query)
        query.answer.assert_awaited()  # spinner always stopped
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/stop"
        assert event.message_type == MessageType.COMMAND
        assert event.source.chat_id == "777"
        assert event.source.user_id == "12345"

    @pytest.mark.asyncio
    async def test_prompt_button_dispatches_anchored_text_turn(self, monkeypatch):
        import gateway.rich_sent_store as rich_sent_store

        monkeypatch.setattr(
            rich_sent_store, "lookup", lambda chat_id, mid: "The final answer."
        )
        adapter = _make_adapter(extra=QA_EXTRA)
        query = _query("qa:retry:888")
        await _dispatch(adapter, query)
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.TEXT
        assert "try again with a different approach" in event.text
        assert "🔁 Retry" in event.text and "888" in event.text
        assert event.reply_to_message_id == "888"
        assert event.reply_to_text == "The final answer."

    @pytest.mark.asyncio
    async def test_stale_key_toasts_without_dispatch(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        query = _query("qa:gone:888")
        await _dispatch(adapter, query)
        assert "no longer configured" in query.answer.await_args.kwargs["text"]
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_feature_toasts_without_dispatch(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_QUICK_ACTIONS", "false")
        adapter = _make_adapter(extra=QA_EXTRA)
        query = _query("qa:stop:888")
        await _dispatch(adapter, query)
        assert "no longer configured" in query.answer.await_args.kwargs["text"]
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_user_denied(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        query = _query("qa:stop:888", user_id=666)
        await _dispatch(adapter, query)
        assert "not authorized" in query.answer.await_args.kwargs["text"]
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_data_answered_without_dispatch(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        for data in ("qa:", "qa:stop", "qa::888", "qa:stop:"):
            query = _query(data)
            await _dispatch(adapter, query)
            query.answer.assert_awaited()
            adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_inaccessible_message_toasts_without_dispatch(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        query = _query("qa:stop:888", with_message=False)
        await _dispatch(adapter, query)
        assert "too old" in query.answer.await_args.kwargs["text"]
        adapter.handle_message.assert_not_called()


class TestCallbackRouting:
    @pytest.mark.asyncio
    async def test_qa_prefix_routes_to_quick_action_handler(self):
        adapter = _make_adapter(extra=QA_EXTRA)
        adapter._handle_quick_action_callback = AsyncMock()
        update = SimpleNamespace(callback_query=_query("qa:stop:888"))
        await adapter._handle_callback_query(update, None)
        adapter._handle_quick_action_callback.assert_awaited_once()
