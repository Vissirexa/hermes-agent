"""Tests for Telegram incoming reaction commands.

A user reaction on an existing message (message_reaction update) maps to a
deterministic action: a synthetic COMMAND event (e.g. 👎 → /stop) that rides
the gateway's early intercept, or a synthetic TEXT turn anchored to the
reacted message. Disabled by default; fail-closed on identity.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType


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
    adapter.handle_message = AsyncMock()
    return adapter


def _reaction_update(
    emojis=("\U0001f44e",),
    old_emojis=(),
    user_id=12345,
    is_bot=False,
    no_user=False,
    chat_type="private",
    message_id=555,
    custom=False,
):
    chat = SimpleNamespace(id=777, type=chat_type, title=None, full_name="Sid")
    user = None if no_user else SimpleNamespace(
        id=user_id, is_bot=is_bot, full_name="Sid", username="sid"
    )
    if custom:
        new = tuple(SimpleNamespace(custom_emoji_id="abc123") for _ in emojis)
    else:
        new = tuple(SimpleNamespace(emoji=e) for e in emojis)
    mr = SimpleNamespace(
        chat=chat,
        user=user,
        message_id=message_id,
        date=None,
        old_reaction=tuple(SimpleNamespace(emoji=e) for e in old_emojis),
        new_reaction=new,
    )
    return SimpleNamespace(message_reaction=mr, update_id=42)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "TELEGRAM_REACTION_COMMANDS",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


class TestImportWiring:
    def test_message_reaction_handler_available(self):
        """PTB 22.6 provides MessageReactionHandler; the module must bind it."""
        from plugins.platforms.telegram import adapter as adapter_mod

        assert adapter_mod.MessageReactionHandler is not None


class TestDisabledByDefault:
    @pytest.mark.asyncio
    async def test_mapped_emoji_noop_when_disabled(self):
        adapter = _make_adapter()
        await adapter._handle_message_reaction(_reaction_update(), None)
        adapter.handle_message.assert_not_called()


class TestCommandDispatch:
    @pytest.mark.asyncio
    async def test_default_map_thumbs_down_dispatches_stop(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_COMMANDS", "1")
        adapter = _make_adapter()
        await adapter._handle_message_reaction(_reaction_update(), None)
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/stop"
        assert event.message_type == MessageType.COMMAND
        assert event.message_id == "555"
        assert event.reply_to_message_id == "555"
        assert event.source.chat_id == "777"
        assert event.source.user_id == "12345"

    @pytest.mark.asyncio
    async def test_config_enabled_without_env(self):
        adapter = _make_adapter(extra={"reaction_commands": {"enabled": True}})
        await adapter._handle_message_reaction(_reaction_update(), None)
        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_env_false_overrides_config_enabled(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_COMMANDS", "false")
        adapter = _make_adapter(extra={"reaction_commands": {"enabled": True}})
        await adapter._handle_message_reaction(_reaction_update(), None)
        adapter.handle_message.assert_not_called()


class TestPromptDispatch:
    @pytest.mark.asyncio
    async def test_mapped_prompt_builds_anchored_text_event(self, monkeypatch):
        adapter = _make_adapter(extra={"reaction_commands": {
            "enabled": True,
            "map": {"\U0001f44d": {"action": "prompt", "prompt": "Approved — proceed."}},
        }})
        import gateway.rich_sent_store as rich_sent_store
        monkeypatch.setattr(
            rich_sent_store, "lookup", lambda chat_id, mid: "Should I proceed?"
        )
        await adapter._handle_message_reaction(
            _reaction_update(emojis=("\U0001f44d",)), None
        )
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.TEXT
        assert "Approved — proceed." in event.text
        assert "\U0001f44d" in event.text
        assert event.reply_to_message_id == "555"
        assert event.reply_to_text == "Should I proceed?"

    @pytest.mark.asyncio
    async def test_variation_selector_in_config_key_matches(self):
        # Config written as ❤️ (with U+FE0F) must match Telegram's ❤ reaction.
        adapter = _make_adapter(extra={"reaction_commands": {
            "enabled": True,
            "map": {"❤️": {"action": "prompt", "prompt": "Noted."}},
        }})
        await adapter._handle_message_reaction(
            _reaction_update(emojis=("❤",)), None
        )
        adapter.handle_message.assert_awaited_once()


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_unauthorized_user_dropped(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_COMMANDS", "1")
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "99999")
        adapter = _make_adapter()
        await adapter._handle_message_reaction(
            _reaction_update(user_id=12345), None
        )
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_authorized_user_passes_allowlist(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_COMMANDS", "1")
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "12345")
        adapter = _make_adapter()
        await adapter._handle_message_reaction(
            _reaction_update(user_id=12345), None
        )
        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_anonymous_reactor_dropped(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_COMMANDS", "1")
        adapter = _make_adapter()
        await adapter._handle_message_reaction(
            _reaction_update(no_user=True), None
        )
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_bot_reactor_dropped(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_COMMANDS", "1")
        adapter = _make_adapter()
        await adapter._handle_message_reaction(
            _reaction_update(is_bot=True), None
        )
        adapter.handle_message.assert_not_called()


class TestNoOps:
    @pytest.mark.asyncio
    async def test_reaction_removal_ignored(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_COMMANDS", "1")
        adapter = _make_adapter()
        await adapter._handle_message_reaction(
            _reaction_update(emojis=(), old_emojis=("\U0001f44e",)), None
        )
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_unmapped_emoji_ignored(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_COMMANDS", "1")
        adapter = _make_adapter()
        await adapter._handle_message_reaction(
            _reaction_update(emojis=("\U0001f525",)), None  # 🔥 unmapped
        )
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_emoji_reaction_ignored(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_COMMANDS", "1")
        adapter = _make_adapter()
        await adapter._handle_message_reaction(
            _reaction_update(custom=True), None
        )
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_without_reaction_payload_ignored(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_COMMANDS", "1")
        adapter = _make_adapter()
        await adapter._handle_message_reaction(
            SimpleNamespace(message_reaction=None, update_id=42), None
        )
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_map_entry_dropped(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REACTION_COMMANDS", "1")
        adapter = _make_adapter(extra={"reaction_commands": {
            "enabled": True,
            "map": {"\U0001f44e": {"action": "command"}},  # missing command
        }})
        await adapter._handle_message_reaction(_reaction_update(), None)
        adapter.handle_message.assert_not_called()
