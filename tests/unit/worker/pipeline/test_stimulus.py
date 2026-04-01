"""
Tests for services/worker/pipeline/stimulus.py — Phase 3.2 gap-fix coverage.

Verifies the two stimulus injection mechanisms:
  §4.E.1 — Auto-trigger (E2E testing) and Redis pub/sub (production).
  §7.4 — Calibration phase ends at stimulus onset.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from services.worker.pipeline.stimulus import (
    STIMULUS_CHANNEL,
    publish_stimulus_trigger,
    setup_auto_trigger,
    start_redis_listener,
)


class TestSetupAutoTrigger:
    """setup_auto_trigger: schedule automatic stimulus injection after delay."""

    def test_returns_none_when_delay_zero(self) -> None:
        """Auto-trigger disabled when delay_s=0."""
        orchestrator = MagicMock()
        result = setup_auto_trigger(orchestrator, delay_s=0)
        assert result is None

    def test_returns_none_when_delay_negative(self) -> None:
        """Auto-trigger disabled when delay_s < 0."""
        orchestrator = MagicMock()
        result = setup_auto_trigger(orchestrator, delay_s=-1)
        assert result is None

    def test_returns_timer_when_delay_positive(self) -> None:
        """Returns a started Timer when delay > 0."""
        orchestrator = MagicMock()
        timer = setup_auto_trigger(orchestrator, delay_s=999)
        assert isinstance(timer, threading.Timer)
        assert timer.is_alive()
        timer.cancel()

    def test_fires_record_stimulus_injection_when_calibrating(self) -> None:
        """Timer fires and calls record_stimulus_injection() when still calibrating."""
        orchestrator = MagicMock()
        orchestrator._is_calibrating = True

        # Use very short delay to trigger immediately
        timer = setup_auto_trigger(orchestrator, delay_s=0.01)
        assert timer is not None
        timer.join(timeout=2)

        orchestrator.record_stimulus_injection.assert_called_once()

    def test_skips_when_already_injected(self) -> None:
        """Timer fires but skips if stimulus was already injected."""
        orchestrator = MagicMock()
        orchestrator._is_calibrating = False

        timer = setup_auto_trigger(orchestrator, delay_s=0.01)
        assert timer is not None
        timer.join(timeout=2)

        orchestrator.record_stimulus_injection.assert_not_called()

    def test_uses_env_var_default(self) -> None:
        """Uses AUTO_STIMULUS_DELAY_S module-level default when delay_s=None."""
        orchestrator = MagicMock()
        with patch("services.worker.pipeline.stimulus.AUTO_STIMULUS_DELAY_S", 0):
            result = setup_auto_trigger(orchestrator, delay_s=None)
        assert result is None


class TestStartRedisListener:
    """start_redis_listener: background thread for Redis pub/sub triggers."""

    def test_returns_thread(self) -> None:
        """Returns a started daemon thread."""
        orchestrator = MagicMock()
        mock_redis = MagicMock()
        mock_pubsub = MagicMock()
        # Return no messages so the thread exits quickly
        mock_pubsub.listen.return_value = iter([])
        mock_redis.from_url.return_value.pubsub.return_value = mock_pubsub

        with patch.dict("sys.modules", {"redis": mock_redis}):
            thread = start_redis_listener(orchestrator)

        assert thread is not None
        assert isinstance(thread, threading.Thread)
        assert thread.daemon is True
        thread.join(timeout=2)

    def test_calls_record_on_message(self) -> None:
        """Triggers record_stimulus_injection() on receiving a pub/sub message."""
        orchestrator = MagicMock()
        orchestrator._is_calibrating = True

        mock_redis = MagicMock()
        mock_pubsub = MagicMock()
        # Simulate subscription confirmation + one actual message
        mock_pubsub.listen.return_value = iter(
            [
                {"type": "subscribe", "data": 1},
                {"type": "message", "data": b"inject"},
            ]
        )
        mock_redis.from_url.return_value.pubsub.return_value = mock_pubsub

        with patch.dict("sys.modules", {"redis": mock_redis}):
            thread = start_redis_listener(orchestrator)

        assert thread is not None
        thread.join(timeout=2)
        orchestrator.record_stimulus_injection.assert_called_once()

    def test_skips_when_not_calibrating(self) -> None:
        """Ignores trigger message if stimulus already injected."""
        orchestrator = MagicMock()
        orchestrator._is_calibrating = False

        mock_redis = MagicMock()
        mock_pubsub = MagicMock()
        mock_pubsub.listen.return_value = iter(
            [
                {"type": "message", "data": b"inject"},
            ]
        )
        mock_redis.from_url.return_value.pubsub.return_value = mock_pubsub

        with patch.dict("sys.modules", {"redis": mock_redis}):
            thread = start_redis_listener(orchestrator)

        assert thread is not None
        thread.join(timeout=2)
        orchestrator.record_stimulus_injection.assert_not_called()

    def test_subscribes_to_stimulus_channel(self) -> None:
        """Subscribes to the correct Redis channel."""
        orchestrator = MagicMock()
        mock_redis = MagicMock()
        mock_pubsub = MagicMock()
        mock_pubsub.listen.return_value = iter([])
        mock_redis.from_url.return_value.pubsub.return_value = mock_pubsub

        with patch.dict("sys.modules", {"redis": mock_redis}):
            thread = start_redis_listener(orchestrator)
            assert thread is not None
            thread.join(timeout=2)

        mock_pubsub.subscribe.assert_called_once_with(STIMULUS_CHANNEL)


class TestPublishStimulusTrigger:
    """publish_stimulus_trigger: publish trigger message to Redis channel."""

    def test_publishes_to_channel(self) -> None:
        """Publishes 'inject' message to STIMULUS_CHANNEL."""
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_client.publish.return_value = 1
        mock_redis.from_url.return_value = mock_client

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = publish_stimulus_trigger(redis_url="redis://localhost:6379/0")

        assert result is True
        mock_client.publish.assert_called_once_with(STIMULUS_CHANNEL, "inject")
        mock_client.close.assert_called_once()

    def test_returns_false_on_failure(self) -> None:
        """Returns False when Redis is unavailable."""
        mock_redis = MagicMock()
        mock_redis.from_url.side_effect = ConnectionError("refused")

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = publish_stimulus_trigger(redis_url="redis://bad:6379/0")

        assert result is False
