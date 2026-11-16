"""Atomic file bridge between Python and MQL5 EA via MT5 Common\\Files."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

from .schemas import (
    EaCommandPayload,
    EaSignalPayload,
    EaState,
    MarketContext,
    RemoteAction,
)

logger = logging.getLogger(__name__)


class Mt5Bridge:
    """Reads MT5 snapshots and writes AI signals/commands atomically."""

    def __init__(self, bridge_dir: Path) -> None:
        self.bridge_dir = bridge_dir
        self.signal_path = self.bridge_dir / "ai_signal.json"
        self.command_path = self.bridge_dir / "ea_command.json"
        self.state_path = self.bridge_dir / "ea_state.json"
        self.context_path = self.bridge_dir / "market_context.json"

    def ensure_dir(self) -> None:
        """Create the shared Common\\Files bridge directory if missing."""
        self.bridge_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write_json(self, target: Path, data: dict) -> Path:
        self.ensure_dir()
        tmp_path = target.with_suffix(".json.tmp")
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, target)
        return target

    def _read_json(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                raw = path.read_text(encoding="utf-8-sig").strip()
                if not raw:
                    return None
                return json.loads(raw)
            except (PermissionError, OSError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.05)
        logger.warning("Failed reading %s: %s", path.name, last_exc)
        return None

    def read_ea_state(self) -> EaState | None:
        """Read the latest EA state snapshot."""
        data = self._read_json(self.state_path)
        if data is None:
            return None
        try:
            return EaState.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Invalid ea_state.json schema: %s", exc)
            return None

    def read_market_context(self) -> MarketContext | None:
        """Read the latest Exness H4/H1/M15 candle snapshot."""
        data = self._read_json(self.context_path)
        if data is None:
            return None
        try:
            return MarketContext.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Invalid market_context.json schema: %s", exc)
            return None

    def write_signal(self, signal: EaSignalPayload) -> Path:
        """Publish an AI signal payload for the MQL5 EA."""
        return self._atomic_write_json(self.signal_path, signal.model_dump())

    def send_command(self, action: RemoteAction) -> EaCommandPayload:
        """Publish an immediate control command for the MQL5 EA."""
        cmd = EaCommandPayload(
            command_id=uuid.uuid4().hex[:12],
            action=action,
            issued_at_unix=int(time.time()),
        )
        self._atomic_write_json(self.command_path, cmd.model_dump())
        return cmd
