"""Tests for the host-memory training guard."""

import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for callback tests")

from birdnet_stm32.cli import train


def test_memory_guard_checks_at_configured_interval(monkeypatch):
    monkeypatch.setattr(train, "_read_meminfo_gb", lambda: (64.0, 10.0))
    guard = train.HostMemoryGuard(check_every=2)

    guard.on_train_batch_end(0)
    with pytest.raises(MemoryError, match="last completed-epoch checkpoint"):
        guard.on_train_batch_end(1)


def test_memory_guard_scales_reserve_for_smaller_hosts(monkeypatch):
    monkeypatch.setattr(train, "_read_meminfo_gb", lambda: (16.0, 3.5))
    guard = train.HostMemoryGuard(check_every=1)

    guard.on_train_batch_end(0)


def test_memory_guard_reports_epoch_memory(monkeypatch, capsys):
    monkeypatch.setattr(train, "_read_meminfo_gb", lambda: (64.0, 40.0))
    guard = train.HostMemoryGuard()

    guard.on_epoch_end(2)

    assert "[memory] epoch=3 available=40.0 GiB reserve=12.0 GiB" in capsys.readouterr().out
