"""Unit tests for the HAWC2 teardown helpers in WindGym.backend.hawc2_adapter."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from WindGym.backend import hawc2_adapter
from WindGym.backend.hawc2_adapter import delete_case_folders, safe_close_h2

pytestmark = pytest.mark.unit


def test_safe_close_h2_ignores_none_and_objects_without_h2():
    safe_close_h2(None, os.getpid())
    safe_close_h2(SimpleNamespace(), os.getpid())  # no h2 attribute -> no-op


def test_safe_close_h2_only_closes_from_owner_process():
    wt = SimpleNamespace(h2=MagicMock())
    safe_close_h2(wt, os.getpid() + 1)
    wt.h2.close.assert_not_called()
    safe_close_h2(wt, os.getpid())
    wt.h2.close.assert_called_once()


def test_safe_close_h2_swallows_teardown_errors():
    for exc in (AssertionError, OSError, EOFError):
        wt = SimpleNamespace(h2=MagicMock())
        wt.h2.close.side_effect = exc("gone")
        safe_close_h2(wt, os.getpid())  # must not raise


def test_delete_case_folders_removes_res_htc_log(monkeypatch):
    calls = []
    monkeypatch.setattr(
        hawc2_adapter.shutil,
        "rmtree",
        lambda p, ignore_errors=False: calls.append((p, ignore_errors)),
    )
    htc = SimpleNamespace(
        modelpath="/model/",
        output=SimpleNamespace(
            filename=SimpleNamespace(values=["res/case_res/turbine"])
        ),
    )
    wts = SimpleNamespace(htc_lst=[htc])
    delete_case_folders(wts)
    # Only the leading "res" is swapped; the case name keeps its own "res".
    assert calls == [
        ("/model/res/case_res", True),
        ("/model/htc/case_res", True),
        ("/model/log/case_res", True),
    ]
