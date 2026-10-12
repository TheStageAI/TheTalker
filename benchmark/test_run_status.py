"""The client's exit status. No server, no GPU.

A run where every request failed still writes a well-formed summary of zeros.
Exiting 0 on it lets "the server was not up yet" pass for a measurement, which
is how a reader ends up with a plausible summary.json full of zeros.
"""

from __future__ import annotations

from thetalker.client import run_status


def summary(scored: int, completed: int) -> dict:
    return {"n": scored, "completed": completed}


def test_clean_run_is_zero():
    assert run_status(summary(300, 300), 0.05) == 0


def test_all_requests_failed_is_non_zero(capsys):
    assert run_status(summary(300, 0), 0.05) == 1
    err = capsys.readouterr().err
    assert "all 300 scored requests errored" in err
    assert "summary.json" in err


def test_no_scored_requests_is_non_zero(capsys):
    assert run_status(summary(0, 0), 0.05) == 1
    assert "no scored requests" in capsys.readouterr().err


def test_error_rate_over_the_threshold_is_non_zero(capsys):
    # 30 of 300 = 10%, over the 5% default
    assert run_status(summary(300, 270), 0.05) == 1
    assert "--max-error-rate" in capsys.readouterr().err


def test_error_rate_under_the_threshold_is_zero():
    # 3 of 300 = 1%, under the 5% default
    assert run_status(summary(300, 297), 0.05) == 0


def test_threshold_is_configurable():
    assert run_status(summary(300, 270), 0.5) == 0
    assert run_status(summary(300, 299), 0.0) == 1
