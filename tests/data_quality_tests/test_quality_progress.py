"""Tests for data-quality progress reporting."""

import logging

from tclean.data_quality._progress import ProgressTracker, _format_duration


def test_format_duration_seconds():
    """Format check for seconds."""
    assert _format_duration(12.34) == "12.3s"


def test_format_duration_minutes():
    """Format check for minutes."""
    assert _format_duration(125) == "2m 05s"


def test_format_duration_hours():
    """Format check for hours."""
    assert _format_duration(3725) == "1h 02m"


def test_progress_tracker_reports_completion(caplog):
    """Report completed work at DEBUG level."""
    logger = logging.getLogger("tclean.data_quality.test")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        progress = ProgressTracker(
            total=2, label="unusual_level [contextual_level]", logger=logger
        )
        progress.complete("ALB")
        progress.complete("AUT")

    messages = [record.getMessage() for record in caplog.records]

    assert any("completed 1/2 (ALB)" in message for message in messages)
    assert any("completed 2/2 (AUT)" in message for message in messages)
