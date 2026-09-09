"""Shared progress reporting for data-quality evaluation."""

import logging
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter


def _format_duration(seconds: float) -> str:
    """Format a duration compactly for progress logging."""
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, seconds = divmod(int(seconds), 60)

    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"

    hours, minutes = divmod(minutes, 60)

    return f"{hours}h {minutes:02d}m"


@dataclass
class ProgressTracker:
    """Track completion progress and report timing estimates at DEBUG level."""

    total: int
    label: str
    logger: logging.Logger
    _started: float = field(default_factory=perf_counter, init=False)
    _completed: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def complete(self, item: str | None = None) -> None:
        """Record one completed unit of work and report current progress."""
        with self._lock:
            self._completed += 1
            elapsed = perf_counter() - self._started

            item_text = f" ({item})" if item is not None else ""

            if self._completed >= 3 and self._completed < self.total:
                average = elapsed / self._completed
                remaining = average * (self.total - self._completed)

                self.logger.debug(
                    "%s: completed %d/%d%s | elapsed=%s | estimated remaining=%s",
                    self.label,
                    self._completed,
                    self.total,
                    item_text,
                    _format_duration(elapsed),
                    _format_duration(remaining),
                )
            else:
                self.logger.debug(
                    "%s: completed %d/%d%s | elapsed=%s",
                    self.label,
                    self._completed,
                    self.total,
                    item_text,
                    _format_duration(elapsed),
                )
