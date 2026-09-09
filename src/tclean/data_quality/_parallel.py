"""Internal parallel-execution helpers for data-quality methods."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor


def ordered_thread_map[InputT, OutputT](
    function: Callable[[InputT], OutputT], items: Sequence[InputT], *, threads: int
) -> list[OutputT]:
    """Evaluate independent items while preserving supplied order."""
    if threads == 1 or len(items) <= 1:
        return [function(item) for item in items]

    max_threads = min(threads, len(items))

    with ThreadPoolExecutor(
        max_workers=max_threads, thread_name_prefix="tclean-data-quality"
    ) as executor:
        return list(executor.map(function, items))
