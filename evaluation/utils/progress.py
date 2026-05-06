"""Progress-bar helpers for terminal evaluation runs."""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: Optional[int] = None,
    unit: str = "it",
    leave: bool = False,
) -> Iterable[T]:
    """Wrap an iterable in tqdm when running in an interactive terminal."""
    disabled = (
        os.environ.get("LAYERSKIP_DISABLE_TQDM") == "1"
        or not sys.stderr.isatty()
    )

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        disable=disabled,
    )

