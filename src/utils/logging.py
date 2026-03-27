"""Logging utilities for VisioLex."""

import logging
import sys
import time
from collections import deque
from typing import Optional


def get_logger(name: str = "visiolex", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class AverageMeter:
    """Track a running mean (+ optional recent window) for a scalar metric."""

    def __init__(self, name: str = "", window: int = 100) -> None:
        self.name = name
        self._window = window
        self._recent: deque = deque(maxlen=window)
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self._recent.clear()

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self._recent.append(val)

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    @property
    def recent_avg(self) -> float:
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f} (recent {self.recent_avg:.4f})"


class Timer:
    """Simple wall-clock timer."""

    def __init__(self) -> None:
        self._start: Optional[float] = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *_) -> None:
        pass
