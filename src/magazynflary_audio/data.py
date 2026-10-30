import pathlib
import typing
from collections.abc import Callable

import joblib

HERE = pathlib.Path(__file__).parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "data"

CACHE_MEMORY = joblib.Memory(DATA_DIR / "cache")


def cache[**P, R](*, ignore: list[str] | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Typed wrapper around CACHE_MEMORY.cache, which joblib ships without type hints"""
    return typing.cast(Callable[[Callable[P, R]], Callable[P, R]], CACHE_MEMORY.cache(ignore=ignore))
