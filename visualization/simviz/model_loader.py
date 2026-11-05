"""
Locate and import ``period_model.py`` without modifying it.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
from types import ModuleType

logger = logging.getLogger(__name__)

__all__ = ["MODEL_ENV_VAR", "load_model", "model_path"]

MODEL_ENV_VAR = "SIMVIZ_MODEL_PATH"

_MODULE_NAME = "period_model"

_cached: ModuleType | None = None


def load_model() -> ModuleType:
    """Import ``period_model`` from the sibling repository, once per process."""
    global _cached
    if _cached is not None:
        return _cached

    repo = pathlib.Path("..")  # use parent directory
    source = repo / f"{_MODULE_NAME}.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} not found (fix path in vizualization/simviz/model_loader.py)"
        )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {source}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)

    logger.info("loaded simulation model from %s", source)
    _cached = module
    return module
