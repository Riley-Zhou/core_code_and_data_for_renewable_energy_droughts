"""
Centralized dependency import and environment check for the code workflow.

Usage:
    python requirements.py
"""

# Standard library dependencies.
import datetime
import gc
import glob
import importlib
import os
import re
import shutil
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

# Core third-party dependencies. Keys are import names; values are common aliases
# used across the workflow scripts.
CORE_DEPENDENCIES = {
    "cc3d": "cc3d",
    "humanize": "humanize",
    "netCDF4": "nc",
    "numpy": "np",
    "pandas": "pd",
    "psutil": "psutil",
    "pyarrow": "pa",
    "pyarrow.parquet": "pq",
    "tqdm": "tqdm_module",
    "xarray": "xr",
}


OPTIONAL_DEPENDENCIES = {
    "torch": "Used by the GPU acceleration path in the solar capacity-factor script.",
    "pynvml": "Used only for GPU utilization monitoring.",
}


def import_dependencies(dependencies):
    """Import dependencies one by one and expose successful imports as globals."""
    missing = {}
    for module_name, alias in dependencies.items():
        try:
            globals()[alias] = importlib.import_module(module_name)
        except Exception as exc:
            missing[module_name] = str(exc)
    if "tqdm_module" in globals():
        globals()["tqdm"] = globals()["tqdm_module"].tqdm
    return missing


def check_optional_dependencies():
    """Check optional dependencies and return missing entries."""
    missing = {}
    for module_name, description in OPTIONAL_DEPENDENCIES.items():
        try:
            globals()[module_name] = importlib.import_module(module_name)
        except Exception:
            missing[module_name] = description
    return missing


if __name__ == "__main__":
    missing_core = import_dependencies(CORE_DEPENDENCIES)
    missing_optional = check_optional_dependencies()

    if missing_core:
        print("The following core dependencies could not be imported:")
        for module_name, error_message in missing_core.items():
            print(f"- {module_name}: {error_message}")
    else:
        print("Core dependencies imported successfully.")

    if missing_optional:
        print("The following optional dependencies could not be imported:")
        for module_name, description in missing_optional.items():
            print(f"- {module_name}: {description}")
    else:
        print("Optional dependencies imported successfully.")

    if missing_core:
        sys.exit(1)
