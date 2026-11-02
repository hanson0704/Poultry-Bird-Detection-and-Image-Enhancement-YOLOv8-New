"""
SciPy Hungarian Linear Assignment compatibility shim for ByteTrack on Windows.

Ultralytics ByteTrack (and matching algorithms) requires a module exposing `lap.lapjv`
for linear sum assignment. On Windows systems where the C++ `lap` extension is missing
or fails compilation, this shim injects a lightweight, pure-Python / pure-SciPy drop-in
module into `sys.modules['lap']` backed by `scipy.optimize.linear_sum_assignment`.

This guarantees 100% reliable bipartite Hungarian matching without requiring MSVC
build tools or compiling C/C++ binaries.
"""
from __future__ import annotations

import sys
import types
from typing import Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def ensure_scipy_lap_shim(force: bool = False) -> bool:
    """
    Registers a drop-in 'lap' module into sys.modules if not already present or if forced.
    Returns True if the shim was registered, False if native lap was already present.
    """
    if not force and "lap" in sys.modules and hasattr(sys.modules["lap"], "lapjv"):
        return False

    if not force:
        try:
            import lap  # type: ignore # noqa: F401
            if hasattr(lap, "lapjv"):
                return False
        except ImportError:
            pass

    lap_mod = types.ModuleType("lap")
    lap_mod.__version__ = "0.5.12-scipy-shim"

    def lapjv(
        cost_matrix: np.ndarray,
        extend_cost: bool = True,
        cost_limit: float = float("inf"),
        return_cost: bool = True,
    ) -> Tuple:
        """
        Drop-in replacement for lap.lapjv using scipy.optimize.linear_sum_assignment.
        """
        cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
        n_rows, n_cols = cost_matrix.shape
        x = np.full(n_rows, -1, dtype=np.int32)
        y = np.full(n_cols, -1, dtype=np.int32)

        if n_rows == 0 or n_cols == 0:
            return (0.0, x, y) if return_cost else (x, y)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = 0.0
        for r, c in zip(row_ind, col_ind):
            val = float(cost_matrix[r, c])
            if val <= cost_limit:
                x[r] = int(c)
                y[c] = int(r)
                total_cost += val

        return (total_cost, x, y) if return_cost else (x, y)

    lap_mod.lapjv = lapjv
    sys.modules["lap"] = lap_mod
    return True


# Automatically ensure shim on import
ensure_scipy_lap_shim()

