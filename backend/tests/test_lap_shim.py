"""
Unit tests for the SciPy Hungarian LAP compatibility shim.
"""
import sys
import numpy as np

from app.core.lap_shim import ensure_scipy_lap_shim


def test_ensure_scipy_lap_shim_registers_module():
    # Force shim creation to test its exact logic
    registered = ensure_scipy_lap_shim(force=True)
    assert registered is True
    assert "lap" in sys.modules
    assert hasattr(sys.modules["lap"], "lapjv")


def test_lapjv_matches_minimum_cost_assignment():
    ensure_scipy_lap_shim(force=True)
    lap = sys.modules["lap"]

    # Cost matrix: 3 workers, 3 tasks
    # Worker 0 prefers Task 1 (cost 1.0)
    # Worker 1 prefers Task 2 (cost 2.0)
    # Worker 2 prefers Task 0 (cost 3.0)
    cost_matrix = np.array([
        [10.0, 1.0, 9.0],
        [8.0, 7.0, 2.0],
        [3.0, 6.0, 5.0],
    ], dtype=np.float64)

    total_cost, x, y = lap.lapjv(cost_matrix)

    # Optimum matching: 0->1, 1->2, 2->0 (sum = 1 + 2 + 3 = 6.0)
    assert np.isclose(total_cost, 6.0)
    assert list(x) == [1, 2, 0]
    assert list(y) == [2, 0, 1]


def test_lapjv_empty_and_rectangular_matrices():
    ensure_scipy_lap_shim(force=True)
    lap = sys.modules["lap"]

    # Empty matrix
    empty = np.empty((0, 0), dtype=np.float64)
    cost, x, y = lap.lapjv(empty)
    assert cost == 0.0
    assert len(x) == 0 and len(y) == 0

    # Rectangular: 2 workers, 3 tasks
    rect = np.array([
        [4.0, 1.0, 5.0],
        [2.0, 8.0, 3.0],
    ], dtype=np.float64)
    cost, x, y = lap.lapjv(rect)
    # Optimal: 0->1 (cost 1.0), 1->0 (cost 2.0) => sum = 3.0
    assert np.isclose(cost, 3.0)
    assert x[0] == 1
    assert x[1] == 0

