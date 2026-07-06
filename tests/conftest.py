"""
Shared pytest fixtures for the gyroid_utils test suite.
"""
import numpy as np
import pytest


@pytest.fixture
def small_grid():
    """
    A small (8x8x8) 3D coordinate grid, cheap enough for fast unit tests.

    This is a pytest fixture: a reusable piece of setup. Any test function
    that declares a parameter named `small_grid` automatically receives this
    freshly-built grid before the test body runs.
    """
    lin = np.linspace(0.0, 4.0, 8)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    return x, y, z
