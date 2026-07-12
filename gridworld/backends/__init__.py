"""
Backend Abstraction for Grid Environments

Provides pluggable backend implementations for gridworld environments.

Available Backends:
    MiniGridBackend: Standard MiniGrid (gymnasium) implementation
        - Square grid only
        - Full mechanism set (keys, doors, switches, gates, blocks, hazards, teleporters)
        - Partial observability: view cone + fog of war
        - Well tested, production-ready

    MultiGridBackend: Custom multigrid with exotic tilings
        - Square, hexagonal, triangle, 3-4-6-4, 4-8-8 tilings
        - Full mechanism set (keys, doors, switches, gates, hazards, teleporters, zones)
        - Partial observability: view cone + fog of war (BFS-based on adjacency graph)

    OgbenchBackend: OGBench's continuous-physics PointyEnv maze
        - Square grid layout, continuous point-mass navigation (not discretized)
        - Mechanism set: keys, doors, switches, gates (no blocks/teleporters/hazards)
        - No partial observability (always full)

Feature Comparison (see base.py for full table):
    - MiniGrid: Best for standard square grid tasks, more mature/tested
    - MultiGrid: Required for hex/triangle tilings or zones/teleporters
    - Ogbench: Continuous navigation instead of discrete grid steps

Usage:
    from gridworld.backends import get_backend

    # Standard square grid
    backend = get_backend("minigrid", render_mode="rgb_array")

    # Exotic tilings (hex, triangle)
    backend = get_backend("multigrid", tiling="triangle", render_mode="rgb_array")

    # Continuous-navigation PointyEnv maze
    backend = get_backend("ogbench", render_mode="rgb_array")
"""

from .base import AbstractGridBackend, GridState
from .minigrid_backend import MiniGridBackend

# MultiGridBackend is optional - requires multigrid module
try:
    from .multigrid_backend import MultiGridBackend
    _MULTIGRID_AVAILABLE = True
except ImportError:
    MultiGridBackend = None
    _MULTIGRID_AVAILABLE = False

# OgbenchBackend is optional - requires the ogbench submodule (and mujoco)
try:
    from .ogbench_backend import OgbenchBackend
    _OGBENCH_AVAILABLE = True
except ImportError:
    OgbenchBackend = None
    _OGBENCH_AVAILABLE = False

__all__ = [
    "AbstractGridBackend",
    "GridState",
    "MiniGridBackend",
    "MultiGridBackend",
    "OgbenchBackend",
]


def get_backend(name: str, **kwargs) -> AbstractGridBackend:
    """
    Get a backend instance by name.

    Args:
        name: Backend name ("minigrid", "multigrid", or "ogbench")
        **kwargs: Arguments passed to backend constructor

    Returns:
        Backend instance

    Raises:
        ValueError: If backend name is unknown or unavailable
    """
    if name == "minigrid":
        return MiniGridBackend(**kwargs)
    elif name == "multigrid":
        if not _MULTIGRID_AVAILABLE:
            raise ValueError(
                "MultiGridBackend not available. "
                "Ensure multigrid module is accessible."
            )
        return MultiGridBackend(**kwargs)
    elif name == "ogbench":
        if not _OGBENCH_AVAILABLE:
            raise ValueError(
                "OgbenchBackend not available. "
                "Ensure the ogbench submodule (and mujoco) is importable."
            )
        return OgbenchBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}")
