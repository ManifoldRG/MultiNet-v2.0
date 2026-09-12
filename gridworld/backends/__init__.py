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

Feature Comparison (see base.py for full table):
    - MiniGrid: Best for standard square grid tasks, more mature/tested
    - MultiGrid: Required for hex/triangle tilings or zones/teleporters

Usage:
    from gridworld.backends import get_backend

    # Standard square grid
    backend = get_backend("minigrid", render_mode="rgb_array")

    # Exotic tilings (hex, triangle)
    backend = get_backend("multigrid", tiling="triangle", render_mode="rgb_array")
"""

from .base import AbstractGridBackend, GridState

__all__ = [
    "AbstractGridBackend",
    "GridState",
    "MiniGridBackend",
    "MultiGridBackend",
    "get_backend",
]


def __getattr__(name: str):
    # Concrete backends load lazily so importing gridworld.backends (or
    # gridworld.backends.base) never imports minigrid: minigrid-free code
    # paths (the 3D backend, interface/*) depend on that.
    if name == "MiniGridBackend":
        from .minigrid_backend import MiniGridBackend

        globals()[name] = MiniGridBackend
        return MiniGridBackend
    if name == "MultiGridBackend":
        try:
            from .multigrid_backend import MultiGridBackend
        except ImportError:
            MultiGridBackend = None
        globals()[name] = MultiGridBackend
        return MultiGridBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_backend(name: str, **kwargs) -> AbstractGridBackend:
    """
    Get a backend instance by name.

    Args:
        name: Backend name ("minigrid" or "multigrid")
        **kwargs: Arguments passed to backend constructor

    Returns:
        Backend instance

    Raises:
        ValueError: If backend name is unknown or unavailable
    """
    if name == "minigrid":
        from .minigrid_backend import MiniGridBackend

        return MiniGridBackend(**kwargs)
    if name == "multigrid":
        try:
            from .multigrid_backend import MultiGridBackend
        except ImportError as exc:
            raise ValueError(
                "MultiGridBackend not available. "
                "Ensure multigrid module is accessible."
            ) from exc
        return MultiGridBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name}")
