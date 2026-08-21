"""Every declared multinet-* console script must resolve to a callable.

Guards the packaging surface: a [project.scripts] target pointing at a
missing function ships an installed command that crashes on first run.
"""
from importlib.metadata import entry_points

import pytest

_EPS = sorted(
    (ep for ep in entry_points(group="console_scripts") if ep.name.startswith("multinet-")),
    key=lambda ep: ep.name,
)


def test_all_declared_entry_points_found():
    assert len(_EPS) >= 10, (
        "expected the 10 multinet-* console scripts from pyproject; "
        "is the package installed (pip install -e .)?"
    )


@pytest.mark.parametrize("ep", _EPS, ids=lambda ep: ep.name)
def test_console_script_target_resolves(ep):
    try:
        target = ep.load()
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional third-party dependency not installed: {exc.name}")
    assert callable(target), f"{ep.name} -> {ep.value} is not callable"
