"""Smoke tests verifying package metadata, structure, and imports."""

import ultimo_marl
from ultimo_marl import agents, domain, env, protocol, training, utils


def test_package_metadata() -> None:
    """Verify package version and author strings."""
    assert hasattr(ultimo_marl, "__version__")
    assert ultimo_marl.__version__ == "0.1.0"
    assert hasattr(ultimo_marl, "__author__")
    assert ultimo_marl.__author__ == "Stanisław Golec"


def test_subpackages_importable() -> None:
    """Verify all architectural subpackages are importable."""
    assert domain is not None
    assert env is not None
    assert agents is not None
    assert training is not None
    assert protocol is not None
    assert utils is not None
