"""Backward-compatible imports for the original standalone module.

New code should import the API from :mod:`maioutils` or
:mod:`maioutils.synthetic`.
"""

from maioutils.synthetic import (
    SyntheticDataConfig,
    SyntheticDataGenerator,
    make_synthetic_dataframe,
)

__all__ = [
    "SyntheticDataConfig",
    "SyntheticDataGenerator",
    "make_synthetic_dataframe",
]
