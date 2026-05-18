"""Profile-driven disease data-mining warehouse engine.

Public surface is kept minimal on purpose: every dataset onboards through a
YAML profile in :mod:`disease_warehouse.profiles`. The engine discovers all
profiles, auto-collects metadata, cleans the data, builds a star schema with
conformed dimensions plus per-disease fact tables, and emits a SQLite gold
warehouse alongside Parquet files.
"""

from disease_warehouse.core.profile import DatasetProfile, load_profile
from disease_warehouse.core.registry import discover_profiles

__all__ = ["DatasetProfile", "load_profile", "discover_profiles"]
