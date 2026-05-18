"""Profile registry: discover every YAML profile in a folder.

The folder is treated as an **arg list** — drop a new ``<disease>.yaml`` in,
and the engine picks it up on the next run with no Python changes.
"""

from __future__ import annotations

from pathlib import Path

from disease_warehouse.core.profile import DatasetProfile, load_profile


def discover_profiles(profiles_dir: str | Path) -> list[DatasetProfile]:
    profiles_dir = Path(profiles_dir)
    if not profiles_dir.is_dir():
        raise FileNotFoundError(f"Profiles directory not found: {profiles_dir}")
    files = sorted(p for p in profiles_dir.iterdir() if p.suffix.lower() in {".yaml", ".yml"})
    profiles = [load_profile(p) for p in files]
    seen: set[str] = set()
    for prof in profiles:
        if prof.name in seen:
            raise ValueError(f"Duplicate profile name {prof.name!r} in {profiles_dir}")
        seen.add(prof.name)
    return profiles
