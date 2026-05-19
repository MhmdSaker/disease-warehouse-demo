"""CLI entrypoint: ``python -m disease_warehouse build``.

Subcommands
-----------
build           Run end-to-end pipeline against every profile in profiles/.
list-profiles   Show every discovered dataset profile.
inspect         Print metadata/auto-profile for a single profile, no DB writes.
scaffold        Generate a starter YAML profile from a CSV header.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Load .env (repo root) so GEMINI_API_KEY etc. are available without the user
# having to export them every shell session. Silent no-op if python-dotenv
# isn't installed or no .env file exists.
try:
    from dotenv import load_dotenv
    _ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
except ImportError:
    pass

from disease_warehouse.core.loader import WarehouseBuilder
from disease_warehouse.core.metadata import profile_dataset
from disease_warehouse.core.profile import load_profile
from disease_warehouse.core.profile_inference import HIGH, MEDIUM, LOW, infer_profile
from disease_warehouse.core.registry import discover_profiles
from disease_warehouse.core.slm_catalog import (
    PRESET_CHOICES,
    PRESETS,
    PRIVACY_CHOICES,
    resolve_preset,
)
from disease_warehouse.core.yaml_writer import emit_yaml


ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILES_DIR = ENGINE_DIR / "profiles"
DEFAULT_OUTPUTS_DIR = ENGINE_DIR / "outputs"
DEFAULT_ROOT_DIR = ENGINE_DIR.parent  # repo root — source paths are relative to here


def _banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def cmd_build(args: argparse.Namespace) -> int:
    profiles = discover_profiles(args.profiles_dir)
    if args.profile:
        profiles = [p for p in profiles if p.name == args.profile]
        if not profiles:
            print(f"[error] no profile named {args.profile!r} in {args.profiles_dir}", file=sys.stderr)
            return 2

    _banner("DISEASE WAREHOUSE BUILD")
    print(f"  profiles_dir : {args.profiles_dir}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"  root_dir     : {args.root_dir}")
    print(f"  profiles     : {[p.name for p in profiles]}")

    builder = WarehouseBuilder(
        root_dir=Path(args.root_dir),
        output_dir=Path(args.output_dir),
    )
    summary = builder.build(profiles)

    _banner("BUILD COMPLETE")
    print(f"  gold DB        : {summary['db_path']}")
    print("  warehouse rows :")
    for table, count in summary["warehouse_counts"].items():
        print(f"      {table:<32} {count:>8,}")
    print(f"  parquet dir    : {builder.parquet_dir}")
    print(f"  data dict      : {builder.dictionary_path}")
    print(f"  star schema    : {builder.diagram_md_path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    profiles = discover_profiles(args.profiles_dir)
    _banner("REGISTERED DISEASE PROFILES")
    for p in profiles:
        print(f"  - {p.name:<14} domain={p.domain:<16} source={p.source_path}")
        print(f"      {p.display_name}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    prof = load_profile(args.profile_path)
    src = Path(args.root_dir) / prof.source_path
    raw = pd.read_csv(src, na_values=prof.na_values)
    records = profile_dataset(raw, prof)
    _banner(f"INSPECT: {prof.name}")
    print(json.dumps([r.to_dict() for r in records], indent=2, default=str))
    return 0


CONF_RANK = {HIGH: 0, MEDIUM: 1, LOW: 2}
CONF_GLYPH = {HIGH: "[OK]", MEDIUM: "[~]", LOW: "[REVIEW]"}


def cmd_scaffold(args: argparse.Namespace) -> int:
    """Auto-profile a CSV into a YAML using statistical + medical heuristics.

    Replaces the old dumb stub. The output is a complete, runnable profile —
    every type, role, mapping, description, hierarchy, and target encoding is
    inferred. Low-confidence decisions are flagged as ``# REVIEW:`` comments.
    """
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}", file=sys.stderr)
        return 2

    # Compute the path the YAML should record (relative to the repo root so
    # `python -m disease_warehouse build` finds it when run from anywhere).
    root_dir = Path(args.root_dir).resolve()
    try:
        yaml_relative = csv_path.resolve().relative_to(root_dir).as_posix()
    except ValueError:
        yaml_relative = csv_path.as_posix()

    # Resolve preset + privacy. The new --slm / --privacy flags win when set;
    # otherwise the back-compat --use-llm / --no-llm are translated and the
    # final fallback is the long-standing "auto-on if GEMINI_API_KEY present"
    # heuristic.
    slm_preset = args.slm
    privacy_mode = args.privacy or "balanced"
    use_llm_legacy: bool | None = None  # only used when slm_preset is None

    if slm_preset is None:
        if args.use_llm is True:
            slm_preset = "cloud"
            privacy_mode = "cloud-only"
        elif args.use_llm is False:
            slm_preset = "off"
        else:
            slm_preset = "auto"
    # If user explicitly passed --slm cloud and didn't touch --privacy,
    # default to cloud-only so it behaves like the old --use-llm did.
    if slm_preset == "cloud" and args.privacy is None:
        privacy_mode = "cloud-only"

    resolved_preset = resolve_preset(slm_preset, privacy_mode)

    result = infer_profile(
        csv_path,
        name=args.name,
        target=args.target,
        source_path_for_yaml=yaml_relative,
        use_llm=use_llm_legacy,
        llm_cache_dir=DEFAULT_OUTPUTS_DIR,
        slm_preset=slm_preset,
        privacy_mode=privacy_mode,
        slm_model_override=args.slm_model,
        disable_embedding=getattr(args, "slm_no_embed", False),
    )

    out_path = Path(args.profiles_dir) / f"{result.profile['name']}.yaml"
    if out_path.exists() and not args.force:
        print(f"[error] {out_path} already exists; pass --force to overwrite", file=sys.stderr)
        return 2

    yaml_text = emit_yaml(result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_text, encoding="utf-8")

    _banner(f"AUTO-PROFILE REPORT: {result.profile['name']}")
    print(f"  source         : {csv_path}")
    print(f"  delimiter      : {result.delimiter!r}")
    print(f"  target column  : {result.profile['target']['column']}")
    print(f"  output         : {out_path}")
    print(f"  slm preset     : {slm_preset} -> {resolved_preset} "
          f"({PRESETS[resolved_preset].model or 'no model'})")
    print(f"  privacy mode   : {privacy_mode}")
    llm_decision = next((d for d in result.decisions if d.field == "llm"), None)
    if llm_decision:
        print(f"  enrichment     : {llm_decision.value}")
    elif slm_preset == "off":
        print(f"  enrichment     : disabled")
    else:
        print(f"  enrichment     : enabled (no LOW-confidence columns needed enrichment)")
    print()
    # Effective counts: collapse multiple decisions per (column, field) to
    # the best confidence so LLM overrides reduce LOW instead of inflating HIGH.
    # Columns with drop:true don't count — their fields are never used.
    dropped_cols = {c["source"] for c in result.profile["columns"] if c.get("drop")}
    best_by_pair: dict[tuple[str, str], object] = {}
    for d in result.decisions:
        if d.column in dropped_cols:
            continue
        key = (d.column, d.field)
        cur = best_by_pair.get(key)
        if cur is None or CONF_RANK[d.confidence] < CONF_RANK[cur.confidence]:
            best_by_pair[key] = d
    counts = {HIGH: 0, MEDIUM: 0, LOW: 0}
    for d in best_by_pair.values():
        counts[d.confidence] = counts.get(d.confidence, 0) + 1
    print(f"  decisions      : HIGH={counts[HIGH]}  MEDIUM={counts[MEDIUM]}  LOW={counts[LOW]} (effective, post-LLM)")
    print()
    print("  Per-column summary (showing low-confidence in full):")
    by_col: dict[str, list] = {}
    for d in result.decisions:
        if d.column == "<file>":
            continue
        by_col.setdefault(d.column, []).append(d)
    for col, decisions in by_col.items():
        # Dropped columns are summarised as a single OK line, no field detail.
        if col in dropped_cols:
            print(f"  {CONF_GLYPH[HIGH]} {col:<22} (dropped)")
            continue
        # Per (column, field), keep the best-confidence decision — this
        # collapses original LOW + later LLM HIGH into the effective HIGH.
        best_by_field: dict[str, object] = {}
        for d in decisions:
            cur = best_by_field.get(d.field)
            if cur is None or CONF_RANK[d.confidence] < CONF_RANK[cur.confidence]:
                best_by_field[d.field] = d
        effective = list(best_by_field.values())
        worst = max(effective, key=lambda d: CONF_RANK[d.confidence])
        glyph = CONF_GLYPH[worst.confidence]
        if worst.confidence == LOW:
            for d in effective:
                if d.is_review():
                    print(f"  {glyph} {col:<22} {d.field:<14} {d.reason}")
        else:
            type_field = next((d for d in effective if d.field == "type"), None)
            role_field = next((d for d in effective if d.field == "role"), None)
            desc_field = next((d for d in effective if d.field == "description"), None)
            map_field = next((d for d in effective if d.field == "mapping"), None)
            extras = []
            if type_field:
                extras.append(f"type={type_field.value}")
            if role_field and role_field.value:
                role_str = f"role={role_field.value}"
                if role_field.provider:
                    role_str += f"[{role_field.provider}]"
                extras.append(role_str)
            # Mark fields that a provider filled (description / mapping) so
            # the provenance is visible even when role came from a pattern.
            provider_tags: list[str] = []
            for f in (desc_field, map_field):
                if f and f.provider and f.provider not in provider_tags:
                    provider_tags.append(f.provider)
            if provider_tags:
                extras.append(f"enriched-by=[{','.join(provider_tags)}]")
            print(f"  {glyph} {col:<22} {' '.join(extras)}")
    if counts[LOW] > 0:
        print()
        print(f"  NOTE: {counts[LOW]} effective low-confidence decision(s) emitted as '# REVIEW:'")
        print("        comments in the YAML. Open the file and resolve them before running build.")
    print()
    print(f"  Next: python -X utf8 -m disease_warehouse build --profile {result.profile['name']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="disease_warehouse")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--profiles-dir", default=str(DEFAULT_PROFILES_DIR))
        sp.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))

    p_build = sub.add_parser("build", help="Build the gold warehouse from all profiles.")
    add_common(p_build)
    p_build.add_argument("--output-dir", default=str(DEFAULT_OUTPUTS_DIR))
    p_build.add_argument("--profile", help="Only build this one profile name.")
    p_build.set_defaults(func=cmd_build)

    p_list = sub.add_parser("list-profiles", help="List discovered dataset profiles.")
    add_common(p_list)
    p_list.set_defaults(func=cmd_list)

    p_inspect = sub.add_parser("inspect", help="Print metadata for a single profile.")
    p_inspect.add_argument("profile_path", help="Path to a YAML profile.")
    p_inspect.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    p_inspect.set_defaults(func=cmd_inspect)

    p_scaffold = sub.add_parser(
        "scaffold",
        help="Auto-profile a CSV into a complete YAML using inference + medical heuristics.",
    )
    p_scaffold.add_argument("csv", help="Path to a CSV file.")
    p_scaffold.add_argument("--name", help="Profile name (default: derived from filename).")
    p_scaffold.add_argument("--target", help="Target column name (default: auto-detected).")
    p_scaffold.add_argument("--profiles-dir", default=str(DEFAULT_PROFILES_DIR))
    p_scaffold.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    p_scaffold.add_argument("--force", action="store_true")

    # Back-compat aliases — kept working.
    llm_group = p_scaffold.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--use-llm", dest="use_llm", action="store_true", default=None,
        help="Back-compat alias for --slm cloud --privacy cloud-only "
             "(requires GEMINI_API_KEY).",
    )
    llm_group.add_argument(
        "--no-llm", dest="use_llm", action="store_false",
        help="Back-compat alias for --slm off.",
    )

    # New SLM controls
    slm_group = p_scaffold.add_argument_group("SLM enrichment")
    slm_group.add_argument(
        "--slm", choices=PRESET_CHOICES, default=None,
        help="SLM preset (default: auto — local if ollama is reachable, "
             "else cloud if GEMINI_API_KEY set, else embed-only, else off).",
    )
    slm_group.add_argument(
        "--slm-model", default=None,
        help="Override the model id for the active backend "
             "(e.g. 'phi3.5:mini' for ollama, or 'gemini-2.5-flash' for cloud).",
    )
    slm_group.add_argument(
        "--slm-batch-size", type=int, default=40,
        help="Max columns per inference call (default: 40).",
    )
    slm_group.add_argument(
        "--slm-no-embed", dest="slm_no_embed", action="store_true",
        help="Skip the embedding pre-pass even when sentence-transformers is "
             "installed. Use for clean A/B benchmarks (deterministic vs "
             "deterministic+SLM vs deterministic+embed+SLM).",
    )

    privacy_group = p_scaffold.add_argument_group("Privacy / fallback")
    privacy_group.add_argument(
        "--privacy", choices=PRIVACY_CHOICES, default=None,
        help="strict = local SLM only, never cloud. "
             "balanced = local first, cloud for residue (default). "
             "cloud-only = legacy --use-llm behavior.",
    )

    p_scaffold.set_defaults(func=cmd_scaffold)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
