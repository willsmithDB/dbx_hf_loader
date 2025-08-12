from __future__ import annotations
import argparse

from .cache import configure_hf_env, compute_cache_paths, DEFAULT_LOCAL_CACHE_ROOT
from .hf_auth import get_hf_token, login_to_hf
from .download import cache_model_with_revision
from .volume import copy_tree_if_missing


def _volume_root(catalog: str, schema: str, volume: str) -> str:
    return f"/Volumes/{catalog}/{schema}/{volume}/hub"


def cmd_cache(args: argparse.Namespace) -> int:
    local_root = args.local_cache_root or DEFAULT_LOCAL_CACHE_ROOT
    volume_root = args.volume_cache_root or _volume_root(args.catalog, args.schema, args.volume)

    configure_hf_env(local_root)
    token = get_hf_token(args.secret_scope, args.secret_key)
    login_to_hf(token)

    volume_cache_path, local_cache_path = compute_cache_paths(
        args.model_id, args.revision, volume_root, local_root
    )

    cache_model_with_revision(args.model_id, local_cache_path, args.revision, token)
    copy_tree_if_missing(local_cache_path, volume_cache_path)

    print(f"Cached to local: {local_cache_path}")
    print(f"Mirrored to volume: {volume_cache_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="dbx-hf", description="Databricks HF utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cache = sub.add_parser("cache", help="Download model to local cache and mirror to UC volume")
    p_cache.add_argument("--model-id", required=True, help="Hugging Face repo id, e.g. meta-llama/Llama-3-8B")
    p_cache.add_argument("--revision", required=True, help="Exact model commit or tag")
    p_cache.add_argument("--catalog", required=True)
    p_cache.add_argument("--schema", required=True)
    p_cache.add_argument("--volume", required=True)
    p_cache.add_argument("--local-cache-root", default=DEFAULT_LOCAL_CACHE_ROOT)
    p_cache.add_argument("--volume-cache-root", default=None)
    p_cache.add_argument("--secret-scope", default=None)
    p_cache.add_argument("--secret-key", default=None)
    p_cache.set_defaults(func=cmd_cache)

    args = parser.parse_args()
    raise SystemExit(args.func(args))
