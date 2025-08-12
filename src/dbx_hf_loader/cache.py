from __future__ import annotations
import os
from typing import Tuple

DEFAULT_LOCAL_CACHE_ROOT = "/local_disk0/hf_cache"


def build_rev_folder(model_id: str, revision: str) -> str:
    # e.g., /models--meta-llama--Llama-3/snapshots/<rev>
    return "/" + "--".join(["models"] + model_id.split("/")) + "/snapshots/" + revision


def compute_cache_paths(
    model_id: str,
    revision: str,
    volume_cache_root: str,
    local_cache_root: str = DEFAULT_LOCAL_CACHE_ROOT,
) -> Tuple[str, str]:
    rev_folder = build_rev_folder(model_id, revision).lstrip("/")
    cache_volume = os.path.join(volume_cache_root, rev_folder)
    cache_local = os.path.join(local_cache_root, rev_folder)
    return cache_volume, cache_local


essential_env = {
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "True",
    "HF_HUB_DOWNLOAD_TIMEOUT": "1000",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}


def configure_hf_env(local_cache_root: str = DEFAULT_LOCAL_CACHE_ROOT) -> None:
    os.environ["HF_HOME"] = local_cache_root
    os.environ["HF_HUB_CACHE"] = local_cache_root
    for key, value in essential_env.items():
        os.environ[key] = value
