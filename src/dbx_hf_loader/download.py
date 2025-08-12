from __future__ import annotations
from typing import Optional
from huggingface_hub import snapshot_download


def cache_model_with_revision(
    model_id: str,
    cache_dir: str,
    revision: str = None,
    token: Optional[str] = None,
) -> str:
    return snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
    )

def cache_model_without_revision(
    model_id: str,
    cache_dir: str,
    token: Optional[str] = None,
) -> str:
    return snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        token=token,
    )
