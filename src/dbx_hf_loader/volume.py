from __future__ import annotations
import os
import shutil


def copy_tree_if_missing(source_dir: str, target_dir: str) -> None:
    if os.path.exists(target_dir):
        return
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    shutil.copytree(source_dir, target_dir)
