from __future__ import annotations
from typing import Optional
import os

from huggingface_hub import login

def get_hf_token(secret_scope: Optional[str] = None, secret_key: Optional[str] = None) -> Optional[str]:
    if secret_scope and secret_key:
        try:
            return dbutils.secrets.get(scope=secret_scope, key=secret_key)  # type: ignore[attr-defined]
        except Exception:
            print("Warning: Could not get token from secrets. Proceeding without persistent auth.")
    return os.getenv("HUGGINGFACE_HUB_TOKEN")


def login_to_hf(token: Optional[str]) -> bool:
    try:
        if token:
            login(token=token, add_to_git_credential=True)
        return True
    except Exception:
        return False
