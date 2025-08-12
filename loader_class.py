from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import os
import shutil
from typing import Optional, Tuple

# New modular helpers
from src.dbx_hf_loader.cache import configure_hf_env, compute_cache_paths
from src.dbx_hf_loader.hf_auth import get_hf_token, login_to_hf
from src.dbx_hf_loader.download import ensure_model_cached
from src.dbx_hf_loader.volume import copy_tree_if_missing


class LoaderClass:
    def __init__(
        self,
        model_name,
        model_path,
        model_type,
        model_config,
        model_signature,
        catalog_name,
        schema_name,
        volume_name,
        revision,
        secret_scope: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        self.model_config = model_config
        self.model_signature = model_signature
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.volume_name = volume_name
        self.revision = revision
        self.secret_scope = secret_scope
        self.secret_key = secret_key

    def _set_environment_vars(
        self,
        catalog_name=None,
        schema_name=None,
        volume_name=None,
        model_name=None,
        revision=None,
    ) -> Tuple[str, str]:
        catalog_name = self.catalog_name if catalog_name is None else catalog_name
        schema_name = self.schema_name if schema_name is None else schema_name
        volume_name = self.volume_name if volume_name is None else volume_name
        model_name = self.model_name if model_name is None else model_name
        revision = self.revision if revision is None else revision

        volume_root = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/hub"
        local_root = "/local_disk0/hf_cache"

        configure_hf_env(local_root)
        cache_volume, cache_local = compute_cache_paths(
            model_id=model_name,
            revision=revision,
            volume_cache_root=volume_root,
            local_cache_root=local_root,
        )
        return cache_volume, cache_local

    def _dbx_login_to_huggingface(self, secret_scope=None, secret_key=None):
        secret_scope = self.secret_scope if secret_scope is None else secret_scope
        secret_key = self.secret_key if secret_key is None else secret_key
        token = get_hf_token(secret_scope=secret_scope, secret_key=secret_key)
        ok = login_to_hf(token)
        if not ok:
            print("Warning: Could not login to Hugging Face. Proceeding without persistent auth.")
        return ok
        

    def _dbx_download_model(
        self,
        model_id,
        revision=None,
        cache_volume=None,
        cache_local=None,
        secret_scope=None,
        secret_key=None,
    ):
        secret_scope = self.secret_scope if secret_scope is None else secret_scope
        secret_key = self.secret_key if secret_key is None else secret_key

        if self._dbx_login_to_huggingface(secret_scope=secret_scope, secret_key=secret_key) is False:
            raise Exception("Failed to login to Hugging Face")

        token = get_hf_token(secret_scope=secret_scope, secret_key=secret_key)

        if cache_volume is None or cache_local is None:
            cache_volume, cache_local = self._set_environment_vars(
                model_name=model_id, revision=revision
            )

        ensure_model_cached(
            model_id=model_id, cache_dir=cache_local, revision=revision, token=token
        )

    def _dbx_copy_model_to_disk(self, cache_source, cache_target):
        cache_source = cache_source
        cache_target = cache_target

        # Copy volume cache to local cache if not already there
        if not os.path.exists(cache_target):
            try:
                print(f"Loading model from {cache_source} to {cache_target}.")
                snapshots_dir = "/".join(cache_target.split("/")[:-1])
                if not os.path.exists(snapshots_dir):
                    os.makedirs(snapshots_dir)
                copy_tree_if_missing(cache_source, cache_target)
                print(
                    f"Successfully loaded model from {cache_source} to {cache_target}!"
                )
            except Exception as e:
                print(f"Error: {e}")
        


    def load_model(self):
        pass

    def load_processor(self):
        pass
    
    def load_model_signature(self):
        pass
    
    def load_model_config(self):
        pass

    def load_model_from_mlflow(self):
        pass

    def load_model_from_local(self):
        pass