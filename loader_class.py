from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from PIL import Image
import pandas as pd
import requests, torch, mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import mlflow.pyfunc
import os
from huggingface_hub import login, snapshot_download
import shutil


class LoaderClass:
    def __init__(self, model_name, model_path, model_type, model_config, model_signature, catalog_name, schema_name, volume_name, revision):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        self.model_config = model_config
        self.model_signature = model_signature
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.volume_name = volume_name
        self.revision = revision

    def _set_environment_vars(
        self,
        catalog_name=None,
        schema_name=None,
        volume_name=None,
        model_name=None,
        revision=None,
    ):

        catalog_name = self.catalog_name if catalog_name is None else catalog_name
        schema_name = self.schema_name if schema_name is None else schema_name
        volume_name = self.volume_name if volume_name is None else volume_name
        model_name = self.model_name if model_name is None else model_name
        revision = self.revision if revision is None else revision

        VOLUME_HUGGINGFACE_HUB_CACHE = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/hub"
        LOCAL_HUGGINGFACE_HUB_CACHE = "/local_disk0/hf_cache"
        MODEL_NAME = model_name
        MODEL_REVISION = revision

        rev_folder =  '/'.join(['', '--'.join(['models',] +  MODEL_NAME.split('/')), 'snapshots', MODEL_REVISION])
        cache_volume =  VOLUME_HUGGINGFACE_HUB_CACHE + rev_folder
        cache_local = LOCAL_HUGGINGFACE_HUB_CACHE + rev_folder

        os.environ["HF_HOME"] = LOCAL_HUGGINGFACE_HUB_CACHE
        os.environ["HF_HUB_CACHE"] = LOCAL_HUGGINGFACE_HUB_CACHE
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "True"
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1000"
        # os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # Enables optimized download backend

        return cache_volume, cache_local

    def _dbx_login_to_huggingface(self, secret_scope = None, secret_key = None):
        secret_scope = self.secret_scope if secret_scope is None else secret_scope
        secret_key = self.secret_key if secret_key is None else secret_key

        try: 
            login(token=dbutils.secrets.get(scope=secret_scope, key=secret_key), add_to_git_credential=True)
        except Exception as e:
            print(f"Error logging in to Hugging Face: {e}")
            return False
        
        return True
        

    def _dbx_download_model(self, model_id, revision = None, cache_volume = None, cache_local = None, secret_scope = None, secret_key = None):

        secret_scope = self.secret_scope if secret_scope is None else secret_scope
        secret_key = self.secret_key if secret_key is None else secret_key  

        if self._dbx_login_to_huggingface(secret_scope=secret_scope, secret_key=secret_key) is False:
            raise Exception("Failed to login to Hugging Face")

        hf_token = dbutils.secrets.get(scope=secret_scope, key=secret_key)

        if cache_volume is None or cache_local is None:
            cache_volume, cache_local = self._set_environment_vars(model_name=model_id, revision=revision)
    
        if revision is not None:
            snapshot_download(
                repo_id=model_id, 
                revision=revision, 
                cache_dir=cache_local,
                token=hf_token
            )
        else:
            snapshot_download(
                repo_id=model_id, 
                cache_dir=cache_local,
                token=hf_token
            )

    def _dbx_copy_model_to_disk(self, cache_source, cache_target):
        cache_source = self.cache_volume if cache_source is None else cache_source
        cache_target = self.cache_local if cache_target is None else cache_target

        # Copy volume cache to local cache if not already there
        if not os.path.exists(cache_target):
            try: 
                print(f"Loading model from {cache_source} to {cache_target}.")
                snapshots_dir = '/'.join(cache_target.split('/')[:-1])
                if not os.path.exists(snapshots_dir):
                    os.makedirs(snapshots_dir)
                    shutil.copytree(cache_source, cache_target) 
                print(f"Successfully loaded model from {cache_source} to {cache_target}!")
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