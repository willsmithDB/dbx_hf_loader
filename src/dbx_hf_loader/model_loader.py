from __future__ import annotations
from typing import Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

def build_bnb_config(
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = False,
) -> Optional[BitsAndBytesConfig]:
    if not (load_in_4bit or load_in_8bit):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def load_tokenizer(
    model_id: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
):
    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )


def load_processor(
    model_id: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
):
    return AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )


def load_causal_lm(
    model_id: str,
    device_map: Union[str, dict] = "auto",
    dtype: Union[str, torch.dtype] = torch.bfloat16,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    bnb_config: Optional[BitsAndBytesConfig] = None,
):
    kwargs = dict(revision=revision, trust_remote_code=trust_remote_code)
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=dtype,
        **kwargs,
    )
