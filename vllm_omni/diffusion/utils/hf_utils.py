from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
from functools import lru_cache
from typing import Any

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

logger = init_logger(__name__)


def get_local_or_hf_file_to_dict(file_name: str, model_name_or_path: str | None) -> dict | None:
    if model_name_or_path is None:
        return None

    if os.path.isdir(model_name_or_path):
        local_path = os.path.join(model_name_or_path, file_name)
        if not os.path.exists(local_path):
            return None
        with open(local_path) as f:
            return json.load(f)

    return get_hf_file_to_dict(file_name, model_name_or_path)


def _link_or_copy_file(src: str, dst: str) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_safetensors_index(shard_paths: list[str], index_path: str) -> None:
    from safetensors import safe_open

    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_path in shard_paths:
        shard_name = os.path.basename(shard_path)
        total_size += os.path.getsize(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = shard_name

    with open(index_path, "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)


def load_pretrained_component_model(
    model_cls: Any,
    model_name_or_path: str,
    *,
    subfolder: str | None = None,
    config_file_name: str = "config.json",
    index_file_name: str = "model.safetensors.index.json",
    **kwargs: Any,
) -> Any:
    component_dir = os.path.join(model_name_or_path, subfolder) if subfolder is not None else model_name_or_path
    if not os.path.isdir(component_dir):
        return model_cls.from_pretrained(model_name_or_path, subfolder=subfolder, **kwargs)

    standard_single_file = index_file_name.removesuffix(".index.json")
    shard_paths = sorted(glob.glob(os.path.join(component_dir, "*.safetensors")))
    if (
        not shard_paths
        or os.path.exists(os.path.join(component_dir, index_file_name))
        or os.path.exists(os.path.join(component_dir, standard_single_file))
    ):
        return model_cls.from_pretrained(model_name_or_path, subfolder=subfolder, **kwargs)

    config_path = os.path.join(component_dir, config_file_name)
    if not os.path.exists(config_path):
        return model_cls.from_pretrained(model_name_or_path, subfolder=subfolder, **kwargs)

    logger.info("Synthesizing %s for local component %s", index_file_name, component_dir)
    forwarded_kwargs = dict(kwargs)
    forwarded_kwargs.pop("subfolder", None)
    forwarded_kwargs["local_files_only"] = True

    with tempfile.TemporaryDirectory(prefix="vllm_omni_component_") as temp_dir:
        _link_or_copy_file(config_path, os.path.join(temp_dir, config_file_name))
        for shard_path in shard_paths:
            _link_or_copy_file(shard_path, os.path.join(temp_dir, os.path.basename(shard_path)))
        _write_safetensors_index(shard_paths, os.path.join(temp_dir, index_file_name))
        return model_cls.from_pretrained(temp_dir, **forwarded_kwargs)


def load_diffusers_config(model_name) -> dict:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    config = DiffusionPipeline.load_config(model_name)
    return config


def _looks_like_bagel(model_name: str) -> bool:
    """Best-effort detection for Bagel (non-diffusers) diffusion models."""
    try:
        cfg = get_local_or_hf_file_to_dict("config.json", model_name)
        if cfg is None:
            return False
        model_type = cfg.get("model_type")
        if model_type == "bagel":
            return True
        architectures = cfg.get("architectures") or []
        return "BagelForConditionalGeneration" in architectures
    except Exception:
        return False


@lru_cache
def is_diffusion_model(model_name: str) -> bool:
    """Check if a model is a diffusion model.

    Uses multiple fallback strategies to detect diffusion models:
    1. Check local file system for model_index.json (fastest, no imports)
    2. Check using vllm's get_hf_file_to_dict utility
    3. Try the standard diffusers approach (may fail due to import issues)
    """
    # Strategy 1: Check local file system first (fastest, avoids import issues)
    if os.path.isdir(model_name):
        model_index_path = os.path.join(model_name, "model_index.json")
        if os.path.exists(model_index_path):
            try:
                import json

                with open(model_index_path) as f:
                    config_dict = json.load(f)
                if config_dict.get("_class_name") and config_dict.get("_diffusers_version"):
                    logger.debug("Detected diffusion model via local model_index.json")
                    return True
            except Exception as e:
                logger.debug("Failed to read local model_index.json: %s", e)

    # Strategy 2: Check using vllm's utility (works for both local and remote models)
    try:
        config_dict = get_local_or_hf_file_to_dict("model_index.json", model_name)
        if config_dict is not None and config_dict.get("_class_name") and config_dict.get("_diffusers_version"):
            logger.debug("Detected diffusion model via model_index.json")
            return True
    except Exception as e:
        logger.debug("Failed to check model_index.json via get_hf_file_to_dict: %s", e)

    # Strategy 3: Try the standard diffusers approach (may fail due to import issues)
    # This is last because it requires importing diffusers/xformers/flash_attn
    # which may have compatibility issues
    try:
        load_diffusers_config(model_name)
        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.debug("Failed to import diffusers dependencies: %s", e)
        logger.debug("This may be due to flash_attn/PyTorch version mismatch")
    except Exception as e:
        logger.debug("Failed to load diffusers config via DiffusionPipeline: %s", e)

        # Bagel is not a diffusers pipeline (no model_index.json), but is still a
        # diffusion-style model in vllm-omni. Detect it via config.json.
    return _looks_like_bagel(model_name)
