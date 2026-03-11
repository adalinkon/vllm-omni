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


_LOCAL_FP8_INDEX_FILES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
)


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


def _list_local_safetensors_keys(component_dir: str) -> set[str]:
    for index_file_name in _LOCAL_FP8_INDEX_FILES:
        index_path = os.path.join(component_dir, index_file_name)
        if not os.path.exists(index_path):
            continue
        with open(index_path) as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        if isinstance(weight_map, dict):
            return set(weight_map)

    shard_paths = sorted(glob.glob(os.path.join(component_dir, "*.safetensors")))
    if not shard_paths:
        return set()

    from safetensors import safe_open

    keys: set[str] = set()
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys.update(f.keys())
    return keys


@lru_cache(maxsize=None)
def has_local_prequantized_fp8_weights(
    model_name_or_path: str,
    subfolder: str | None = None,
) -> bool:
    component_dir = os.path.join(model_name_or_path, subfolder) if subfolder is not None else model_name_or_path
    if not os.path.isdir(component_dir):
        return False

    keys = _list_local_safetensors_keys(component_dir)
    return any(
        key.endswith((".weight_scale", ".weight_scale_inv", ".input_scale", ".scale_weight", ".comfy_quant"))
        for key in keys
    )


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


_LEGACY_WAN_RESIDUAL_REWRITES = {
    ".residual.0.gamma": ".norm1.gamma",
    ".residual.2.weight": ".conv1.weight",
    ".residual.2.bias": ".conv1.bias",
    ".residual.3.gamma": ".norm2.gamma",
    ".residual.6.weight": ".conv2.weight",
    ".residual.6.bias": ".conv2.bias",
}

_LEGACY_WAN_DECODER_RESIDUAL_MAP = {
    0: (0, 0),
    1: (0, 1),
    2: (0, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
    8: (2, 0),
    9: (2, 1),
    10: (2, 2),
    12: (3, 0),
    13: (3, 1),
    14: (3, 2),
}

_LEGACY_WAN_DECODER_UPSAMPLER_MAP = {
    3: 0,
    7: 1,
    11: 2,
}


def _rewrite_legacy_wan_residual_suffix(suffix: str) -> str:
    for old, new in _LEGACY_WAN_RESIDUAL_REWRITES.items():
        if suffix.startswith(old):
            return suffix.replace(old, new, 1)
    return suffix


def _map_legacy_wan_vae_key(key: str) -> str:
    if key.startswith("conv1."):
        return key.replace("conv1.", "quant_conv.", 1)
    if key.startswith("conv2."):
        return key.replace("conv2.", "post_quant_conv.", 1)
    if key.startswith("encoder.conv1."):
        return key.replace("encoder.conv1.", "encoder.conv_in.", 1)
    if key.startswith("decoder.conv1."):
        return key.replace("decoder.conv1.", "decoder.conv_in.", 1)
    if key == "encoder.head.0.gamma":
        return "encoder.norm_out.gamma"
    if key.startswith("encoder.head.2."):
        return key.replace("encoder.head.2.", "encoder.conv_out.", 1)
    if key == "decoder.head.0.gamma":
        return "decoder.norm_out.gamma"
    if key.startswith("decoder.head.2."):
        return key.replace("decoder.head.2.", "decoder.conv_out.", 1)
    if key.startswith("encoder.middle.1."):
        return key.replace("encoder.middle.1.", "encoder.mid_block.attentions.0.", 1)
    if key.startswith("decoder.middle.1."):
        return key.replace("decoder.middle.1.", "decoder.mid_block.attentions.0.", 1)
    if key.startswith("encoder.middle.0"):
        return "encoder.mid_block.resnets.0" + _rewrite_legacy_wan_residual_suffix(key[len("encoder.middle.0") :])
    if key.startswith("encoder.middle.2"):
        return "encoder.mid_block.resnets.1" + _rewrite_legacy_wan_residual_suffix(key[len("encoder.middle.2") :])
    if key.startswith("decoder.middle.0"):
        return "decoder.mid_block.resnets.0" + _rewrite_legacy_wan_residual_suffix(key[len("decoder.middle.0") :])
    if key.startswith("decoder.middle.2"):
        return "decoder.mid_block.resnets.1" + _rewrite_legacy_wan_residual_suffix(key[len("decoder.middle.2") :])
    if key.startswith("encoder.downsamples."):
        parts = key.split(".")
        block_idx = parts[2]
        suffix = key[len(f"encoder.downsamples.{block_idx}") :]
        if suffix.startswith(".shortcut."):
            return f"encoder.down_blocks.{block_idx}.conv_shortcut.{suffix.split('.', 2)[2]}"
        if suffix.startswith(".residual."):
            return f"encoder.down_blocks.{block_idx}" + _rewrite_legacy_wan_residual_suffix(suffix)
        return f"encoder.down_blocks.{block_idx}{suffix}"
    if key.startswith("decoder.upsamples."):
        parts = key.split(".")
        old_idx = int(parts[2])
        suffix = key[len(f"decoder.upsamples.{old_idx}") :]
        if suffix.startswith(".shortcut."):
            block_idx, resnet_idx = _LEGACY_WAN_DECODER_RESIDUAL_MAP[old_idx]
            return f"decoder.up_blocks.{block_idx}.resnets.{resnet_idx}.conv_shortcut.{suffix.split('.', 2)[2]}"
        if suffix.startswith(".residual."):
            block_idx, resnet_idx = _LEGACY_WAN_DECODER_RESIDUAL_MAP[old_idx]
            return f"decoder.up_blocks.{block_idx}.resnets.{resnet_idx}" + _rewrite_legacy_wan_residual_suffix(suffix)
        if old_idx in _LEGACY_WAN_DECODER_UPSAMPLER_MAP:
            block_idx = _LEGACY_WAN_DECODER_UPSAMPLER_MAP[old_idx]
            return f"decoder.up_blocks.{block_idx}.upsamplers.0{suffix}"
    return key


def _is_legacy_wan_vae_checkpoint(shard_paths: list[str]) -> bool:
    if not shard_paths:
        return False
    from safetensors import safe_open

    with safe_open(shard_paths[0], framework="pt", device="cpu") as f:
        keys = set(f.keys())
    return "encoder.downsamples.0.residual.0.gamma" in keys and "encoder.down_blocks.0.norm1.gamma" not in keys


def load_pretrained_wan_vae_model(
    model_cls: Any,
    model_name_or_path: str,
    *,
    subfolder: str | None = None,
    config_file_name: str = "config.json",
    index_file_name: str = "diffusion_pytorch_model.safetensors.index.json",
    **kwargs: Any,
) -> Any:
    component_dir = os.path.join(model_name_or_path, subfolder) if subfolder is not None else model_name_or_path
    if not os.path.isdir(component_dir):
        return load_pretrained_component_model(
            model_cls,
            model_name_or_path,
            subfolder=subfolder,
            config_file_name=config_file_name,
            index_file_name=index_file_name,
            **kwargs,
        )

    shard_paths = sorted(glob.glob(os.path.join(component_dir, "*.safetensors")))
    if not _is_legacy_wan_vae_checkpoint(shard_paths):
        return load_pretrained_component_model(
            model_cls,
            model_name_or_path,
            subfolder=subfolder,
            config_file_name=config_file_name,
            index_file_name=index_file_name,
            **kwargs,
        )

    config_path = os.path.join(component_dir, config_file_name)
    if not os.path.exists(config_path):
        return load_pretrained_component_model(
            model_cls,
            model_name_or_path,
            subfolder=subfolder,
            config_file_name=config_file_name,
            index_file_name=index_file_name,
            **kwargs,
        )

    logger.info("Loading legacy Wan VAE checkpoint from %s", component_dir)
    with open(config_path) as f:
        config = json.load(f)

    torch_dtype = kwargs.get("torch_dtype")
    model = model_cls.from_config(config)

    from safetensors import safe_open

    state_dict = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[_map_legacy_wan_vae_key(key)] = f.get_tensor(key)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        logger.warning(
            "Legacy Wan VAE remap missing=%d unexpected=%d",
            len(missing_keys),
            len(unexpected_keys),
        )
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    if hasattr(model, "init_distributed"):
        model.init_distributed()
    return model


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
