"""
Model loading, caching, and inference utilities.

Provides efficient model management and inference wrappers for HunyuanWorld-Mirror.
"""

import torch
from typing import Dict, Optional, Any, Tuple
from .memory import MemoryManager


class ModelCache:
    """Thread-safe model cache for ComfyUI."""

    _cache: Dict[str, Any] = {}

    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        """
        Get model from cache.

        Args:
            key: Cache key (typically model_name + device + precision)

        Returns:
            Cached model or None if not found
        """
        return cls._cache.get(key, None)

    @classmethod
    def set(cls, key: str, model: Any) -> None:
        """
        Store model in cache.

        Args:
            key: Cache key
            model: Model instance to cache
        """
        cls._cache[key] = model
        print(f"Model cached with key: {key}")

    @classmethod
    def clear(cls, key: Optional[str] = None) -> None:
        """
        Clear model from cache.

        Args:
            key: Specific key to clear, or None to clear all
        """
        if key is None:
            cls._cache.clear()
            print("Model cache cleared")
        elif key in cls._cache:
            del cls._cache[key]
            print(f"Model removed from cache: {key}")

    @classmethod
    def get_size(cls) -> int:
        """Get number of cached models."""
        return len(cls._cache)

    @classmethod
    def list_keys(cls) -> list:
        """Get list of all cache keys."""
        return list(cls._cache.keys())


class InferenceWrapper:
    """Wrapper for HunyuanWorld-Mirror model inference."""

    def __init__(
        self,
        model: Any,
        device: str = "cuda",
        precision: str = "fp32"
    ):
        """
        Initialize inference wrapper.

        Args:
            model: HunyuanWorld-Mirror model instance
            device: Device to run on ('cuda' or 'cpu')
            precision: Precision mode ('fp32', 'fp16', 'bf16')
        """
        self.model = model
        self.device = device
        self.precision = precision

        # Set model to eval mode
        self.model.eval()

    @torch.no_grad()
    def infer(
        self,
        images: torch.Tensor,
        condition: Optional[list] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on image sequence.

        Args:
            images: Input images in HWM format [1, N, 3, H, W]
            condition: Optional condition flags [pose, depth, intrinsic]
            **kwargs: Additional model arguments

        Returns:
            Dictionary of output tensors:
                - pts3d: 3D points [1, N, H, W, 3]
                - depth: Depth maps [1, N, H, W]
                - normals: Surface normals [1, N, H, W, 3]
                - conf: Confidence maps [1, N, H, W]
                - camera_poses: Camera poses [1, N, 4, 4]
                - camera_intrinsics: Intrinsics [1, N, 3, 3]
                - gaussian_params: Gaussian splatting parameters (dict)
        """
        # Move images to device
        images = images.to(self.device)

        # Apply precision conversion
        if self.precision == "fp16":
            images = images.half()
        elif self.precision == "bf16":
            images = images.bfloat16()

        # Run inference
        try:
            outputs = self.model(images, condition=condition, **kwargs)
        except Exception as e:
            print(f"Inference error: {e}")
            raise

        # Convert outputs back to fp32 for compatibility
        if self.precision in ["fp16", "bf16"]:
            outputs = {k: v.float() if isinstance(v, torch.Tensor) else v
                      for k, v in outputs.items()}

        return outputs

    def clear_memory(self) -> None:
        """Clear GPU memory after inference."""
        MemoryManager.clear_cache()

    def get_memory_stats(self) -> Optional[Dict[str, float]]:
        """Get current memory statistics."""
        return MemoryManager.get_memory_stats()


def load_model(
    model_name: str = "tencent/HunyuanWorld-Mirror",
    device: str = "auto",
    precision: str = "fp32",
    cache_dir: Optional[str] = None,
    use_cache: bool = True
) -> Tuple[Any, str]:
    """
    Load HunyuanWorld-Mirror model with caching.

    Args:
        model_name: Model identifier or path
        device: Target device ('auto', 'cuda', 'cpu')
        precision: Precision mode ('fp32', 'fp16', 'bf16')
        cache_dir: Custom cache directory for model files
        use_cache: Whether to use model cache

    Returns:
        Tuple of (model, cache_key)
    """
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create cache key
    cache_key = f"{model_name}_{device}_{precision}"

    # Check cache
    if use_cache:
        cached_model = ModelCache.get(cache_key)
        if cached_model is not None:
            print(f"✓ Model loaded from cache: {cache_key}")
            return cached_model, cache_key

    # Load model
    print(f"Loading HunyuanWorld-Mirror model from: {model_name}")
    print(f"Device: {device}, Precision: {precision}")

    try:
        # Set cache directory if provided
        if cache_dir:
            import os
            os.environ['HF_HOME'] = cache_dir
            os.environ['TRANSFORMERS_CACHE'] = cache_dir

        # Import model class
        # NOTE: This assumes the HunyuanWorld-Mirror model is properly installed
        # Users need to clone and install the official repository
        try:
            from src.models.models.worldmirror import WorldMirror
        except ImportError:
            raise ImportError(
                "HunyuanWorld-Mirror model not found. Please install it by:\n"
                "1. Clone: git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror\n"
                "2. Install: cd HunyuanWorld-Mirror && pip install -e .\n"
                "Or make sure the model code is in your Python path."
            )

        # Load model
        model = WorldMirror.from_pretrained(model_name)
        model = model.to(device)

        # Set precision
        if precision == "fp16":
            model = model.half()
        elif precision == "bf16":
            model = model.bfloat16()

        # Set to eval mode
        model.eval()

        # Cache the model
        if use_cache:
            ModelCache.set(cache_key, model)

        print(f"✓ Model loaded successfully")
        MemoryManager.print_memory_stats("After model loading - ")

        return model, cache_key

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


def prepare_model_inputs(
    images: torch.Tensor,
    camera_poses: Optional[torch.Tensor] = None,
    depth_maps: Optional[torch.Tensor] = None,
    intrinsics: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[list], Dict[str, Any]]:
    """
    Prepare inputs for model inference.

    Args:
        images: Input images [1, N, 3, H, W]
        camera_poses: Optional camera poses [N, 4, 4]
        depth_maps: Optional depth priors [N, H, W]
        intrinsics: Optional camera intrinsics [N, 3, 3]

    Returns:
        Tuple of (images, condition_flags, additional_inputs)
    """
    # Determine condition flags
    condition = None
    if camera_poses is not None or depth_maps is not None or intrinsics is not None:
        condition = [
            camera_poses is not None,  # pose condition
            depth_maps is not None,     # depth condition
            intrinsics is not None      # intrinsic condition
        ]

    # Prepare additional inputs
    additional_inputs = {}
    if camera_poses is not None:
        additional_inputs['camera_poses'] = camera_poses
    if depth_maps is not None:
        additional_inputs['depth_priors'] = depth_maps
    if intrinsics is not None:
        additional_inputs['intrinsics'] = intrinsics

    return images, condition, additional_inputs
