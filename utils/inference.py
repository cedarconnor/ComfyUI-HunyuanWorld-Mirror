"""
Model loading, caching, and inference utilities.

Provides efficient model management and inference wrappers for HunyuanWorld-Mirror.
"""

import os
import torch
from pathlib import Path
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
        model_name: Model identifier, path, or filename
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

    # Resolve model path
    model_path = _resolve_model_path(model_name)

    # Create cache key based on actual path
    cache_key = f"{model_path}_{device}_{precision}"

    # Check cache
    if use_cache:
        cached_model = ModelCache.get(cache_key)
        if cached_model is not None:
            print(f"✓ Model loaded from cache: {os.path.basename(model_path)}")
            return cached_model, cache_key

    # Load model
    print(f"Loading HunyuanWorld-Mirror model from: {model_path}")
    print(f"Device: {device}, Precision: {precision}")

    try:
        # Load based on file type
        if model_path.endswith('.safetensors'):
            model = _load_safetensors_model(model_path, device, precision)
        elif model_path.endswith('.pt') or model_path.endswith('.pth'):
            model = _load_pytorch_model(model_path, device, precision)
        elif os.path.isdir(model_path):
            model = _load_from_directory(model_path, device, precision, cache_dir)
        else:
            # Try HuggingFace Hub
            model = _load_from_hub(model_name, device, precision, cache_dir)

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


def _resolve_model_path(model_name: str) -> str:
    """
    Resolve model name to actual file path.

    Checks in order:
    1. Direct path (if exists)
    2. ComfyUI/models/HunyuanWorld-Mirror/{model_name}
    3. ComfyUI/models/HunyuanWorld-Mirror/HunyuanWorld-Mirror.safetensors
    4. Returns model_name as-is (for HuggingFace Hub)
    """
    # If it's already a valid path, use it
    if os.path.exists(model_name):
        return os.path.abspath(model_name)

    # Try to find ComfyUI models directory
    # Work backwards from this file's location
    current_dir = Path(__file__).parent.parent  # ComfyUI-HunyuanWorld-Mirror directory

    # Look for ComfyUI root (should be parent of custom_nodes)
    if current_dir.parent.name == "custom_nodes":
        comfy_root = current_dir.parent.parent
        models_dir = comfy_root / "models" / "HunyuanWorld-Mirror"

        # Check various possible locations
        candidates = [
            models_dir / model_name,
            models_dir / f"{model_name}.safetensors",
            models_dir / "HunyuanWorld-Mirror.safetensors",
        ]

        for candidate in candidates:
            if candidate.exists():
                print(f"Found model at: {candidate}")
                return str(candidate)

    # Return as-is for HuggingFace Hub
    return model_name


def _load_safetensors_model(model_path: str, device: str, precision: str) -> Any:
    """Load model from safetensors file."""
    from safetensors.torch import load_file

    print(f"Loading from safetensors: {os.path.basename(model_path)}")

    # Load state dict
    state_dict = load_file(model_path)

    # Try to import model class
    try:
        from src.models.models.worldmirror import WorldMirror

        # Create model instance and load weights
        model = WorldMirror()
        model.load_state_dict(state_dict)

    except ImportError:
        # Fallback: Create a simple wrapper that holds the state dict
        print("Warning: WorldMirror class not found, loading weights only")
        print("For full functionality, install: https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror")

        class SafetensorsModelWrapper:
            """Simple wrapper for loaded safetensors weights."""
            def __init__(self, state_dict, device, precision):
                self.state_dict = state_dict
                self.device = device
                self.precision = precision

            def eval(self):
                return self

            def to(self, device):
                self.device = device
                return self

            def half(self):
                self.precision = "fp16"
                return self

            def bfloat16(self):
                self.precision = "bf16"
                return self

            def __call__(self, *args, **kwargs):
                raise NotImplementedError(
                    "Model weights loaded but WorldMirror class not available.\n"
                    "Please install: https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror"
                )

        model = SafetensorsModelWrapper(state_dict, device, precision)

    # Move to device
    if hasattr(model, 'to'):
        model = model.to(device)

    # Set precision
    if precision == "fp16" and hasattr(model, 'half'):
        model = model.half()
    elif precision == "bf16" and hasattr(model, 'bfloat16'):
        model = model.bfloat16()

    return model


def _load_pytorch_model(model_path: str, device: str, precision: str) -> Any:
    """Load model from PyTorch checkpoint file."""
    print(f"Loading from PyTorch checkpoint: {os.path.basename(model_path)}")

    checkpoint = torch.load(model_path, map_location=device)

    # Try to import model class
    try:
        from src.models.models.worldmirror import WorldMirror
        model = WorldMirror()

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint

    except ImportError:
        raise ImportError(
            "WorldMirror class not found. Please install:\n"
            "https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror"
        )

    model = model.to(device)

    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.bfloat16()

    return model


def _load_from_directory(model_path: str, device: str, precision: str, cache_dir: Optional[str]) -> Any:
    """Load model from directory (HuggingFace format)."""
    print(f"Loading from directory: {model_path}")

    try:
        from src.models.models.worldmirror import WorldMirror
        model = WorldMirror.from_pretrained(model_path)
    except ImportError:
        raise ImportError(
            "WorldMirror class not found. Please install:\n"
            "https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror"
        )

    model = model.to(device)

    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.bfloat16()

    return model


def _load_from_hub(model_name: str, device: str, precision: str, cache_dir: Optional[str]) -> Any:
    """Load model from HuggingFace Hub."""
    print(f"Loading from HuggingFace Hub: {model_name}")

    # Set cache directory if provided
    if cache_dir:
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir

    try:
        from src.models.models.worldmirror import WorldMirror
        model = WorldMirror.from_pretrained(model_name)
    except ImportError:
        raise ImportError(
            "WorldMirror class not found. Please install:\n"
            "https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror"
        )

    model = model.to(device)

    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.bfloat16()

    return model


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
