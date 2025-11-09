"""
HunyuanWorld-Mirror ComfyUI Node Pack

Transform 2D images into 3D worlds using Tencent's HunyuanWorld-Mirror model.

Repository: https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror
"""

import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import all nodes
from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    WEB_DIRECTORY,
)

# Version info
__version__ = "1.0.0"
__author__ = "Cedar Connor"
__license__ = "Apache-2.0"

# Export for ComfyUI
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

# Print welcome message
print("\n" + "=" * 70)
print(" HunyuanWorld-Mirror ComfyUI Node Pack")
print(f" Version: {__version__}")
print(" Transform 2D images into 3D worlds")
print("=" * 70)
print(f" Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name in NODE_DISPLAY_NAME_MAPPINGS.values():
    print(f"   • {node_name}")
print("=" * 70 + "\n")

# Check for potential issues
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠ CUDA not available - inference will be slow on CPU")
except Exception as e:
    print(f"⚠ Warning: {e}")

print()
