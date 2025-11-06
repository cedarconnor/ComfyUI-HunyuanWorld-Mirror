"""
Memory management utilities for GPU and system memory.

Handles CUDA memory caching, cleanup, and estimation for efficient inference.
"""

import gc
import torch
from typing import Dict, Optional, Tuple


class MemoryManager:
    """Manage GPU and system memory for efficient inference."""

    @staticmethod
    def clear_cache() -> None:
        """
        Clear CUDA cache and run garbage collection.

        Call this after inference to free up GPU memory.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def get_memory_stats() -> Optional[Dict[str, float]]:
        """
        Get current GPU memory usage statistics.

        Returns:
            Dictionary with memory stats in GB, or None if CUDA unavailable.
            Keys: 'allocated_gb', 'reserved_gb', 'free_gb', 'total_gb'
        """
        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()

        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        free = total - allocated

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'total_gb': total,
        }

    @staticmethod
    def print_memory_stats(prefix: str = "") -> None:
        """
        Print current memory statistics to console.

        Args:
            prefix: String to prepend to output
        """
        stats = MemoryManager.get_memory_stats()

        if stats is None:
            print(f"{prefix}CUDA not available")
            return

        print(f"{prefix}GPU Memory: "
              f"{stats['allocated_gb']:.2f}GB allocated, "
              f"{stats['free_gb']:.2f}GB free, "
              f"{stats['total_gb']:.2f}GB total")

    @staticmethod
    def estimate_sequence_memory(
        num_frames: int,
        height: int,
        width: int,
        precision: str = "fp32"
    ) -> float:
        """
        Estimate memory required for processing a sequence.

        Args:
            num_frames: Number of frames in sequence
            height: Frame height in pixels
            width: Frame width in pixels
            precision: Precision mode ('fp32', 'fp16', 'bf16')

        Returns:
            Estimated memory usage in GB
        """
        bytes_per_element = {
            'fp32': 4,
            'fp16': 2,
            'bf16': 2,
        }.get(precision, 4)

        # Input images: [1, N, 3, H, W]
        input_size = num_frames * 3 * height * width * bytes_per_element

        # Rough estimate for outputs (multiple outputs: depth, normals, points, etc.)
        # Assume ~5x input size for all outputs combined
        output_multiplier = 5

        # Model parameters and activations (rough estimate ~4GB)
        model_overhead = 4 * (1024 ** 3)

        total_bytes = input_size * (1 + output_multiplier) + model_overhead
        total_gb = total_bytes / (1024 ** 3)

        return total_gb

    @staticmethod
    def check_memory_available(
        required_gb: float,
        safety_margin_gb: float = 2.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if enough GPU memory is available.

        Args:
            required_gb: Required memory in GB
            safety_margin_gb: Additional safety margin in GB

        Returns:
            Tuple of (is_available, message)
            - is_available: True if enough memory
            - message: Description of memory status
        """
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        stats = MemoryManager.get_memory_stats()

        if stats is None:
            return False, "Could not get memory stats"

        available_gb = stats['free_gb'] - safety_margin_gb

        if available_gb >= required_gb:
            return True, f"Sufficient memory: {available_gb:.2f}GB available, {required_gb:.2f}GB required"
        else:
            return False, f"Insufficient memory: {available_gb:.2f}GB available, {required_gb:.2f}GB required"

    @staticmethod
    def reset_peak_memory_stats() -> None:
        """Reset peak memory statistics for tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def get_peak_memory() -> Optional[float]:
        """
        Get peak memory allocated since last reset.

        Returns:
            Peak memory in GB, or None if CUDA unavailable
        """
        if not torch.cuda.is_available():
            return None

        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return peak

    @staticmethod
    def optimize_memory_for_inference() -> None:
        """
        Optimize memory settings for inference.

        Sets PyTorch memory management flags for better inference performance.
        """
        if torch.cuda.is_available():
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except AttributeError:
                pass

            # Set memory allocator config for better fragmentation handling
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)
            except Exception:
                pass


class MemoryMonitor:
    """Context manager for monitoring memory usage."""

    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialize memory monitor.

        Args:
            name: Name of operation being monitored
            verbose: Print memory stats
        """
        self.name = name
        self.verbose = verbose
        self.start_allocated = 0
        self.peak_allocated = 0

    def __enter__(self):
        """Enter context - record starting memory."""
        MemoryManager.reset_peak_memory_stats()
        MemoryManager.clear_cache()

        if torch.cuda.is_available():
            self.start_allocated = torch.cuda.memory_allocated() / (1024 ** 3)

        if self.verbose:
            print(f"[{self.name}] Starting memory: {self.start_allocated:.2f}GB")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - report memory usage."""
        if torch.cuda.is_available():
            end_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            peak = MemoryManager.get_peak_memory()

            if self.verbose:
                print(f"[{self.name}] Ending memory: {end_allocated:.2f}GB")
                print(f"[{self.name}] Peak memory: {peak:.2f}GB")
                print(f"[{self.name}] Memory change: {end_allocated - self.start_allocated:+.2f}GB")

            self.peak_allocated = peak

        MemoryManager.clear_cache()


def estimate_and_check_memory(
    num_frames: int,
    height: int,
    width: int,
    precision: str = "fp32",
    verbose: bool = True
) -> bool:
    """
    Estimate memory requirements and check availability.

    Args:
        num_frames: Number of frames
        height: Frame height
        width: Frame width
        precision: Precision mode
        verbose: Print details

    Returns:
        True if sufficient memory available
    """
    required = MemoryManager.estimate_sequence_memory(num_frames, height, width, precision)

    if verbose:
        print(f"Estimated memory requirement: {required:.2f}GB for {num_frames} frames at {height}x{width}")

    available, message = MemoryManager.check_memory_available(required)

    if verbose:
        print(message)

    return available
