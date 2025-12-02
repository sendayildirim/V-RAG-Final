import os
from dataclasses import dataclass

import psutil
import torch

MB = 1024 * 1024


@dataclass
class MemorySnapshot:
    """Holds intermediate memory readings for easier logging."""

    current_mb: float      # Total (CPU + GPU)
    peak_mb: float         # Total peak
    delta_mb: float        # Total delta
    cpu_mb: float          # CPU only
    gpu_mb: float          # GPU only


class PeakMemoryMonitor:
    """
    Tracks CPU RAM (RSS) + GPU memory and reports total usage.

    Instead of comparing only start/end memory (which can become negative if tensors are
    freed), we keep the highest total seen during the measurement window.

    CPU: psutil RSS (process memory)
    GPU: torch.cuda.memory_allocated() if available
    Total: CPU + GPU
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.has_gpu = torch.cuda.is_available()
        self.initial_mb = self._current_total_memory()
        self.peak_mb = self.initial_mb

    def _current_cpu_memory(self) -> float:
        """Return current CPU RSS in MB."""
        return self.process.memory_info().rss / MB

    def _current_gpu_memory(self) -> float:
        """Return current GPU allocated memory in MB."""
        if self.has_gpu:
            return torch.cuda.memory_allocated() / MB
        return 0.0

    def _current_total_memory(self) -> float:
        """Return current total (CPU + GPU) memory in MB."""
        return self._current_cpu_memory() + self._current_gpu_memory()

    def record(self) -> MemorySnapshot:
        """Capture current total (CPU + GPU) memory and update the peak if necessary."""
        cpu_mb = self._current_cpu_memory()
        gpu_mb = self._current_gpu_memory()
        total_mb = cpu_mb + gpu_mb

        if total_mb > self.peak_mb:
            self.peak_mb = total_mb

        return MemorySnapshot(
            current_mb=total_mb,
            peak_mb=self.peak_mb,
            delta_mb=max(0.0, self.peak_mb - self.initial_mb),
            cpu_mb=cpu_mb,
            gpu_mb=gpu_mb
        )

    @property
    def memory_used_mb(self) -> float:
        """Difference between initial memory and the recorded peak."""
        return max(0.0, self.peak_mb - self.initial_mb)

    def summary(self) -> dict:
        """Return a serializable summary for logging or persistence."""
        snapshot = self.record()
        return {
            'initial_memory_mb': self.initial_mb,
            'peak_memory_mb': self.peak_mb,
            'memory_used_mb': self.memory_used_mb,
            'cpu_memory_mb': snapshot.cpu_mb,
            'gpu_memory_mb': snapshot.gpu_mb
        }


if __name__ == "__main__":
    # Test
    print("PeakMemoryMonitor Test (CPU + GPU)")

    monitor = PeakMemoryMonitor()
    print(f"GPU available: {monitor.has_gpu}")
    print(f"Initial total memory: {monitor.initial_mb:.2f} MB")

    snapshot = monitor.record()
    print(f"\nSnapshot:")
    print(f"  CPU: {snapshot.cpu_mb:.2f} MB")
    print(f"  GPU: {snapshot.gpu_mb:.2f} MB")
    print(f"  Total: {snapshot.current_mb:.2f} MB")

    summary = monitor.summary()
    print(f"\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
