import os
import psutil
import logging
import platform
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SystemResources:
    """System resource analyzer for optimal parallel processing configuration."""

    def __init__(self):
        self.memory_info = self.get_memory_info()
        self.cpu_info = self.get_cpu_info()
        self.io_info = self.get_io_info()

        logger.info(f"System resources detected: {self.cpu_info['logical_cores']} CPU cores, "
                   f"{self.memory_info['total_gb']:.1f}GB RAM")

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            'total': mem.total,
            'total_gb': mem.total / (1024**3),
            'available': mem.available,
            'available_gb': mem.available / (1024**3),
            'percent_used': mem.percent,
            'swap_total': swap.total,
            'swap_total_gb': swap.total / (1024**3),
            'swap_free': swap.free,
            'swap_percent': swap.percent
        }

    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        return {
            'logical_cores': psutil.cpu_count(logical=True),
            'physical_cores': psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True),
            'current_load': psutil.cpu_percent(interval=0.1),
            'per_cpu_load': psutil.cpu_percent(interval=0.1, percpu=True)
        }

    def get_io_info(self) -> Dict[str, Any]:
        """Get I/O information for disk performance estimation."""
        try:
            # Get disk IO counters
            disk_io = psutil.disk_io_counters()
            # Get disk usage for the main disk
            disk_usage = psutil.disk_usage('/')

            return {
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count,
                'disk_usage_percent': disk_usage.percent,
                'disk_free_gb': disk_usage.free / (1024**3)
            }
        except:
            # Fallback if disk IO information is not available
            return {
                'disk_free_gb': os.statvfs('/').f_bavail * os.statvfs('/').f_frsize / (1024**3),
                'disk_usage_percent': 0,
                'read_bytes': 0,
                'write_bytes': 0,
                'read_count': 0,
                'write_count': 0
            }

    def optimize_workers(self, task_type='video_rendering', memory_per_worker_gb=1.0,
                         cpu_per_worker=1, reserved_memory_gb=1.0,
                         reserved_cpu_cores=1) -> Dict[str, Any]:
        """
        Calculate optimal number of workers based on available system resources.

        Args:
            task_type: Type of task ('video_rendering', 'audio_processing', etc.)
            memory_per_worker_gb: Estimated memory required pexr worker in GB
            cpu_per_worker: Number of CPU cores required per worker
            reserved_memory_gb: Memory to reserve for system in GB
            reserved_cpu_cores: CPU cores to reserve for system

        Returns:
            Dictionary with optimized settings
        """
        # Update real-time system resource information
        self.memory_info = self.get_memory_info()
        self.cpu_info = self.get_cpu_info()

        # Calculate memory-based worker limit
        available_memory_gb = self.memory_info['available_gb'] - reserved_memory_gb
        memory_based_limit = max(1, int(available_memory_gb / memory_per_worker_gb))
        logger.info(f"Memory-based worker limit: {memory_based_limit}")

        # Calculate CPU-based worker limit
        available_cpu_cores = self.cpu_info['logical_cores'] - reserved_cpu_cores
        cpu_based_limit = max(1, int(available_cpu_cores / cpu_per_worker))
        logger.info(f"CPU-based worker limit: {cpu_based_limit}")


        # Calculate IO-based adjustments (reduce workers on high IO load or low disk space)
        io_adjustment = 1.0
        if self.io_info['disk_usage_percent'] > 90:
            io_adjustment = 0.5  # Reduce workers if disk nearly full

        # Take the minimum of memory and CPU constraints
        worker_count = min(memory_based_limit, cpu_based_limit)
        worker_count = max(1, int(worker_count * io_adjustment))  # Apply IO adjustment
        worker_count = 3 #experiment with 3 workers
        logger.info(f"Optimized worker count: {worker_count}")

        # Task-specific optimizations
        if task_type == 'video_rendering':
            # For video rendering, configure FFmpeg thread allocation
            ffmpeg_threads = max(1, min(4, int(available_cpu_cores / worker_count)))
            logger.info(f"Optimized for video rendering: {worker_count} workers with {ffmpeg_threads} FFmpeg threads each")

            return {
                'worker_count': worker_count,
                'ffmpeg_threads': ffmpeg_threads,
                'memory_per_worker_gb': memory_per_worker_gb,
                'total_cpu_cores': self.cpu_info['logical_cores'],
                'available_memory_gb': available_memory_gb,
                'memory_limited': memory_based_limit < cpu_based_limit
            }
        else:
            # Generic configuration
            return {
                'worker_count': worker_count,
                'total_cpu_cores': self.cpu_info['logical_cores'],
                'available_memory_gb': available_memory_gb
            }

# Helper functions for easy access
def get_system_resources():
    """Get system resource information."""
    return SystemResources()

def optimize_workers_for_rendering(memory_per_task_gb=1.0):
    """Get optimized worker configuration for video rendering."""
    system = SystemResources()
    return system.optimize_workers(
        task_type='video_rendering',
        memory_per_worker_gb=memory_per_task_gb,
        cpu_per_worker=1,
        reserved_memory_gb=1.0,
        reserved_cpu_cores=1
    )
