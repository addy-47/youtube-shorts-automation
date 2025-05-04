"""
Benchmark utility for measuring performance improvements in the YouTube Shorts generator.

This module provides tools for testing the performance of different 
video generation approaches and measuring improvements.
"""

import time
import logging
import os
import sys
from pathlib import Path
import datetime
import tempfile
import json
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure basic logging when running as main script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """
    Benchmark for measuring YouTube Shorts generation performance.
    
    This class provides methods for testing different configurations and
    measuring their relative performance.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the benchmark.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        if output_dir is None:
            self.output_dir = Path.cwd() / "benchmark_results"
        else:
            self.output_dir = Path(output_dir)
            
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track benchmark results
        self.results = {}
        
    def test_parallel_renderer(self, num_clips=10, clip_duration=2, resolution=(1080, 1920)):
        """
        Test the performance of parallel rendering with different configurations.
        
        Args:
            num_clips: Number of clips to render
            clip_duration: Duration of each clip in seconds
            resolution: Resolution of clips as (width, height)
            
        Returns:
            Dictionary with benchmark results
        """
        from automation.parallel_renderer import render_clips_in_parallel
        from helper.keyframe_animation import KeyframeTrack, InterpolationType
        
        logger.info(f"Testing parallel renderer with {num_clips} clips of {clip_duration}s each")
        
        # Generate test clips
        clips = []
        for i in range(num_clips):
            # Create a simple text clip
            text_clip = TextClip(
                f"Test Clip {i+1}",
                fontsize=70,
                color='white',
                size=resolution,
                method='caption',
                align='center',
                stroke_color='black',
                stroke_width=2
            ).set_duration(clip_duration)
            
            # Add some callable functions for position to stress test serialization
            if i % 2 == 0:
                # For even-indexed clips, use callable position function
                position_func = lambda t: (resolution[0]//2, resolution[1]//2 + int(100 * np.sin(t*np.pi)))
                text_clip = text_clip.set_position(position_func)
            
            clips.append(text_clip)
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define output path
            output_path = os.path.join(temp_dir, "parallel_renderer_test.mp4")
            
            # Test with different numbers of processes
            process_counts = [1, 2, 4]
            timing_results = {}
            
            for num_processes in process_counts:
                logger.info(f"Testing with {num_processes} processes")
                
                # Time the rendering process
                start_time = time.time()
                render_clips_in_parallel(
                    clips, 
                    output_path, 
                    fps=30, 
                    num_processes=num_processes
                )
                elapsed_time = time.time() - start_time
                
                # Record results
                timing_results[f"{num_processes}_processes"] = elapsed_time
                logger.info(f"Rendered with {num_processes} processes in {elapsed_time:.2f}s")
                
                # Delete output file to ensure clean test for next run
                if os.path.exists(output_path):
                    os.unlink(output_path)
            
            # Calculate speedup factors
            baseline = timing_results["1_processes"]
            speedups = {
                k: baseline / v 
                for k, v in timing_results.items()
            }
            
            # Record results
            result = {
                "num_clips": num_clips,
                "clip_duration": clip_duration,
                "resolution": resolution,
                "timing": timing_results,
                "speedup": speedups
            }
            
            self.results["parallel_renderer"] = result
            return result
            
    def test_keyframe_vs_callable(self, duration=5, num_frames=150):
        """
        Test performance of keyframe-based animations vs callable functions.
        
        Args:
            duration: Duration of test animation in seconds
            num_frames: Number of frames to evaluate
            
        Returns:
            Dictionary with benchmark results
        """
        from helper.keyframe_animation import KeyframeTrack, Animation, InterpolationType
        
        logger.info(f"Testing keyframe vs callable performance with {num_frames} frames")
        
        # Create a test callable function
        def callable_position(t):
            return (
                500 + int(200 * np.sin(t * np.pi)),
                500 + int(100 * np.cos(t * np.pi * 2))
            )
            
        # Create equivalent keyframe track
        keyframe_track = KeyframeTrack("position", InterpolationType.LINEAR)
        # Sample the callable at regular intervals
        for i in range(20):
            t = i * duration / 19
            keyframe_track.add_keyframe(t, callable_position(t))
            
        # Convert keyframe track to callable for comparison
        keyframe_callable = keyframe_track.to_callable()
        
        # Compare performance
        callable_times = []
        keyframe_times = []
        
        # Test callable function performance
        start_time = time.time()
        for frame in range(num_frames):
            t = frame * duration / (num_frames - 1)
            result = callable_position(t)
        callable_total = time.time() - start_time
        
        # Test keyframe function performance
        start_time = time.time()
        for frame in range(num_frames):
            t = frame * duration / (num_frames - 1)
            result = keyframe_callable(t)
        keyframe_total = time.time() - start_time
        
        # Record results
        result = {
            "duration": duration,
            "num_frames": num_frames,
            "callable_time": callable_total,
            "keyframe_time": keyframe_total,
            "speedup": callable_total / keyframe_total if keyframe_total > 0 else 0
        }
        
        self.results["keyframe_vs_callable"] = result
        logger.info(f"Callable time: {callable_total:.4f}s, Keyframe time: {keyframe_total:.4f}s")
        logger.info(f"Keyframe speedup: {result['speedup']:.2f}x")
        
        return result
        
    def test_parallel_tasks(self, num_tasks=20, task_time=0.1):
        """
        Test performance of parallel task processor.
        
        Args:
            num_tasks: Number of tasks to process
            task_time: Simulated time for each task in seconds
            
        Returns:
            Dictionary with benchmark results
        """
        from helper.parallel_tasks import ParallelTaskManager
        
        logger.info(f"Testing parallel tasks with {num_tasks} tasks of {task_time}s each")
        
        # Create a simple task function that sleeps for task_time
        def task_func(task_id, sleep_time):
            time.sleep(sleep_time)
            return f"Task {task_id} completed"
            
        # Test sequential execution
        sequential_start = time.time()
        sequential_results = {}
        for i in range(num_tasks):
            sequential_results[f"task_{i}"] = task_func(i, task_time)
        sequential_time = time.time() - sequential_start
        
        # Test parallel execution with different numbers of workers
        worker_counts = [2, 4, 8, 16]
        parallel_times = {}
        
        for workers in worker_counts:
            # Create task map
            task_map = {}
            for i in range(num_tasks):
                task_name = f"task_{i}"
                task_map[task_name] = (task_func, (i, task_time), {})
                
            # Execute tasks in parallel and time it
            task_manager = ParallelTaskManager(max_workers=workers)
            parallel_start = time.time()
            parallel_results = task_manager.execute_tasks(task_map)
            parallel_time = time.time() - parallel_start
            
            parallel_times[f"{workers}_workers"] = parallel_time
            logger.info(f"Parallel execution with {workers} workers: {parallel_time:.2f}s")
            
        # Calculate speedups
        speedups = {
            k: sequential_time / v 
            for k, v in parallel_times.items()
        }
        
        # Record results
        result = {
            "num_tasks": num_tasks,
            "task_time": task_time,
            "sequential_time": sequential_time,
            "parallel_times": parallel_times,
            "speedup": speedups
        }
        
        self.results["parallel_tasks"] = result
        logger.info(f"Sequential time: {sequential_time:.2f}s")
        logger.info(f"Maximum speedup: {max(speedups.values()):.2f}x")
        
        return result
        
    def save_results(self, filename=None):
        """
        Save benchmark results to file.
        
        Args:
            filename: Optional filename to save results to
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        # Add timestamp to results
        self.results["timestamp"] = datetime.datetime.now().isoformat()
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Saved benchmark results to {output_path}")
        
    def plot_results(self, filename=None):
        """
        Plot benchmark results.
        
        Args:
            filename: Optional filename to save plot to
        """
        if not self.results:
            logger.warning("No benchmark results to plot")
            return
            
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_plot_{timestamp}.png"
            
        output_path = self.output_dir / filename
        
        fig, axs = plt.subplots(1, len(self.results), figsize=(15, 5))
        
        # Handle case with only one result
        if len(self.results) == 1:
            axs = [axs]
            
        for i, (name, result) in enumerate(self.results.items()):
            if name == "timestamp":
                continue
                
            ax = axs[i]
            
            if name == "parallel_renderer":
                # Plot parallel renderer results
                processes = [int(k.split('_')[0]) for k in result["timing"].keys()]
                times = list(result["timing"].values())
                
                ax.bar(processes, times)
                ax.set_xlabel("Number of Processes")
                ax.set_ylabel("Render Time (s)")
                ax.set_title("Parallel Renderer Performance")
                
                # Add speedup annotations
                for j, p in enumerate(processes):
                    speedup = result["speedup"].get(f"{p}_processes", 1.0)
                    ax.annotate(
                        f"{speedup:.1f}x",
                        (p, times[j]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center'
                    )
                    
            elif name == "keyframe_vs_callable":
                # Plot keyframe vs callable results
                methods = ["Callable", "Keyframe"]
                times = [result["callable_time"], result["keyframe_time"]]
                
                ax.bar(methods, times)
                ax.set_ylabel("Execution Time (s)")
                ax.set_title("Keyframe vs Callable Performance")
                
                # Add speedup annotation
                ax.annotate(
                    f"{result['speedup']:.1f}x faster",
                    (1, times[1]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )
                
            elif name == "parallel_tasks":
                # Plot parallel tasks results
                workers = [int(k.split('_')[0]) for k in result["parallel_times"].keys()]
                workers.insert(0, 1)  # Add sequential as "1 worker"
                
                times = list(result["parallel_times"].values())
                times.insert(0, result["sequential_time"])
                
                ax.bar(workers, times)
                ax.set_xlabel("Number of Workers")
                ax.set_ylabel("Execution Time (s)")
                ax.set_title("Parallel Tasks Performance")
                
                # Add speedup annotations
                for j, w in enumerate(workers):
                    if j == 0:  # Sequential, no speedup
                        speedup = 1.0
                    else:
                        speedup = result["speedup"].get(f"{w}_workers", 1.0)
                        
                    ax.annotate(
                        f"{speedup:.1f}x",
                        (w, times[j]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center'
                    )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved benchmark plot to {output_path}")
        
def run_full_benchmark():
    """Run a full set of benchmarks and save results"""
    benchmark = PerformanceBenchmark()
    
    # Test keyframe vs callable performance
    benchmark.test_keyframe_vs_callable()
    
    # Test parallel tasks performance
    benchmark.test_parallel_tasks()
    
    # Test parallel renderer if available
    # Use fewer clips and shorter duration for faster testing
    benchmark.test_parallel_renderer(num_clips=5, clip_duration=1)
    
    # Save and plot results
    benchmark.save_results()
    benchmark.plot_results()
    
if __name__ == "__main__":
    run_full_benchmark() 