#!/usr/bin/env python3
"""
Benchmark script for YouTube Shorts generation optimization.

This script runs benchmarks to measure the performance improvements 
from our optimizations and produces comparison reports.
"""

import os
import logging
import time
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main benchmark runner function"""
    parser = argparse.ArgumentParser(description="Run benchmark tests for YouTube Shorts generation")
    parser.add_argument("--test", choices=["all", "keyframe", "parallel", "encoder"], 
                      default="all", help="Which test to run")
    parser.add_argument("--output", default="benchmark_results", 
                      help="Directory to save benchmark results")
    parser.add_argument("--clean", action="store_true", 
                      help="Clean previous benchmark results")
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Clean results if requested
    if args.clean and output_dir.exists():
        logger.info(f"Cleaning previous benchmark results in {output_dir}")
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    
    # Set temporary directory
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    os.environ["TEMP_DIR"] = str(temp_dir)
    
    # Run appropriate benchmarks
    if args.test in ["all", "keyframe"]:
        run_keyframe_benchmark(output_dir)
    
    if args.test in ["all", "parallel"]:
        run_parallel_benchmark(output_dir)
    
    if args.test in ["all", "encoder"]:
        run_encoder_benchmark(output_dir)
    
    # Generate summary report
    if args.test == "all":
        generate_summary(output_dir)
    
    logger.info("Benchmarks completed")

def run_keyframe_benchmark(output_dir):
    """Run benchmarks for keyframe animation system"""
    logger.info("Running keyframe animation benchmarks")
    
    try:
        from helper.benchmark import PerformanceBenchmark
        
        benchmark = PerformanceBenchmark(output_dir=str(output_dir / "keyframe"))
        result = benchmark.test_keyframe_vs_callable(duration=5, num_frames=500)
        
        # Save and plot results
        benchmark.save_results(filename="keyframe_benchmark_results.json")
        benchmark.plot_results(filename="keyframe_benchmark_plot.png")
        
        logger.info(f"Keyframe vs Callable speedup: {result['speedup']:.2f}x")
    except Exception as e:
        logger.error(f"Error running keyframe benchmark: {e}")

def run_parallel_benchmark(output_dir):
    """Run benchmarks for parallel task processing"""
    logger.info("Running parallel task benchmarks")
    
    try:
        from helper.benchmark import PerformanceBenchmark
        
        benchmark = PerformanceBenchmark(output_dir=str(output_dir / "parallel"))
        result = benchmark.test_parallel_tasks(num_tasks=32, task_time=0.1)
        
        # Save and plot results
        benchmark.save_results(filename="parallel_benchmark_results.json")
        benchmark.plot_results(filename="parallel_benchmark_plot.png")
        
        # Calculate maximum speedup
        max_speedup = max(result['speedup'].values())
        logger.info(f"Parallel task maximum speedup: {max_speedup:.2f}x")
    except Exception as e:
        logger.error(f"Error running parallel benchmark: {e}")

def run_encoder_benchmark(output_dir):
    """Run benchmarks for video encoding optimizations"""
    logger.info("Running encoder benchmarks")
    
    try:
        import tempfile
        from moviepy.editor import TextClip, CompositeVideoClip
        from helper.video_encoder import VideoEncoder
        
        # Create a sample clip for testing
        temp_dir = tempfile.mkdtemp(dir=output_dir)
        
        # Generate sample clip
        text_clip = TextClip(
            "Encoder Benchmark Test", 
            fontsize=70, 
            color='white',
            size=(1080, 1920),
            bg_color='purple',
            method='caption',
            align='center'
        ).set_duration(3)
        
        # Original MoviePy rendering (baseline)
        start_time = time.time()
        baseline_path = os.path.join(temp_dir, "baseline.mp4")
        text_clip.write_videofile(
            baseline_path,
            fps=30,
            codec="libx264",
            preset="medium"  # Default preset
        )
        baseline_time = time.time() - start_time
        baseline_size = os.path.getsize(baseline_path)
        
        # Optimized rendering
        start_time = time.time()
        optimized_path = os.path.join(temp_dir, "optimized.mp4")
        VideoEncoder.write_clip(
            text_clip,
            optimized_path,
            fps=30,
            is_final=True
        )
        optimized_time = time.time() - start_time
        optimized_size = os.path.getsize(optimized_path)
        
        # Calculate speedup
        speedup = baseline_time / optimized_time
        size_ratio = optimized_size / baseline_size
        
        # Save results
        results = {
            "baseline_time": baseline_time,
            "optimized_time": optimized_time,
            "speedup": speedup,
            "baseline_size": baseline_size,
            "optimized_size": optimized_size,
            "size_ratio": size_ratio,
            "timestamp": datetime.now().isoformat()
        }
        
        import json
        results_path = output_dir / "encoder" / "encoder_benchmark_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Encoder speedup: {speedup:.2f}x with size ratio {size_ratio:.2f}")
        
        # Close clips
        text_clip.close()
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        logger.error(f"Error running encoder benchmark: {e}")

def generate_summary(output_dir):
    """Generate summary report of all benchmarks"""
    try:
        import json
        from datetime import datetime
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {}
        }
        
        # Collect keyframe results
        keyframe_path = output_dir / "keyframe" / "keyframe_benchmark_results.json"
        if keyframe_path.exists():
            with open(keyframe_path) as f:
                keyframe_data = json.load(f)
                if "keyframe_vs_callable" in keyframe_data:
                    summary["benchmarks"]["keyframe"] = {
                        "speedup": keyframe_data["keyframe_vs_callable"]["speedup"]
                    }
        
        # Collect parallel results
        parallel_path = output_dir / "parallel" / "parallel_benchmark_results.json"
        if parallel_path.exists():
            with open(parallel_path) as f:
                parallel_data = json.load(f)
                if "parallel_tasks" in parallel_data:
                    speedups = parallel_data["parallel_tasks"]["speedup"]
                    max_speedup = max(float(v) for v in speedups.values())
                    summary["benchmarks"]["parallel"] = {
                        "max_speedup": max_speedup
                    }
        
        # Collect encoder results
        encoder_path = output_dir / "encoder" / "encoder_benchmark_results.json"
        if encoder_path.exists():
            with open(encoder_path) as f:
                encoder_data = json.load(f)
                summary["benchmarks"]["encoder"] = {
                    "speedup": encoder_data["speedup"],
                    "size_ratio": encoder_data["size_ratio"]
                }
        
        # Save summary
        summary_path = output_dir / "benchmark_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("=== BENCHMARK SUMMARY ===")
        if "keyframe" in summary["benchmarks"]:
            logger.info(f"Keyframe Animation: {summary['benchmarks']['keyframe']['speedup']:.2f}x faster")
        if "parallel" in summary["benchmarks"]:
            logger.info(f"Parallel Processing: {summary['benchmarks']['parallel']['max_speedup']:.2f}x faster")
        if "encoder" in summary["benchmarks"]:
            logger.info(f"Video Encoding: {summary['benchmarks']['encoder']['speedup']:.2f}x faster with size ratio {summary['benchmarks']['encoder']['size_ratio']:.2f}")
        logger.info("========================")
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")

if __name__ == "__main__":
    main() 