"""
Parallel task processing for YouTube Shorts automation.
Handles concurrent execution of independent tasks like audio processing, 
background fetching, and text clip generation.
"""

import os
import logging
import concurrent.futures
from functools import partial
import time
import traceback

logger = logging.getLogger(__name__)

class ParallelTaskManager:
    """
    Manages parallel execution of independent tasks.
    
    This class handles concurrent processing of tasks that don't depend on
    each other, like generating audio, fetching backgrounds, and creating text clips.
    """
    
    def __init__(self, max_workers=None):
        """
        Initialize the task manager.
        
        Args:
            max_workers: Maximum number of worker threads/processes
        """
        # If max_workers not specified, use a reasonable default
        if max_workers is None:
            import multiprocessing
            max_workers = min(multiprocessing.cpu_count() + 4, 16)
            
        self.max_workers = max_workers
        self.results = {}
        self.errors = {}
        
    def execute_tasks(self, task_map):
        """
        Execute tasks in parallel.
        
        Args:
            task_map: Dictionary mapping task names to (function, args, kwargs) tuples
            
        Returns:
            Dictionary mapping task names to results
        """
        start_time = time.time()
        logger.info(f"Starting parallel execution of {len(task_map)} tasks")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task_name, (func, args, kwargs) in task_map.items():
                logger.info(f"Submitting task: {task_name}")
                future = executor.submit(self._execute_task, task_name, func, args, kwargs)
                future_to_task[future] = task_name
                
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    self.results[task_name] = result
                    logger.info(f"Task completed: {task_name}")
                except Exception as e:
                    self.errors[task_name] = str(e)
                    logger.error(f"Error in task {task_name}: {e}")
                    logger.debug(traceback.format_exc())
        
        total_time = time.time() - start_time
        logger.info(f"Completed parallel execution in {total_time:.2f}s. "
                    f"Successful: {len(self.results)}, Failed: {len(self.errors)}")
        
        return self.results
    
    def _execute_task(self, task_name, func, args, kwargs):
        """
        Execute a single task and handle exceptions.
        
        Args:
            task_name: Name of the task for logging
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
        """
        try:
            start_time = time.time()
            logger.info(f"Starting task: {task_name}")
            
            result = func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            logger.info(f"Task {task_name} completed in {elapsed:.2f}s")
            
            return result
        except Exception as e:
            logger.error(f"Error in task {task_name}: {e}")
            logger.debug(traceback.format_exc())
            raise

def fetch_backgrounds_parallel(fetch_func, queries, max_workers=None):
    """
    Fetch background images or videos in parallel.
    
    Args:
        fetch_func: Function to fetch a single background
        queries: List of queries to fetch backgrounds for
        max_workers: Maximum number of worker threads
        
    Returns:
        Dictionary mapping indices to fetched backgrounds
    """
    task_manager = ParallelTaskManager(max_workers=max_workers)
    
    # Create task map
    task_map = {}
    for i, query in enumerate(queries):
        task_name = f"fetch_background_{i}"
        task_map[task_name] = (fetch_func, (query,), {})
    
    # Execute tasks
    results = task_manager.execute_tasks(task_map)
    
    # Remap results to be indexed by original position
    indexed_results = {}
    for task_name, result in results.items():
        # Extract index from task name (fetch_background_X)
        try:
            idx = int(task_name.split('_')[-1])
            indexed_results[idx] = result
        except (ValueError, IndexError):
            logger.warning(f"Could not extract index from task name: {task_name}")
    
    return indexed_results

def generate_audio_clips_parallel(generate_func, script_sections, max_workers=None):
    """
    Generate audio clips in parallel.
    
    Args:
        generate_func: Function to generate a single audio clip
        script_sections: List of script sections to generate audio for
        max_workers: Maximum number of worker threads
        
    Returns:
        Dictionary mapping indices to generated audio clips
    """
    task_manager = ParallelTaskManager(max_workers=max_workers)
    
    # Create task map
    task_map = {}
    for i, section in enumerate(script_sections):
        task_name = f"generate_audio_{i}"
        # Each section should be a dict with at least 'text' and optionally 'voice_style'
        text = section.get('text', '')
        voice_style = section.get('voice_style', 'neutral')
        
        task_map[task_name] = (generate_func, (text, voice_style), {})
    
    # Execute tasks
    results = task_manager.execute_tasks(task_map)
    
    # Remap results to be indexed by original position
    indexed_results = {}
    for task_name, result in results.items():
        # Extract index from task name (generate_audio_X)
        try:
            idx = int(task_name.split('_')[-1])
            indexed_results[idx] = result
        except (ValueError, IndexError):
            logger.warning(f"Could not extract index from task name: {task_name}")
    
    return indexed_results

def generate_text_clips_parallel(generate_func, script_sections, max_workers=None, **kwargs):
    """
    Generate text clips in parallel.
    
    Args:
        generate_func: Function to generate a single text clip
        script_sections: List of script sections to generate text clips for
        max_workers: Maximum number of worker threads
        **kwargs: Additional keyword arguments to pass to generate_func
        
    Returns:
        Dictionary mapping indices to generated text clips
    """
    task_manager = ParallelTaskManager(max_workers=max_workers)
    
    # Create task map
    task_map = {}
    for i, section in enumerate(script_sections):
        task_name = f"generate_text_{i}"
        text = section.get('text', '')
        
        # Create a partial function to include the additional kwargs
        partial_func = partial(generate_func, **kwargs)
        
        task_map[task_name] = (partial_func, (text, i), {})
    
    # Execute tasks
    results = task_manager.execute_tasks(task_map)
    
    # Remap results to be indexed by original position
    indexed_results = {}
    for task_name, result in results.items():
        # Extract index from task name (generate_text_X)
        try:
            idx = int(task_name.split('_')[-1])
            indexed_results[idx] = result
        except (ValueError, IndexError):
            logger.warning(f"Could not extract index from task name: {task_name}")
    
    return indexed_results 