import os
import time
import logging
import concurrent.futures
from functools import partial
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents a task to be executed with its dependencies"""
    name: str
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    dependencies: List[str] = None
    result: Any = None
    completed: bool = False
    started: bool = False

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = []

class ParallelTaskExecutor:
    """Handles parallel execution of interdependent tasks"""

    def __init__(self, max_workers: int = None):
        """
        Initialize the task executor

        Args:
            max_workers (int): Maximum number of concurrent workers
        """
        self.tasks = {}
        self.max_workers = max_workers or min(32, os.cpu_count() * 2)
        self.results = {}

    def add_task(self, name: str, func: Callable, args: tuple = (), kwargs: dict = None,
                dependencies: List[str] = None) -> None:
        """
        Add a task to be executed

        Args:
            name (str): Unique name for the task
            func (Callable): Function to execute
            args (tuple): Positional arguments for the function
            kwargs (dict): Keyword arguments for the function
            dependencies (List[str]): Names of tasks that must complete before this one
        """
        self.tasks[name] = Task(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            dependencies=dependencies or []
        )

    def _can_execute(self, task_name: str) -> bool:
        """Check if a task can be executed based on its dependencies"""
        task = self.tasks[task_name]
        if task.started or task.completed:
            return False

        for dep in task.dependencies:
            if dep not in self.tasks:
                logger.warning(f"Dependency '{dep}' for task '{task_name}' not found")
                return False
            if not self.tasks[dep].completed:
                return False

        return True

    def _prepare_task_kwargs(self, task: Task) -> dict:
        """Prepare task kwargs by injecting dependency results if needed"""
        kwargs = task.kwargs.copy()

        # Add results from dependencies as kwargs if their names match
        for dep in task.dependencies:
            if dep in self.results:
                # If dependency name matches a parameter name, inject the result
                if dep in kwargs:
                    kwargs[dep] = self.results[dep]

        return kwargs

    def execute(self) -> Dict[str, Any]:
        """
        Execute all tasks respecting dependencies

        Returns:
            Dict[str, Any]: Results of all tasks
        """
        start_time = time.time()
        pending_tasks = set(self.tasks.keys())
        self.results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            while pending_tasks:
                # Find tasks that can be executed
                ready_tasks = [t for t in pending_tasks if self._can_execute(t)]

                if not ready_tasks and not futures:
                    # No tasks can be executed and no tasks are running
                    # This means we have a deadlock or all tasks are done
                    if pending_tasks:
                        logger.error(f"Deadlock detected. Pending tasks: {pending_tasks}")
                        break

                # Submit ready tasks
                for task_name in ready_tasks:
                    task = self.tasks[task_name]
                    task.started = True
                    logger.info(f"Starting task: {task_name}")

                    # Prepare kwargs with dependency results
                    kwargs = self._prepare_task_kwargs(task)

                    # Submit the task
                    future = executor.submit(task.func, *task.args, **kwargs)
                    futures[future] = task_name
                    pending_tasks.remove(task_name)

                # Process completed futures
                done_futures = []
                for future in concurrent.futures.as_completed(futures.keys()):
                    task_name = futures[future]
                    try:
                        result = future.result()
                        self.results[task_name] = result
                        self.tasks[task_name].result = result
                        self.tasks[task_name].completed = True
                        logger.info(f"Completed task: {task_name}")
                    except Exception as e:
                        logger.error(f"Task '{task_name}' failed: {str(e)}")
                        # Mark task as completed even if it failed
                        self.tasks[task_name].completed = True
                        self.results[task_name] = None

                    done_futures.append(future)

                # Remove completed futures
                for future in done_futures:
                    del futures[future]

                # Small sleep to prevent CPU spinning
                if not ready_tasks and futures:
                    time.sleep(0.01)

        # Check if all tasks were completed
        incomplete = [t for t in self.tasks if not self.tasks[t].completed]
        if incomplete:
            logger.warning(f"Tasks not completed: {incomplete}")

        total_time = time.time() - start_time
        logger.info(f"All tasks executed in {total_time:.2f} seconds")

        return self.results

# Helper functions for common parallel task patterns
def process_in_parallel(items, process_func, max_workers=None, executor_cls=concurrent.futures.ThreadPoolExecutor):
    """
    Process a list of items in parallel

    Args:
        items (list): Items to process
        process_func (callable): Function to process each item
        max_workers (int): Maximum number of workers
        executor_cls: Executor class to use (ThreadPoolExecutor or ProcessPoolExecutor)

    Returns:
        list: Results in the same order as input items
    """
    if not items:
        return []

    workers = max_workers or min(len(items), os.cpu_count() * 2)
    results = []

    with executor_cls(max_workers=workers) as executor:
        futures = [executor.submit(process_func, item) for item in items]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel processing: {str(e)}")

    return results

def process_items_with_index(items, process_func, max_workers=None,
                            executor_cls=concurrent.futures.ThreadPoolExecutor):
    """
    Process items in parallel and return results with their original indices

    Args:
        items (list): Items to process
        process_func (callable): Function to process each item
        max_workers (int): Maximum number of workers
        executor_cls: Executor class to use (ThreadPoolExecutor or ProcessPoolExecutor)

    Returns:
        list: Results in the same order as input items
    """
    if not items:
        return []

    workers = max_workers or min(len(items), os.cpu_count() * 2)
    results = [None] * len(items)

    def process_with_index(idx, item):
        try:
            result = process_func(item)
            return idx, result
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            return idx, None

    with executor_cls(max_workers=workers) as executor:
        futures = [executor.submit(process_with_index, i, item) for i, item in enumerate(items)]
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                logger.error(f"Error in parallel processing: {str(e)}")

    return results
