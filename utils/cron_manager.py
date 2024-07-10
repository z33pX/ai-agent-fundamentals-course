from typing import List, Tuple, Dict, Any, Callable

import threading
import logging
import time

# Define a Task type for better clarity in function signatures.
Task = Tuple[Callable, Dict[str, Any], int]


class CronManager:
    def __init__(self) -> None:
        """
        Initialize the CronManager instance, which manages scheduled tasks.

        Attributes:
            tasks (List[Task]): A list of tasks where each task is a tuple containing
                                a callable function, a payload dictionary, and an interval in minutes.
            threads (List[threading.Thread]): A list of threads corresponding to the tasks being executed.
        """
        self.tasks: List[Task] = []
        self.threads: List[threading.Thread] = []

    def add_task(self, func: Callable, payload: Dict[str, Any], interval: int) -> None:
        """
        Add a new task to the cron manager for periodic execution.

        Args:
            func (Callable): The function to be executed periodically.
            payload (Dict[str, Any]): A dictionary containing arguments for the function `func`.
            interval (int): Time interval in minutes at which the function should be executed.

        Example:
            (summarize_emails, {"context": None, "nr_of_emails": 3}, 5)
            Here `summarize_emails` is a callable, and its payload includes necessary information,
            executed every 5 minutes.
        """
        self.tasks.append((func, payload, interval))

    def run_task(self, task: Task) -> None:
        """
        Execute the given task repeatedly according to its interval.

        Args:
            task (Task): The task to execute, containing the function, payload, and interval.

        Note:
            This method uses an infinite loop to repeatedly execute the task
            every `interval` minutes. Ensure this method is always run in a separate thread to avoid blocking.
        """
        while True:
            logging.info(f"Running task {task[0].__name__}.")
            task[0](**task[1])
            time.sleep(
                task[2] * 60
            )  # Convert minutes to seconds for the sleep function.

    def run(self, block: bool = False) -> None:
        """
        Start all scheduled tasks in separate daemon threads.

        Note:
            Each task is run in its own thread to allow simultaneous execution.
            Threads are set as daemons so they will automatically terminate when the main program exits.
        """
        for task in self.tasks:
            thread = threading.Thread(target=self.run_task, args=(task,), daemon=True)
            self.threads.append(thread)
            thread.start()

        if block:
            for thread in self.threads:
                thread.join()
