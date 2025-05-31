import os

from concurrent.futures import ThreadPoolExecutor


class ThreadPool:
    _instance = None
    _executor: ThreadPoolExecutor

    def __new__(cls, max_workers=os.cpu_count()):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(max_workers)
        return cls._instance

    def _initialize(self, max_workers):
        """Initialize the thread pool executor."""
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit_task(self, fn, *args, **kwargs):
        """Submit a task to the thread pool."""
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True):
        """Shutdown the thread pool."""
        self.executor.shutdown(wait=wait)
        self._instance = None
