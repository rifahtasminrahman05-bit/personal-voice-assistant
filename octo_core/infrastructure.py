# -*- coding: utf-8 -*-
"""System Intelligence & Infrastructure.

Async task scheduler, TTL cache, fault-tolerant execution (retry +
circuit breaker), resource manager and a lightweight performance profiler.
"""
import functools
import os
import queue
import sys
import threading
import time


class CacheManager:
    """Intelligent Caching Layer: TTL + LRU-bounded memoization."""

    def __init__(self, capacity=256, ttl=600.0):
        self.capacity = capacity
        self.ttl = ttl
        self._store = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, expires = item
            if expires < time.time():
                del self._store[key]
                return None
            return value

    def set(self, key, value):
        with self._lock:
            if len(self._store) >= self.capacity:
                oldest = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest]
            self._store[key] = (value, time.time() + self.ttl)

    def clear(self):
        with self._lock:
            self._store.clear()

    def __len__(self):
        return len(self._store)

    def cached(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            hit = self.get(key)
            if hit is not None:
                return hit
            value = fn(*args, **kwargs)
            self.set(key, value)
            return value
        return wrapper


def retry_with_backoff(fn, retries=3, base_delay=0.2, exceptions=(Exception,)):
    """Fault-Tolerant Architecture: exponential backoff retries."""
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except exceptions as e:
            last = e
            time.sleep(base_delay * (2 ** attempt))
    raise last


class CircuitBreaker:
    """Circuit breaker: fail fast after repeated failures, auto half-open reset."""

    def __init__(self, failure_threshold=3, reset_after=30.0):
        self.threshold = failure_threshold
        self.reset_after = reset_after
        self._failures = 0
        self._opened_at = None
        self._lock = threading.Lock()

    @property
    def state(self):
        with self._lock:
            if self._opened_at and time.time() - self._opened_at > self.reset_after:
                return "half-open"
            return "open" if self._opened_at else "closed"

    def call(self, fn, *args, **kwargs):
        if self.state == "open":
            raise RuntimeError("circuit open; call rejected")
        try:
            result = fn(*args, **kwargs)
            with self._lock:
                self._failures = 0
                self._opened_at = None
            return result
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= self.threshold:
                    self._opened_at = time.time()
            raise


class Scheduler:
    """Async Task Scheduler: background thread pool with a result queue."""

    def __init__(self, workers=2):
        self.workers = workers
        self._queue = queue.Queue()
        self._results = queue.Queue()
        self._threads = []
        self._started = False
        self._lock = threading.Lock()

    def _worker(self):
        while True:
            task = self._queue.get()
            if task is None:
                break
            fn, args, kwargs = task
            try:
                self._results.put(("ok", fn(*args, **kwargs)))
            except Exception as e:
                self._results.put(("error", str(e)))

    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
            for _ in range(self.workers):
                t = threading.Thread(target=self._worker, daemon=True)
                t.start()
                self._threads.append(t)

    def submit(self, fn, *args, **kwargs):
        self.start()
        self._queue.put((fn, args, kwargs))

    def poll(self):
        items = []
        while True:
            try:
                items.append(self._results.get_nowait())
            except queue.Empty:
                return items

    def shutdown(self):
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout=5)


class ResourceManager:
    """Intelligent Resource Manager: device pick + light memory reporting."""

    @staticmethod
    def pick_device():
        if torch_cuda_available():
            return "cuda"
        if torch_mps_available():
            return "mps"
        return "cpu"

    @staticmethod
    def report():
        try:
            import torch
            mem = torch.get_num_threads()
        except Exception:
            mem = 0
        device = "cuda" if torch_cuda_available() else \
            "mps" if torch_mps_available() else "cpu"
        return {"cpu_count": os.cpu_count(), "torch_threads": mem,
                "device": device, "os": sys.platform}


def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def torch_mps_available():
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


class Profiler:
    """Model Performance Profiler: wall time + rough peak RSS of a callable."""

    @staticmethod
    def measure(fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        rss_mb = 0.0
        try:
            import resource
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            pass
        return {"result": result, "latency_ms": round(elapsed_ms, 2),
                "peak_rss_mb": round(rss_mb, 1)}
