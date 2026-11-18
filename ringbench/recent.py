"""Small revision-keyed product cache with coalesced duplicate work."""
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import fields, is_dataclass
from sys import getsizeof
from threading import RLock
from time import monotonic

import numpy as np

MAX_PENDING = 16
CACHE_BYTES = 64 * 1024 * 1024


def _footprint(item, seen=None):
    if seen is None:
        seen = set()
    if id(item) in seen:
        return 0
    seen.add(id(item))
    own = getsizeof(item)
    if isinstance(item, np.ndarray):
        if isinstance(item.base, np.ndarray):
            return own + _footprint(item.base, seen)
        return max(own, item.nbytes)
    if isinstance(item, dict):
        return own + sum(_footprint(key, seen) + _footprint(value, seen) for key, value in item.items())
    if isinstance(item, (list, tuple, set)):
        return own + sum(_footprint(value, seen) for value in item)
    if is_dataclass(item):
        return own + sum(_footprint(getattr(item, field.name), seen) for field in fields(item))
    return own


class RecentProducts:
    def __init__(self, capacity=12, time_to_live=180, memory_limit=CACHE_BYTES):
        self.capacity = capacity
        self.time_to_live = time_to_live
        self._stored = OrderedDict()
        self._memory_limit = memory_limit
        self._memory_used = 0
        self._pending = {}
        self._mutex = RLock()

    def _get(self, key, now):
        entry = self._stored.get(key)
        if entry is None:
            return None
        until, value, used_bytes = entry
        if until <= now:
            self._drop(key)
            return None
        self._stored.move_to_end(key)
        return value

    def _drop(self, key):
        _, _, size = self._stored.pop(key)
        self._memory_used -= size

    def _finish(self, key, value):
        size = _footprint(value)
        if size > self._memory_limit:
            return
        with self._mutex:
            now = monotonic()
            for old_key, (expiry, _, _) in list(self._stored.items()):
                if expiry <= now:
                    self._drop(old_key)
            if key in self._stored:
                self._drop(key)
            self._stored[key] = (now + self.time_to_live, value, size)
            self._memory_used += size
            self._stored.move_to_end(key)
            while len(self._stored) > self.capacity or self._memory_used > self._memory_limit:
                self._drop(next(iter(self._stored)))

    def solve(self, key, build):
        with self._mutex:
            result = self._get(key, monotonic())
            if result is not None:
                return result
            running = self._pending.get(key)
            owner = running is None and len(self._pending) < MAX_PENDING
            if owner:
                running = Future()
                self._pending[key] = running
        if not owner:
            return running.result() if running is not None else build()
        try:
            result = build()
            self._finish(key, result)
            running.set_result(result)
            return result
        except BaseException as exc:
            running.set_exception(exc)
            raise
        finally:
            with self._mutex:
                self._pending.pop(key, None)

    def discard_project(self, project_id):
        with self._mutex:
            for key in [key for key in self._stored if key[0] == project_id]:
                self._drop(key)

    def status(self):
        with self._mutex:
            return {
                "cached_products": len(self._stored),
                "cached_bytes": self._memory_used,
                "running_products": len(self._pending),
            }
