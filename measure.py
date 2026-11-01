"""Process-resident peak memory sampling for reconstruction runs."""

import threading
import time


class PeakRSS:
    def __init__(self, interval_s: float = 0.02):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("Install psutil to measure native process memory") from exc
        self.process = psutil.Process()
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            info = self.process.memory_info()
            self.peak_bytes = max(self.peak_bytes, getattr(info, "peak_wset", info.rss))
            time.sleep(self.interval_s)
        info = self.process.memory_info()
        self.peak_bytes = max(self.peak_bytes, getattr(info, "peak_wset", info.rss))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
