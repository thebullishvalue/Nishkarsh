"""
Nishkarsh v2.0.0 — Circuit Breaker + Retry-with-Backoff fault tolerance.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

State machine (thread-safe; every transition happens under one lock):
    CLOSED → OPEN       after ``failure_threshold`` CONSECUTIVE failures (a success resets
                        the run — audit A5-08: the old lifetime counter opened the circuit
                        after 5 isolated failures spread over 100 successes)
    OPEN → HALF_OPEN    once ``recovery_timeout`` has elapsed since the last failure
    HALF_OPEN           admits ``half_open_max_calls`` probes (default 2 — the pipeline fires
                        the macro and OHLCV batches concurrently on the same breaker; one
                        probe slot refused the second — audit A5-09)
    HALF_OPEN → CLOSED  on a probe success;  HALF_OPEN → OPEN on a probe failure

``excluded_exceptions`` propagate without counting as a failure: they prove the service
answered (e.g. a private Google Sheet returning its login page is the caller's problem,
not an outage), so they count as a success for availability.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable

log = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when a call is blocked because the circuit is OPEN (or HALF_OPEN and full)."""


class FetchRejected(Exception):
    """The service answered, but the request is unusable (bad URL, private sheet, HTML instead
    of CSV). Never retried and never counted as an outage by the module breakers."""


class CircuitBreaker:
    """Per-service circuit breaker. Thread-safe."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 2,
        name: str = "default",
        excluded_exceptions: tuple[type[BaseException], ...] = (),
    ) -> None:
        if failure_threshold < 1 or half_open_max_calls < 1:
            raise ValueError("failure_threshold and half_open_max_calls must be ≥ 1")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name
        self.excluded_exceptions = tuple(excluded_exceptions)

        self.state = CircuitState.CLOSED
        self.failure_count = 0          # CONSECUTIVE failures (reset by any success)
        self.total_failures = 0         # lifetime, diagnostics only
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.last_success_time: float | None = None
        self.half_open_calls = 0
        self._lock = threading.Lock()

    @staticmethod
    def _now() -> float:
        return time.time()

    def _admit(self) -> None:
        with self._lock:
            if self.state == CircuitState.OPEN:
                elapsed = self._now() - (self.last_failure_time or 0.0)
                if elapsed < self.recovery_timeout:
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' is OPEN — retry in {self.recovery_timeout - elapsed:.1f}s"
                    )
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                log.info("Circuit '%s' HALF_OPEN — admitting up to %d probe(s)", self.name,
                         self.half_open_max_calls)
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit '{self.name}' HALF_OPEN — {self.half_open_max_calls} probe(s) already in flight"
                    )
                self.half_open_calls += 1

    def call(self, func: Callable, *args, **kwargs) -> Any:
        self._admit()
        try:
            result = func(*args, **kwargs)
        except self.excluded_exceptions:
            self._on_success()
            raise
        except Exception:
            self._on_failure()
            raise
        self._on_success()
        return result

    def _on_success(self) -> None:
        with self._lock:
            recovered = self.state == CircuitState.HALF_OPEN
            self.success_count += 1
            self.last_success_time = self._now()
            self.failure_count = 0
            if recovered:
                self.state = CircuitState.CLOSED
                self.half_open_calls = 0
        if recovered:
            log.info("Circuit '%s' CLOSED — service recovered", self.name)

    def _on_failure(self) -> None:
        opened = reopened = False
        with self._lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = self._now()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.half_open_calls = 0
                reopened = True
            elif self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                opened = True
            n = self.failure_count
        if reopened:
            log.warning("Circuit '%s' recovery probe failed — back to OPEN", self.name)
        if opened:
            log.error("Circuit '%s' OPEN — %d consecutive failures (threshold %d)",
                      self.name, n, self.failure_threshold)

    def protect(self, func: Callable) -> Callable:
        """Decorator form: wraps ``func`` with circuit breaker protection."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def get_state(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "total_failures": self.total_failures,
                "success_count": self.success_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "half_open_max_calls": self.half_open_max_calls,
                "last_failure": self.last_failure_time,
                "last_success": self.last_success_time,
            }

    def reset(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.total_failures = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_success_time = None
            self.half_open_calls = 0
        log.info("Circuit '%s' manually reset", self.name)


class RetryWithBackoff:
    """Exponential-backoff retry decorator (initial → ×factor → … capped at ``max_delay``).

    ``max_retries`` is the number of RE-tries (total attempts = max_retries + 1). Exceptions in
    ``no_retry`` (and CircuitBreakerError) propagate immediately.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: tuple = (Exception,),
        no_retry: tuple = (),
    ) -> None:
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.no_retry = tuple(no_retry) + (CircuitBreakerError, FetchRejected)

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = self.initial_delay
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except self.no_retry:
                    raise
                except self.exceptions as e:
                    if attempt >= self.max_retries:
                        log.error("All %d attempts failed. Last error: %s", self.max_retries + 1, e)
                        raise
                    log.warning("Attempt %d/%d failed (%s) — retrying in %.1fs",
                                attempt + 1, self.max_retries + 1, type(e).__name__, delay)
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
            raise AssertionError("unreachable")
        return wrapper


# ── Global circuit breakers (one per external service, shared process-wide) ──
yfinance_circuit = CircuitBreaker(name="yfinance", failure_threshold=5, recovery_timeout=60.0,
                                  excluded_exceptions=(FetchRejected,))
sheets_circuit = CircuitBreaker(name="google_sheets", failure_threshold=3, recovery_timeout=30.0,
                                excluded_exceptions=(FetchRejected,))
# One failure of the WHOLE live chain (niftyindices → Wikipedia) opens it for 5 min, so an
# outage costs one 25 s timeout per 5 min instead of one per run.
constituents_circuit = CircuitBreaker(name="constituents", failure_threshold=1, recovery_timeout=300.0,
                                      half_open_max_calls=1, excluded_exceptions=(FetchRejected,))


def all_circuits() -> list[CircuitBreaker]:
    """Return all module-level circuit breakers for diagnostics."""
    return [yfinance_circuit, sheets_circuit, constituents_circuit]
