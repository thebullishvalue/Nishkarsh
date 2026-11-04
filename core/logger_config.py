"""
Nishkarsh v2.0.0 — console output for the analysis pipeline.
निष्कर्ष (Nishkarsha) — "Conclusion / Inference"

CORE — styled, sectioned terminal output (adapted from the Pragyam 7.0.5 design), fixed per
audit A5-20:
  * ONE run id per run: ``console.begin_run()`` sets it; ``header()`` prints it only when a
    run is active (v1 printed an import-time id AND a second freshly generated one).
  * No ANSI escapes unless the sink is a TTY (``NO_COLOR`` / ``FORCE_COLOR`` honoured).
    Captured text (``tee``) is always plain.
  * Injectable sink: ``ConsoleOutput(sink=...)`` takes a text stream or a ``str -> None``
    callable; ``console.tee(sink)`` temporarily copies output to a second sink, so the
    pipeline can capture the run log for its JSON report.
  * Data-layer ``logging`` events (``data.*``) can be routed into a console with
    ``route_logging(console)``; they otherwise go to stderr as before.
  * No global side effects at import (v1 wrapped/reconfigured ``sys.stdout``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Iterator, TextIO

__all__ = ["Colors", "ConsoleOutput", "console", "generate_run_id", "route_logging",
           "unroute_logging", "ConsoleLogHandler", "strip_ansi"]

if os.name == "nt":  # enable ANSI on Windows 10+ consoles; no-op elsewhere
    try:
        import colorama
        colorama.just_fix_windows_console()
    except Exception:  # noqa: BLE001
        pass

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def generate_run_id() -> str:
    """A new unique run id ``YYYYmmdd_HHMMSS_xxxxxxxx`` (pure; does not start a run)."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class Colors:
    """ANSI color codes (stripped automatically when the sink is not a TTY)."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    SUCCESS = "✓"
    WARNING = "⚠"
    ERROR = "✗"
    INFO = "ℹ"


Sink = TextIO | Callable[[str], Any]


def _emit(sink: Sink, text: str) -> None:
    if callable(getattr(sink, "write", None)):
        try:
            sink.write(text)
        except (UnicodeEncodeError, UnicodeDecodeError):
            enc = getattr(sink, "encoding", None) or "utf-8"
            sink.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
        flush = getattr(sink, "flush", None)
        if callable(flush):
            flush()
    else:
        sink(text)


class ConsoleOutput:
    """Direct console output with styled sections and phase timing.

    Args:
        sink: text stream or ``str -> None`` callable; ``None`` → ``sys.stdout`` resolved at
              write time (so test capture works).
        color: force ANSI on/off; ``None`` → auto (TTY, unless ``NO_COLOR``; ``FORCE_COLOR`` wins).
    """

    def __init__(self, sink: Sink | None = None, color: bool | None = None) -> None:
        self._sink = sink
        self._color = color
        self._extra: list[Sink] = []
        self._lock = threading.RLock()
        self._section_depth = 0
        self._phase_timers: dict[str, datetime] = {}
        self.run_id: str | None = None

    # -- run identity -----------------------------------------------------
    def begin_run(self, run_id: str | None = None) -> str:
        """Start a run: sets and returns the single run id printed by ``header``."""
        self.run_id = run_id or generate_run_id()
        self._phase_timers.clear()
        self._section_depth = 0
        return self.run_id

    def end_run(self) -> None:
        self.run_id = None

    # -- sinks ------------------------------------------------------------
    @property
    def sink(self) -> Sink:
        return self._sink if self._sink is not None else sys.stdout

    def use_color(self) -> bool:
        if self._color is not None:
            return self._color
        if os.environ.get("FORCE_COLOR"):
            return True
        if os.environ.get("NO_COLOR"):
            return False
        isatty = getattr(self.sink, "isatty", None)
        try:
            return bool(isatty()) if callable(isatty) else False
        except Exception:  # noqa: BLE001
            return False

    @contextlib.contextmanager
    def tee(self, sink: Sink) -> Iterator[Sink]:
        """Copy everything written inside the block to ``sink`` (ANSI-stripped)."""
        with self._lock:
            self._extra.append(sink)
        try:
            yield sink
        finally:
            with self._lock:
                if sink in self._extra:
                    self._extra.remove(sink)

    def _write(self, message: str = "", end: str = "\n") -> None:
        text = f"{message}{end}"
        plain = strip_ansi(text)
        with self._lock:
            _emit(self.sink, text if self.use_color() else plain)
            for extra in list(self._extra):
                _emit(extra, plain)

    # -- formatting helpers ----------------------------------------------
    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _elapsed(self, start: datetime) -> str:
        total_secs = int((datetime.now() - start).total_seconds())
        return f"{total_secs}s" if total_secs < 60 else f"{total_secs // 60}m {total_secs % 60}s"

    def line(self, char: str = "─", length: int = 70) -> None:
        self._write(f"{Colors.GRAY}{char * length}{Colors.RESET}")

    def header(self, title: str, version: str = "", run_id: str | None = None) -> None:
        """Run header. Prints the run id only if one is active (``begin_run``) or given."""
        rid = run_id or self.run_id
        self._write()
        self.line("═", 70)
        self._write(f"  {Colors.BOLD}{Colors.CYAN}{title} {version}{Colors.RESET}")
        if rid:
            self._write(f"  {Colors.GRAY}Run ID: {rid}{Colors.RESET}")
        self._write(f"  {Colors.GRAY}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        self.line("═", 70)
        self._write()

    def main_header(self, title: str, details: dict[str, Any]) -> None:
        self._write()
        self.line("═", 70)
        self._write(f"  {Colors.BOLD}{Colors.CYAN}{title}{Colors.RESET}")
        self.line("─", 70)
        for key, value in details.items():
            self._write(f"  {Colors.GRAY}{key}:{Colors.RESET} {value}")
        self.line("═", 70)
        self._write()

    def section(self, title: str, phase: str = "") -> None:
        self._write()
        if phase:
            self.line("═", 70)
            self._write(f"  {Colors.BOLD}{Colors.BLUE}{phase}: {title}{Colors.RESET}")
            self.line("═", 70)
        else:
            self._write(f"{Colors.BOLD}{title}{Colors.RESET}")
            self._write(Colors.GRAY + "─" * len(title) + Colors.RESET)
        self._section_depth += 1

    def item(self, label: str, value: Any, indent: int = 4) -> None:
        self._write(f"{' ' * indent}{Colors.GRAY}{label}:{Colors.RESET} {value}")

    def detail(self, message: str) -> None:
        self._write(f"    {Colors.CYAN}→{Colors.RESET} {message}")

    def success(self, message: str) -> None:
        self._write(f"  {Colors.GREEN}{Colors.SUCCESS} SUCCESS:{Colors.RESET} {message}")

    def warning(self, message: str) -> None:
        self._write(f"  {Colors.YELLOW}{Colors.WARNING} WARNING:{Colors.RESET} {message}")

    def error(self, message: str) -> None:
        self._write(f"  {Colors.RED}{Colors.ERROR} ERROR:{Colors.RESET} {message}")

    def failure(self, step: str, error: str) -> None:
        self._write(f"  {Colors.RED}{Colors.ERROR} FAILURE:{Colors.RESET} {step}")
        self._write(f"      {Colors.GRAY}Reason:{Colors.RESET} {error}")

    def summary(self, title: str, data: dict[str, Any]) -> None:
        self._write()
        self._write(f"  {Colors.GRAY}┌─ {title}{Colors.RESET}")
        for key, value in data.items():
            self._write(f"  {Colors.GRAY}│   {key}:{Colors.RESET} {value}")
        self._write(f"  {Colors.GRAY}└─{Colors.RESET}")

    def start_phase(self, phase: str, num: int = 0, total: int = 0) -> None:
        self._phase_timers[phase] = datetime.now()
        self._write()
        if num and total:
            self.line("═", 70)
            self._write(f"  {Colors.BOLD}{Colors.BLUE}Phase {num}/{total}: {phase}{Colors.RESET}")
            self.line("═", 70)
        else:
            self.line("─", 60)
            self._write(f"  {Colors.BOLD}{Colors.BLUE}Phase: {phase}{Colors.RESET}")
            self.line("─", 60)

    def end_phase(self, phase: str) -> None:
        if phase in self._phase_timers:
            elapsed = self._elapsed(self._phase_timers.pop(phase))
            self._write(f"\n  {Colors.GREEN}{Colors.SUCCESS} Phase Complete: {phase} "
                        f"{Colors.GRAY}[{elapsed}]{Colors.RESET}")


# ── logging → console routing (data-layer events into the run log) ──────────

class ConsoleLogHandler(logging.Handler):
    """Forward ``logging`` records to a ConsoleOutput (warning/error/detail by level)."""

    def __init__(self, out: ConsoleOutput, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.out = out

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = f"[{record.name}] {record.getMessage()}"
            if record.levelno >= logging.ERROR:
                self.out.error(msg)
            elif record.levelno >= logging.WARNING:
                self.out.warning(msg)
            else:
                self.out.detail(msg)
        except Exception:  # noqa: BLE001
            self.handleError(record)


def route_logging(out: ConsoleOutput | None = None, names: tuple[str, ...] = ("data",),
                  level: int = logging.INFO) -> ConsoleLogHandler:
    """Attach a handler routing the named loggers (default: the data layer) into ``out``.
    Returns the handler; pass it to ``unroute_logging`` when the run ends."""
    h = ConsoleLogHandler(out or console, level)
    h._nishkarsh_names = tuple(names)  # type: ignore[attr-defined]
    for n in names:
        lg = logging.getLogger(n)
        lg.addHandler(h)
        if lg.level == logging.NOTSET or lg.level > level:
            lg.setLevel(level)
    return h


def unroute_logging(handler: ConsoleLogHandler) -> None:
    for n in getattr(handler, "_nishkarsh_names", ()):
        logging.getLogger(n).removeHandler(handler)


# ── Global instance (stdout) ────────────────────────────────────────────────
console = ConsoleOutput()
