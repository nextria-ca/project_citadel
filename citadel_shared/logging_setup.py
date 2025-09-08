# citadel_shared/logging_setup.py
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Final, Tuple


if os.name == "nt":
    try:
        import colorama
        colorama.just_fix_windows_console()
    except ImportError:
        pass

_COLOURS: Final = {
    logging.DEBUG:    "\033[36m",
    logging.INFO:     "\033[32m",
    logging.WARNING:  "\033[33m",
    logging.ERROR:    "\033[31m",
    logging.CRITICAL: "\033[41m",
}
_RESET: Final = "\033[0m"


class _ColourFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelno, "")
        if colour:
            original = record.levelname
            record.levelname = f"{colour}{original}{_RESET}"
            try:
                return super().format(record)
            finally:
                record.levelname = original
        return super().format(record)


class RecentDedupFilter(logging.Filter):
    """
    Drops duplicate (name, level, message) seen within a sliding window.
    Configure window via CITADEL_LOG_DEDUP_MS (default: 250).
    """
    def __init__(self, window_ms: int = 250):
        super().__init__()
        self.window_ms = window_ms
        self._last: dict[Tuple[str, int, str], float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            key = (record.name, record.levelno, record.getMessage())
        except Exception:
            return True
        now = time.monotonic() * 1000.0
        last = self._last.get(key)
        self._last[key] = now
        if last is None:
            return True
        return (now - last) > self.window_ms


class SafeStreamHandler(logging.StreamHandler):
    """Never crash on UnicodeEncodeError; fall back to backslashreplace."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                record.msg = record.getMessage().encode(
                    "ascii", "backslashreplace"
                ).decode("ascii")
                record.args = ()
                super().emit(record)
            except Exception:
                self.handleError(record)


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    def doRollover(self) -> None:
        try:
            super().doRollover()
        except PermissionError as e:
            try:
                sys.stderr.write(f"WARNING | logging_setup | Log rollover skipped (file in use): {e}\n")
            except Exception:
                pass


def _console_handler(level: int, dedup_ms: int) -> logging.Handler:
    handler = SafeStreamHandler(sys.__stdout__)
    handler.setLevel(level)
    handler.setFormatter(
        _ColourFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
    if dedup_ms > 0:
        handler.addFilter(RecentDedupFilter(dedup_ms))
    return handler


def _file_handler(level: int, directory: Path, dedup_ms: int) -> logging.Handler:
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{logging.getLevelName(level).lower()}.log"
    handler = SafeRotatingFileHandler(
        file_path,
        backupCount=3,
        maxBytes=5_000_000,
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    if dedup_ms > 0:
        handler.addFilter(RecentDedupFilter(dedup_ms))
    return handler


def _env_flag(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).lower() not in ("0", "false", "no", "off")


def setup_logging(
    env: str | None = None,
    log_dir: str | Path = "logs",
    *,
    enable_console: bool | None = None,
    enable_files: bool | None = None,
) -> None:
    """
    Env overrides:
      CITADEL_LOG_CONSOLE=0/1
      CITADEL_LOG_FILES=0/1
      CITADEL_LOG_DEDUP_MS=<int>   (default 250; 0 disables)
    """
    env = (env or os.getenv("LOGGING_LEVEL") or "beta").lower()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove existing handlers to prevent duplication on reinit
    for h in list(root.handlers):
        root.removeHandler(h)

    use_console = _env_flag("CITADEL_LOG_CONSOLE") if enable_console is None else bool(enable_console)
    use_files   = _env_flag("CITADEL_LOG_FILES")   if enable_files   is None else bool(enable_files)
    dedup_ms    = int(os.getenv("CITADEL_LOG_DEDUP_MS", "250"))

    console_level = logging.DEBUG if env == "beta" else logging.WARNING
    if use_console:
        root.addHandler(_console_handler(console_level, dedup_ms))

    if use_files:
        log_dir_path = Path(log_dir)
        for lvl in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
            root.addHandler(_file_handler(lvl, log_dir_path, dedup_ms))

    root.info(
        "Logging initialised in %s mode - logs dir: %s (console=%s, files=%s, dedup=%sms)",
        env.upper(),
        Path(log_dir).resolve(),
        use_console,
        use_files,
        dedup_ms,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    root = logging.getLogger()
    if not root.handlers:
        setup_logging()
    return logging.getLogger(name)
