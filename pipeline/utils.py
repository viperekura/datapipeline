import functools
import logging
import sys
import os
from typing import Optional, Callable, Any


def error_handler(
    logger: Optional[logging.Logger] = None,
    reraise: bool = True,
    log_level: int = logging.ERROR,
    capture_keyboard_interrupt: bool = False,
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not capture_keyboard_interrupt and isinstance(e, KeyboardInterrupt):
                    raise

                nonlocal logger
                log = logger or logging.getLogger(func.__module__)
                log.log(
                    log_level,
                    f"Error in {func.__name__}: {e}",
                    exc_info=True,
                )
                if reraise:
                    raise
                return None

        return wrapper

    return decorator


def setup_logging(level: Optional[int] = None) -> None:

    if level is None:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
