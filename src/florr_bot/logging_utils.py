from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def configure_logging(debug_dir: Path, verbose: bool = False) -> logging.Logger:
    debug_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger("florr_bot")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = RichHandler(rich_tracebacks=True, markup=False)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(debug_dir / "bot.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
