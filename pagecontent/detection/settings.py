from enum import IntEnum
from typing import NamedTuple, Optional


class PageContentSettings(NamedTuple):

    debug: bool = False
    line_space_height: int = 0
    target_line_space_height: int = 6

    model: Optional[str] = None
    model_foreground_threshold: float = 0.5
    model_foreground_normalize: bool = True

    debug_model: bool = False
    processes: int = 12
