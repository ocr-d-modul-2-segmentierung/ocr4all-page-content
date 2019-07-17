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

    min_contour_area: float = 0.03  # Todo: Increase min contour area needed to include in the generation of the page
                                    # Todo: content the further the contour is located at the page border
    min_area_ratio: float = 0.7



