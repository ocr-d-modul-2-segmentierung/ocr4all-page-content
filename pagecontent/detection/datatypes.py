from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ImageData:
    path: str = None
    height: int = None
    image: np.array = None
    average_letter_height: int = None
    binary_image: np.array = None
    pixel_classifier_prediction: np.array = None
