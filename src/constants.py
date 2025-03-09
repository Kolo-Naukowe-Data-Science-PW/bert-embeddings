"""
This module defines constants used throughout the project.
"""

from typing import Final
import numpy as np

DICT_PATH: Final[str] = 'src/data_preparation/dict/cp_dict.pkl'
DEFAULT_OUTPUT_PATH: Final[str] = 'data/cp_data/tmp'

# utils constants

# parameters for input
DEFAULT_VELOCITY_BINS = np.array([0, 32, 48, 64, 80, 96, 128])
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480
