"""
This module defines constants used throughout the project.
"""
from pathlib import Path
from typing import Final
import numpy as np

# paths
DICT_PATH: Final[str] = str(Path(__file__).resolve().parent / 'data_preparation/dict/cp_dict.pkl')
DEFAULT_OUTPUT_PATH: Final[str] = 'data/cp_data/tmp'
DEFAULT_DATA_PATH: Final[str] = 'data/CP_data/tmp/Ryuchi_Opus.npy'

# data creation parameters
DEFAULT_VELOCITY_BINS = np.array([0, 32, 48, 64, 80, 96, 128])
DEFAULT_FRACTION: Final[int] = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]
DEFAULT_RESOLUTION: Final[int] = 480

# model parameters
DEFAULT_MODEL_NAME: Final[str] = 'MidiBERT'
DEFAULT_NUM_WORKERS: Final[int] = 5
DEFAULT_BATCH_SIZE: Final[int] = 12
DEFAULT_MASK_PERCENT: Final[float] = 0.15
DEFAULT_MAX_SEQ_LEN: Final[int] = 512
DEFAULT_HIDDEN_SIZE: Final[int] = 768
DEFAULT_EPOCHS: Final[int] = 500
DEFAULT_LR = 2e-5
SPLIT_FACTOR: Final[float] = 0.85

