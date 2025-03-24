"""
This module provides functionality to display the contents of a dictionary
stored in a pickle file.
"""

import pickle
from pprint import pprint
from src.constants import DICT_PATH

def display_dict(file_path: str):
    """
    Display the contents of a dictionary stored in a pickle file.

    Args:
        file_path (str): The path to the pickle file containing the dictionary.
    """
    with open(file_path, 'rb') as f:
        cp_data = pickle.load(f)
    pprint(cp_data)

if __name__ == "__main__":
    display_dict(DICT_PATH)
