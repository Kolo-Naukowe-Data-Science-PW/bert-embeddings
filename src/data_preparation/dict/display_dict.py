import pickle
from pprint import pprint
from src.constants import DICT_PATH

def display_dict(file_path: str):
    with open(file_path, 'rb') as f:
        cp_data = pickle.load(f)
    pprint(cp_data)

if __name__ == "__main__":
    display_dict(DICT_PATH)