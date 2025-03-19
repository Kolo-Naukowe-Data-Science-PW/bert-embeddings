"""
This module defines functions for creating and saving dictionaries that map musical events to
integer indices and vice versa. The dictionaries support four main event classes: 'Bar', 'Position',
'Pitch', and 'Duration'. Additionally, special tokens like '<PAD>' and '<MASK>' are added for each
event class.

The module includes:
- special_token: A helper function for adding special tokens ('<PAD>' and '<MASK>') to the
  dictionaries.
- create_cp_file: A function for generating the event-to-word and word-to-event dictionaries and
  saving them as a tuple to a specified file using pickle.

Usage:
1. Call `create_cp_file(file_path)` with a valid file path to generate and save the dictionaries.
2. The dictionaries include mappings for each event class and special tokens.
"""

import pickle
from src.constants import DICT_PATH

PAD_TOKEN: str = ' <PAD>'
MASK_TOKEN: str = ' <MASK>'

def special_token(cnt: int, cls: str, event2word: dict, word2event: dict) -> int:
    """
    Add special tokens for a given class (e.g., <PAD> and <MASK>) to the dictionaries.
    """
    event2word[cls][cls + PAD_TOKEN] = cnt
    word2event[cls][cnt] = cls + PAD_TOKEN
    cnt += 1

    event2word[cls][cls + MASK_TOKEN] = cnt
    word2event[cls][cnt] = cls + MASK_TOKEN
    cnt += 1

    return cnt

def create_cp_file(file_path):
    """
    Creates two dictionaries mapping musical events to integer indices and vice versa,
    and saves them as a tuple to the specified file using pickle.
    """

    # Initialize dictionaries for each event type
    event2word = {'Bar': {}, 'Position': {}, 'Pitch': {}, 'Duration': {}}
    word2event = {'Bar': {}, 'Position': {}, 'Pitch': {}, 'Duration': {}}

    # Process Bar tokens
    cnt, cls = 0, 'Bar'
    event2word[cls]['Bar New'] = cnt
    word2event[cls][cnt] = 'Bar New'
    cnt += 1

    event2word[cls]['Bar Continue'] = cnt
    word2event[cls][cnt] = 'Bar Continue'
    cnt += 1

    cnt = special_token(cnt, cls, event2word, word2event)

    # Process Position tokens
    cnt, cls = 0, 'Position'
    for i in range(1, 17):
        token = f'Position {i}/16'
        event2word[cls][token] = cnt
        word2event[cls][cnt] = token
        cnt += 1
    cnt = special_token(cnt, cls, event2word, word2event)

    # Process Pitch tokens
    cnt, cls = 0, 'Pitch'
    for i in range(22, 108):
        token = f'Pitch {i}'
        event2word[cls][token] = cnt
        word2event[cls][cnt] = token
        cnt += 1
    cnt = special_token(cnt, cls, event2word, word2event)

    # Process Duration tokens
    cnt, cls = 0, 'Duration'
    for i in range(64):
        token = f'Duration {i}'
        event2word[cls][token] = cnt
        word2event[cls][cnt] = token
        cnt += 1
    cnt = special_token(cnt, cls, event2word, word2event)

    # Save the tuple (event2word, word2event) to the specified file path using pickle
    with open(file_path, 'wb') as f:
        pickle.dump((event2word, word2event), f)

if __name__ == '__main__':
    create_cp_file(DICT_PATH)
