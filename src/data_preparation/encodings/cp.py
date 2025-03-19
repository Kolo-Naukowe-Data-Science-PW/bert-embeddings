"""
Module for the CP encoding and data preparation.

This module contains the CP class used to extract events from MIDI files,
pad data, and prepare data segments using the provided dictionary.
"""

import pickle
from tqdm import tqdm
import numpy as np

from src.data_preparation import utils

class CP:
    """
    CP encoding class for processing MIDI files.

    Attributes:
        event2word (dict): Mapping from event names to words.
        word2event (dict): Mapping from words back to event names.
        pad_word (list): List of pad words for each event type.
    """

    def __init__(self, dict_path: str):
        """
        Initialize the CP object with a dictionary file.

        Args:
            dict_path (str): Path to the pickle file containing event2word and word2event mappings.
        """
        with open(dict_path, 'rb') as f:
            self.event2word, self.word2event = pickle.load(f)
        self.pad_word = [self.event2word[etype][f'{etype} <PAD>'] for etype in self.event2word]

    def extract_events(self, input_path: str):
        """
       Extract events from a MIDI file at the given path.

       Args:
           input_path (str): Path to the MIDI file.

       Returns:
           list or None: List of events if extraction is successful, or None
           if the MIDI file is empty.
       """
        note_items, tempo_items = utils.read_items(input_path)
        if len(note_items) == 0:  # if the midi contains nothing
            return None
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events

    def padding(self, data, max_len, ans):
        """
        Pad the given data to reach the maximum length.

        Args:
            data (list): List of data elements to be padded.
            max_len (int): The maximum length desired.
            ans (bool): Flag to determine type of padding.

        Returns:
            list: The padded data list.
        """
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            if not ans:
                data.append(self.pad_word)
            else:
                data.append(0)
        return data

    # pylint: disable=too-many-locals
    def prepare_data(self, midi_paths, max_len):
        """
        Prepare and segment data from a list of MIDI file paths.

        Args:
            midi_paths (list): List of MIDI file paths.
            max_len (int): Maximum length for each data segment.

        Returns:
            tuple: A tuple containing NumPy arrays of words and associated labels.
        """
        all_words, all_ys = [], []

        for path in tqdm(midi_paths):
            # Extract events from the MIDI file
            events = self.extract_events(path)
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue

            # Convert events to words
            words, ys = [], []
            for note_tuple in events:
                nts = []
                for e in note_tuple:
                    e_text = f'{e.name} {e.value}'
                    nts.append(self.event2word[e.name][e_text])
                words.append(nts)

            # Slice into chunks so that each chunk has length max_len
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i + max_len])
                slice_ys.append(ys[i:i + max_len])

            # If the last chunk is shorter than max_len, pad it
            if slice_words and len(slice_words[-1]) < max_len:
                slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)

            all_words += slice_words
            all_ys += slice_ys

        all_words = np.array(all_words)
        all_ys = np.array(all_ys)

        return all_words, all_ys
