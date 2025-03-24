"""
Main module for data extraction and processing for the MIDI-BERT project.

This script parses command line arguments, prepares data in the CP encoding, and saves
the extracted segments to a NumPy file.
"""

import os
import argparse
from pathlib import Path
from typing import List

import numpy as np

from src.constants import DICT_PATH, DEFAULT_OUTPUT_PATH
from src.data_preparation.encodings.cp import CP


def get_args():
    """
    Parse and return command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Extract MIDI data and process it with the CP model.'
    )

    # Path arguments
    parser.add_argument('--dict', type=str, default=DICT_PATH,
                        help='Path to the dictionary file.')
    parser.add_argument('--input_dir', type=str, default='',
                        help='Input directory containing MIDI files.')
    parser.add_argument('--input_file', type=str, default='',
                        help='Input file path for a MIDI file.')

    # Parameter arguments
    parser.add_argument('--max_len', type=int, default=512,
                        help='Maximum length for data segments.')

    # Output arguments
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_PATH,
                        help='Directory to save the output file.')
    parser.add_argument('--name', default="",
                        help='Name for the output file (without extension).')

    args = parser.parse_args()
    return args

def extract(files, args, model, mode=''):
    """
    Extract data segments from MIDI files using the provided model.

    Args:
       files (list): List of file paths.
       args (argparse.Namespace): Command-line arguments.
       model: Model instance used for data preparation.
       mode (str, optional): Mode indicator for file types. Defaults to ''.
    """
    assert len(files) > 0, "No files provided for extraction."
    print(f'Number of {mode} files: {len(files)}')

    segments, _ = model.prepare_data(files, int(args.max_len))

    if args.input_dir == '' and args.input_file == '':
        print('No input specified')
        return

    name = args.input_dir or args.input_file
    if args.name == '':
        args.name = Path(name).stem
    output_file = os.path.join(args.output_dir, f'{args.name}.npy')

    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')


def find_midi_files(directory, midi_list):
    """
    Recursively iterate through all files in a directory and its subdirectories,
    appending paths of all MIDI files to the provided list.

    Args:
        directory (str): Path to the directory to search
        midi_list (list): List to which MIDI file paths will be appended

    Returns:
        None: The function modifies the provided midi_list in-place
    """
    try:
        # Ensure the directory exists
        if not os.path.isdir(directory):
            print(f"Warning: {directory} is not a valid directory.")
            return

        # Walk through directory tree
        for root, _, files in os.walk(directory):
            for file in files:
                # Check if file has a MIDI extension
                if file.lower().endswith(('.mid', '.midi')):
                    # Get the full path and append to the list
                    full_path = os.path.join(root, file)
                    midi_list.append(full_path)
    except OSError as e:
        print(f"Error searching directory {directory}: {e}")

def main():
    """
    Main function to run data extraction and processing.
    """
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    encoding = CP(dict_path=args.dict)

    # Use only the input_file as specified in arguments
    files: List[str] = []

    if args.input_file != "":
        files.append(args.input_file)
    elif args.input_dir != "":
        find_midi_files(args.input_dir, files)

    extract(files, args, encoding)

if __name__ == '__main__':
    main()
