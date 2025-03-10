"""
Main module for data extraction and processing for the MIDI-BERT project.

This script parses command line arguments, prepares data in the CP encoding, and saves
the extracted segments to a NumPy file.
"""

import os
import argparse
from pathlib import Path

import numpy as np

from src.constants import DICT_PATH, DEFAULT_OUTPUT_PATH
from src.data_preparation.encodings.CP import CP


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
    parser.add_argument(
        '--dataset',
        type=str,
        choices=[
            "pop909",
            "pop1k7",
            "ASAP",
            "pianist8",
            "emopia"
        ],
        help='Dataset to use.'
    )
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

def main():
    """
    Main function to run data extraction and processing.
    """
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    encoding = CP(dict=args.dict)

    # Use only the input_file as specified in arguments
    files = [args.input_file]
    extract(files, args, encoding)


if __name__ == '__main__':
    main()
