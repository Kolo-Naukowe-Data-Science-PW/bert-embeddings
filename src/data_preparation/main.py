import os
import argparse
import glob
import numpy as np
from pathlib import Path

from src.constants import DICT_PATH, DEFAULT_OUTPUT_PATH
from src.data_preparation.encodings.CP import CP

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path ###
    parser.add_argument('--dict', type=str, default=DICT_PATH)
    parser.add_argument('--dataset', type=str, choices=["pop909", "pop1k7", "ASAP", "pianist8", "emopia"])
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--input_file', type=str, default='')

    ### parameter ###
    parser.add_argument('--max_len', type=int, default=512)

    ### output ###
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--name', default="")  # will be saved as "{output_dir}/{name}.npy"

    args = parser.parse_args()

    return args

def extract(files, args, model, mode=''):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files)

    print(f'Number of {mode} files: {len(files)}')

    segments, ans = model.prepare_data(files, args.task, int(args.max_len))

    dataset = args.dataset if args.dataset != 'pianist8' else 'composer'

    if args.input_dir != '' or args.input_file != '':
        name = args.input_dir or args.input_file
        if args.name == '':
            args.name = Path(name).stem
        output_file = os.path.join(args.output_dir, f'{args.name}.npy')

    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = CP(dict=args.dict)

    dataset = 'joann8512-Pianist8-ab9f541'

    train_files = glob.glob(f'Data/Dataset/{dataset}/train/*/*.mid')
    valid_files = glob.glob(f'Data/Dataset/{dataset}/valid/*/*.mid')
    test_files = glob.glob(f'Data/Dataset/{dataset}/test/*/*.mid')

    files = [args.input_file]

    extract(files, args, model)

if __name__ == '__main__':
    main()