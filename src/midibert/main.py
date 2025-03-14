"""
This module is designed to train a MidiBERT model. It provides functionality to:

1. Parse command-line arguments using argparse for configuration.
2. Load and preprocess MIDI data from specified datasets for training and validation.
3. Build and configure the MidiBERT model using the Hugging Face BERT architecture,
   with custom settings tailored to MIDI data.
4. Train the model using the BERTTrainer class.
5. Save the trained model and logs for future use.

The module allows flexibility in setting training parameters, including learning rate, batch size,
maximum sequence length, and masking percentage, while also supporting multi-device (CUDA)
training or CPU-only execution.
"""
import argparse
import os
import pickle

import numpy as np
from torch.utils.data import DataLoader
from transformers import BertConfig

import src.constants as const
from src.constants import SPLIT_FACTOR
from src.midibert.midi_bert import MidiBert
from src.midibert.midi_dataset import MidiDataset
from src.midibert.bert_trainer import BERTTrainer

def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the MidiBERT model training.
    """
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default=const.DICT_PATH)
    parser.add_argument('--name', type=str, default=const.DEFAULT_MODEL_NAME)
    parser.add_argument('--input_data', type=str, default=const.DEFAULT_DATA_PATH)

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=const.DEFAULT_NUM_WORKERS)
    parser.add_argument('--batch_size', type=int, default=const.DEFAULT_BATCH_SIZE)
    parser.add_argument('--mask_percent', type=float, default=const.DEFAULT_MASK_PERCENT,
        help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=const.DEFAULT_MAX_SEQ_LEN,
                        help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=const.DEFAULT_HIDDEN_SIZE)
    parser.add_argument('--epochs', type=int, default=const.DEFAULT_EPOCHS,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=const.DEFAULT_LR,
                        help='initial learning rate')

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default: False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1, 2, 3],
                        help="CUDA device ids")

    args = parser.parse_args()

    return args

def load_data(input_data: str):
    """
    Load and concatenate MIDI data from specified datasets for training and validation.
    TODO: Loading data from multiple datasets is currently disabled.
    :param input_data: path to the input data file
    :return:
    """
    to_concat = []

    # Load data
    data = np.load(input_data, allow_pickle=True)
    print(f'   {input_data}: {data.shape}')
    to_concat.append(data)

    training_data = np.vstack(to_concat)
    print('   > all training data:', training_data.shape)

    # shuffle during training phase
    index = np.arange(len(training_data))
    np.random.shuffle(index)
    training_data = training_data[index]
    split = int(len(training_data) * SPLIT_FACTOR)
    x_train, x_val = training_data[:split], training_data[split:]

    return x_train, x_val

# pylint: disable=too-many-locals
def main():
    """
    Trains the MidiBERT model by parsing command-line arguments, loading datasets,
    setting up the model and DataLoader, and running the training loop.
    It includes early stopping based on validation accuracy and saves the model
    and logs periodically.
    """
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset")
    x_train, x_val = load_data(args.input_data)

    trainset = MidiDataset(x=x_train)
    validset = MidiDataset(x=x_val)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))

    print("\nBuilding BERT model")
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                               position_embedding_type='relative_key_query',
                               hidden_size=args.hs)
    midibert = MidiBert(bert_config=configuration, e2w=e2w, w2e=w2e)

    print("\nCreating BERT Trainer")
    trainer = BERTTrainer(midibert, train_loader, valid_loader, args.lr, args.batch_size,
                          args.max_seq_len, args.mask_percent, args.cpu, args.cuda_devices)

    print("\nTraining Start")
    save_dir = 'MidiBERT/result/pretrain/' + args.name
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {filename}")

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 30:
            print('valid acc not improving for 30 epochs')
            break

        train_loss, train_acc = trainer.train()
        valid_loss, valid_acc = trainer.valid()

        weighted_score = [x * y for (x, y) in zip(valid_acc, midibert.n_tokens)]
        avg_acc = sum(weighted_score) / sum(midibert.n_tokens)

        is_best = avg_acc > best_acc
        best_acc = max(avg_acc, best_acc)

        if is_best:
            bad_cnt, best_epoch = 0, epoch
        else:
            bad_cnt += 1

        print(f'epoch: {epoch + 1}/{args.epochs} | Train Loss: {train_loss} | '
              f'Train acc: {train_acc} | Valid Loss: {valid_loss} | Valid acc: {valid_acc}')

        trainer.save_checkpoint(epoch, best_acc, valid_acc,
                                valid_loss, train_loss, is_best, filename)

        with open(os.path.join(save_dir, 'log'), 'a', encoding='utf-8') as outfile:
            outfile.write(f'Epoch {epoch + 1}: train_loss={train_loss}, train_acc={train_acc}, '
                          f'valid_loss={valid_loss}, valid_acc={valid_acc}\n')

if __name__ == '__main__':
    main()
