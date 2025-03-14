"""
This module implements the BERTTrainer class, which provides training and validation routines
for the MidiBERT model— a BERT-based transformer adapted for symbolic music representation learning.
The trainer handles key tasks such as masked language modeling (MLM) with custom masking strategies,
accuracy computation across multiple MIDI token types (e.g., Bar, Position, Pitch, Duration),
gradient clipping, and checkpoint saving. It supports multi-GPU training using DataParallel
and employs the AdamW optimizer with weight decay for regularization.
"""

import random
import sys
import shutil
import copy

import tqdm
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW
import numpy as np

from src.midibert.midi_bert import MidiBert
from src.midibert.midi_bert_lm import MidiBertLM
from src.constants import WEIGHT_DECAY

#pylint: disable=too-many-instance-attributes
class BERTTrainer:
    """
    Trainer for the MidiBERT model.
    """
    #pylint: disable=too-many-positional-arguments
    #pylint: disable=too-many-arguments
    def __init__(self, midibert: MidiBert, train_dataloader, valid_dataloader,
                 lr, batch_size, max_seq_len, mask_percent, cpu, cuda_devices=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else 'cpu')
        self.midibert = midibert
        self.model = MidiBertLM(midibert).to(self.device)
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('# total parameters:', self.total_params)

        if torch.cuda.device_count() > 1 and not cpu:
            print(f"Use {torch.cuda.device_count()} GPUS")
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.mask_percent = mask_percent
        self.lseq = list(range(self.max_seq_len))
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, predict, target, loss_mask):
        """
        Compute the loss for the given prediction, target, and loss mask.
        The loss mask specifies which tokens should be considered for loss calculation.
        """
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def get_mask_ind(self):
        """
        Get the indices for masking tokens in the input sequence.
        masked - 80% of the tokens will be masked out
        rand - 10% of the tokens will be replaced with random tokens
        cur - 10% of the tokens will remain unchanged
        """
        mask_ind = random.sample(self.lseq, round(self.max_seq_len * self.mask_percent))
        masked = random.sample(mask_ind, round(len(mask_ind) * 0.8))
        left = list(set(mask_ind) - set(masked))
        rand = random.sample(left, round(len(mask_ind) * 0.1))
        cur = list(set(left) - set(rand))
        return masked, rand, cur

    def train(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data, self.max_seq_len)
        return train_loss, train_acc

    def valid(self):
        """
        Validate the model on the validation dataset.
        """
        self.model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = self.iteration(self.valid_data, self.max_seq_len, train=False)
        return valid_loss, valid_acc

    # pylint: disable=too-many-locals
    def iteration(self, training_data, max_seq_len, train=True):
        """
        Iterate over the training or validation data and perform one epoch of training.
        """
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc = [0] * len(self.midibert.e2w)
        total_losses = 0

        for batch in pbar:
            batch_size = batch.shape[0]
            batch = batch.to(self.device)
            input_ids = copy.deepcopy(batch)
            loss_mask = torch.zeros(batch_size, max_seq_len)

            for b in range(batch_size):
                # get index for masking
                masked, rand, cur = self.get_mask_ind()
                # apply mask, random, remain current token
                for i in masked:
                    mask_word = torch.tensor(self.midibert.mask_word_np).to(self.device)
                    input_ids[b][i] = mask_word
                    loss_mask[b][i] = 1
                for i in rand:
                    rand_word = torch.tensor(self.midibert.get_rand_tok()).to(self.device)
                    input_ids[b][i] = rand_word
                    loss_mask[b][i] = 1
                for i in cur:
                    loss_mask[b][i] = 1

            loss_mask = loss_mask.to(self.device)

            # avoid attend to pad word
            attn_mask = (input_ids[:, :, 0] != self.midibert.bar_pad_word).float().to(self.device)

            y = self.model.forward(input_ids, attn_mask)

            # get the most likely choice with max
            outputs = []
            for i, _ in enumerate(self.midibert.e2w):
                output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
                outputs.append(output)
            outputs = np.stack(outputs, axis=-1)
            outputs = torch.from_numpy(outputs).to(self.device)  # (batch, seq_len)

            # accuracy
            all_acc = []
            for i in range(4):
                acc = torch.sum((batch[:, :, i] == outputs[:, :, i]).float() * loss_mask)
                acc /= torch.sum(loss_mask)
                all_acc.append(acc)
            total_acc = [sum(x) for x in zip(total_acc, all_acc)]

            # reshape (b, s, f) -> (b, f, s)
            for i, etype in enumerate(self.midibert.e2w):
                # print('before',y[i][:,...].shape)
                # each: (4,512,5), (4,512,20), (4,512,90), (4,512,68)
                y[i] = y[i][:, ...].permute(0, 2, 1)

            # calculate losses
            losses, n_tok = [], []
            for i, etype in enumerate(self.midibert.e2w):
                n_tok.append(len(self.midibert.e2w[etype]))
                losses.append(self.compute_loss(y[i], batch[..., i], loss_mask))
            total_loss_all = [x * y for x, y in zip(losses, n_tok)]
            total_loss = sum(total_loss_all) / sum(n_tok)  # weighted

            # udpate only in train
            if train:
                self.model.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 3.0)
                self.optim.step()

            # acc
            accs = list(map(float, all_acc))
            sys.stdout.write(
                f'Loss: {total_loss:06f} | loss: {losses[0]:03f}, {losses[1]:03f}, '
                f'{losses[2]:03f}, {losses[3]:03f} acc: {accs[0]:03f}, {accs[1]:03f},'
                f' {accs[2]:03f}, {accs[3]:03f} \r'
            )

            losses = list(map(float, losses))
            total_losses += total_loss.item()

        return round(total_losses / len(training_data), 3), [round(x.item() / len(training_data), 3)
                                                             for x in total_acc]

    def save_checkpoint(self, epoch, best_acc, valid_acc,
                        valid_loss, train_loss, is_best, filename):
        """
        Saving checkpoint to a file.
        """
        state = {
            'epoch': epoch + 1,
            'state_dict': self.midibert.state_dict(),
            'best_acc': best_acc,
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'optimizer': self.optim.state_dict()
        }

        torch.save(state, filename)

        best_mdl = filename.split('.')[0] + '_best.ckpt'
        if is_best:
            shutil.copyfile(filename, best_mdl)
