"""
This module defines the core components of the MidiBERT model for symbolic music representation
learning. MidiBERT is a BERT-based transformer model adapted for MIDI data, where musical events
(e.g., Bar, Position, Pitch, Duration) are treated as distinct token types. The module includes
custom embedding layers for multi-field MIDI tokens and integrates them into a unified
representation for input to the BERT encoder.

Classes:
    - Embeddings: Embedding layer for individual MIDI token types, scaled appropriately.
    - MidiBert: The main transformer-based encoder that processes MIDI event sequences
      with support for masked language modeling.
"""

import math
import random

import numpy as np
import torch
from torch import nn
from transformers import BertModel

from src.constants import EMBEDDING_SIZE

class Embeddings(nn.Module):
    """
    Embeddings module for the MidiBERT
    """
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Forward pass for the Embeddings module. Multiplies the resulting embedding by the
        square root of the model dimension to balance the magnitude of the embeddings.
        """
        return self.lut(x) * math.sqrt(self.d_model)


# BERT model: similar approach to "felix"
# pylint: disable=too-many-instance-attributes
class MidiBert(nn.Module):
    """
    This model adapts the BERT architecture to handle structured MIDI data by embedding
    each event type (Bar, Position, Pitch, Duration) separately, merging their representations,
    and feeding them into a shared transformer encoder.
    """
    def __init__(self, bert_config, e2w, w2e):
        super().__init__()

        # Bert model initialization
        self.bert = BertModel(bert_config)
        bert_config.d_model = bert_config.hidden_size
        self.hidden_size = bert_config.hidden_size
        self.bert_config = bert_config

        # token types: [Bar, Position, Pitch, Duration]
        self.n_tokens = []  # [3,18,88,66]
        self.classes = ['Bar', 'Position', 'Pitch', 'Duration']

        # Number of tokens for each class
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))

        # Embedding sizes for each class
        # (every token class is embedded into a space of the same size)
        self.emb_sizes = [EMBEDDING_SIZE] * len(self.classes)

        # Token to word and word to token mappings
        self.e2w = e2w
        self.w2e = w2e

        # For deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']

        # Array containing the token ids for masking
        self.mask_word_np = np.array([self.e2w[etype][f'{etype} <MASK>'] for etype in self.classes],
                                     dtype=np.int64)

        # Array containing the token ids for padding
        self.pad_word_np = np.array([self.e2w[etype][f'{etype} <PAD>'] for etype in self.classes],
                                    dtype=np.int64)

        # Embedding modules to change token ids into embeddings for each class
        self.word_emb = []
        for i, key in enumerate(self.classes):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # Linear layer to merge embeddings from different token types
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bert_config.d_model)

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        """
        Forward pass for the MidiBERT model. Converts input_ids into embeddings and merges them.
        """
        embs = []
        for i, _ in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # feed to bert
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask,
                      output_hidden_states=output_hidden_states)
        return y

    def get_rand_tok(self):
        """
        Get a random token for the MidiBERT model. Used in the MLM process.
        """
        c1, c2, c3, c4 = self.n_tokens[0], self.n_tokens[1], self.n_tokens[2], self.n_tokens[3]
        return np.array([random.choice(range(c1)), random.choice(range(c2)),
                         random.choice(range(c3)), random.choice(range(c4))])
