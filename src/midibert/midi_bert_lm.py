"""
This module defines the MidiBertLM class, a language modeling head built on top of a pretrained
MidiBert model for masked language modeling tasks on symbolic music data. The module is designed
to predict masked tokens within a sequence of MIDI events.

TODO: Currently masking whole cp events, yet to test masking specific attributes of the whole bar.
"""
from torch import nn

from src.midibert.midi_bert import MidiBert

class MidiBertLM(nn.Module):
    """
    Language modeling head for MidiBert to perform masked language modeling on MIDI sequences.
    """
    def __init__(self, midibert: MidiBert):
        super().__init__()

        self.midibert = midibert
        self.mask_lm = MLM(self.midibert.e2w, self.midibert.n_tokens, self.midibert.hidden_size)

    def forward(self, x, attn):
        """
        Forward pass through MidiBert and the masked language modeling head.

        Args:
            x: input tensor of shape (batch_size, seq_len, n_features)
            attn: attention mask tensor of shape (batch_size, seq_len, seq_len)
        """
        x = self.midibert(x, attn)
        return self.mask_lm(x)


class MLM(nn.Module):
    """
    Masked Language Modeling (MLM) head for predicting token logits from MidiBert outputs.
    """
    def __init__(self, e2w, n_tokens, hidden_size):
        super().__init__()

        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, _ in enumerate(e2w):
            self.proj.append(nn.Linear(hidden_size, n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)

        self.e2w = e2w

    def forward(self, y):
        """
        Forward pass to project hidden states to logits for each token type.

        Args:
            y (ModelOutput): Output of MidiBert, containing hidden states.
        """
        # feed to bert
        y = y.hidden_states[-1]

        # convert embeddings back to logits for prediction
        ys = []
        for i, _ in enumerate(self.e2w):
            ys.append(self.proj[i](y))  # (batch_size, seq_len, dict_size)
        return ys
