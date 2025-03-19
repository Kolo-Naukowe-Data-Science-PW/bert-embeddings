# Symbolic Piano Music Encoder

This project implements a Transformer-based encoder inspired by the BERT pre-training framework for symbolic piano music. The primary focus is on building a robust encoder that correctly processes MIDI data using a compound token (CP) representation. A REMI-based encoding scheme is planned for future implementation.

## Overview

This encoder is designed following ideas from the paper *BERT-like Pre-training for Symbolic Piano Music Classification 
Tasks* by Chou et al. It leverages a BERT-Base architecture consisting of 6 Transformer encoder layers. 
The goal is to capture the rich structure of symbolic music through self-supervised learning with masked token
prediction, focusing on encoding the musical information in a compact and efficient manner.

### Key aspects:

- **Focus on Encoding:** The project is centered on implementing the encoder. No downstream classification tasks are 
integrated yet.
- **CP Encoding:** The current implementation uses the Compound Token (CP) representation. The CP method groups
related tokens (Bar, Position, Pitch, Duration) into a single super token, 
reducing sequence length and potentially improving contextual learning.
- **REMI Encoding (Planned):** While REMI encoding is discussed in the literature and in the paper, it is not yet 
implemented in this project.

## Data Representation

### CP Encoding

#### Token Grouping:
In the CP representation, tokens are grouped into super tokens. For instance, for MIDI scores, a super token consists of:
- **Bar Token:** Marks the beginning or continuation of a bar.
- **Position Token:** Indicates a discrete position within the bar.
- **Pitch Token:** Represents the MIDI pitch value.
- **Duration Token:** Specifies the note duration in quantized steps.

### REMI Encoding (To Be Implemented)

- **Token Sequence:** REMI represents musical events as individual tokens in sequence 
(e.g., separate tokens for Bar, Sub-bar, Pitch, Duration, etc.).
- **Future Work:** Implementation of REMI encoding will be added to compare with the CP-based approach.

## Model Architecture

The encoder is built on a BERT-Base-like architecture with the following specifications:

- **Layers:** 6 Transformer encoder layers.
- **Attention Heads:** 6 attention heads per layer.
- **Hidden Dimension:** 384 units.
- **Input Embedding:** Each token (or super token in CP) is first mapped to an embedding vector and then augmented 
with a relative positional encoding.
- **Masked Language Modeling (MLM):** The encoder is pre-trained using the MLM objective where 
15% of the tokens (or super tokens) are randomly masked and predicted from context.
- **Loss Weighting:** During training, the loss for each token type is weighted based on its
vocabulary size.

The implementation uses the [HuggingFace Transformers](https://github.com/huggingface/transformers) library as 
a backbone for model components and training utilities.

## Data preparation
For details check:
```
python -m src.data_preparation.main -h
```

## Training
For details check:
```bash
python -m src.midibert.main -h
```

## Project Structure

```
.
├── data
│ ├── cp_data
│ └── dataset
├── MidiBERT
├── README.md
└── src
    ├── constants.py
    ├── data_preparation
    │ ├── dict
    │ │ ├── cp_dict.pkl
    │ │ ├── display_dict.py
    │ │ ├── __init__.py
    │ │ └── make_dict.py
    │ ├── encodings
    │ │ ├── cp.py
    │ │ ├── __init__.py
    │ │ └── prtint_cp_encoding.py
    │ ├── __init__.py
    │ ├── main.py
    │ └── utils.py
    ├── __init__.py
    └── midibert
        ├── bert_trainer.py
        ├── __init__.py
        ├── main.py
        ├── midi_bert_lm.py
        ├── midi_bert.py
        └── midi_dataset.py
```

## Future Work

- **Implement REMI Encoding:** Add support for REMI token representation to compare its performance with CP encoding.
- **Downstream Tasks:** Integrate downstream classification tasks (e.g., melody extraction, style classification) to fine-tune the encoder.
- **Evaluation:** Evaluate the encoder on various symbolic music datasets to assess its performance and generalization capabilities.