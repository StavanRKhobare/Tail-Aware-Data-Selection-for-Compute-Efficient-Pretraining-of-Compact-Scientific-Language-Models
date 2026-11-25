# Transformer trained from scratch

TailLab's custom model is a GPT-style decoder-only Transformer implemented directly in PyTorch. The 17M and 34M runs both initialize their weights randomly and train with next-token prediction over scientific paper titles and abstracts. The model and checkpoint loader live in [src/model.py](../src/model.py).

## Model configurations

| Component | 17M controlled experiment | 34M larger baseline |
|---|---:|---:|
| Trainable parameters | 17,135,616 | 33,870,848 |
| Decoder blocks | 6 | 8 |
| Hidden width | 384 | 512 |
| Attention heads | 6 | 8 |
| Dimensions per head | 64 | 64 |
| Feed-forward width | 1,536 | 2,048 |
| Maximum context | 512 tokens | 512 tokens |
| Vocabulary | 16,384 | 16,384 |
| Dropout | 0.0 | 0.0 |
| Training tokens processed per completed run | 59,998,208 | 119,996,416 |

The 17M experiment holds architecture, optimizer, token budget, and validation split fixed to compare data-selection strategies. The 34M baseline uses random selection at seed 42 and changes both capacity and training duration.

## Forward pass

1. Add a learned token embedding and a learned absolute position embedding at each input position.
2. In each block, apply LayerNorm and causal multi-head self-attention, then add the input residual. A single projection creates queries, keys, and values. PyTorch scaled dot-product attention blocks access to future tokens.
3. Apply a second LayerNorm and a feed-forward network with GELU and an inner width of four times the hidden width. Add the second residual.
4. Apply a final LayerNorm and a bias-free vocabulary projection tied to the token embedding matrix. The model returns logits of shape `[batch, sequence, vocabulary]`.

Linear and embedding weights are initialized from a normal distribution with standard deviation 0.02. Linear biases start at zero. The attention and MLP output projections are scaled by `1 / sqrt(2 × number of blocks)` at initialization.

## Data and objective

The fixed [byte-level BPE tokenizer](../tokenizer/tokenizer.json) is trained on the corpus training split. Each paper is represented as a title, a blank line, and an abstract. [Token export](../src/export_tokens.py) appends one end-of-text token per paper before concatenating the tokens into each subset's binary stream.

[Token windows](../src/data.py) read a 512-token sequence and its one-token-shifted targets. Windows may span document boundaries, with the end-of-text token marking the transition. Cross-entropy measures next-token prediction against the targets. The training loop uses AdamW, gradient clipping, linear warmup, and cosine decay. Run configuration and checkpoint details are in the [training workflow](../PART3.md).

## Generation and drafting

The [generation entry point](../src/generate.py) uses the checkpoint configuration to reconstruct the architecture. Prompt continuation begins with the encoded prompt, while document generation begins with one end-of-text token. Both generate autoregressively until an end-of-text prediction or the generation limit. The Generation Lab selects a 17M checkpoint by data-selection strategy and training seed.

[Draft Workspace](phase1_drafting_application.md) also exposes supervised drafting checkpoints initialized from the custom models. These are separate from the original next-token pretraining runs. SmolLM2 continued pretraining uses a separate pretrained architecture and tokenizer.
