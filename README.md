# GPT Lyrics Transformer

A decoder-only Transformer language model built from scratch in PyTorch, trained on poetry and hip-hop lyrics for structured and conditional creative text generation.

The project implements the core components of the GPT architecture manually (multi-head causal self-attention, feed-forward sublayers with GELU, pre-layer normalization, and weight tying) along with ablation benchmarks, tokenizer trade-off comparisons, and conditional decoding controls.

## Architecture

- **Model Type:** Autoregressive decoder-only Transformer
- **Context Window:** 256 tokens (configurable up to 1024)
- **Embedding Dimension:** 256
- **Attention Heads:** 4 heads (64 dimensions per head)
- **Transformer Layers:** 4 pre-norm layers
- **Feed-Forward Multiplier:** 4x expansion with GELU activation
- **Weight Tying:** Shared weights between input token embeddings and final projection head
- **Initialization:** Normal distribution $N(0, 0.02)$

## Features

- **Dual Tokenization Pipelines:**
  - Byte-Pair Encoding (BPE) via OpenAI `tiktoken` (GPT-2 vocabulary, 50,257 tokens).
  - Custom character-level tokenizer (1,747 unique characters) for comparing parameter efficiency and perplexity against subword tokenization.
- **Conditional Generation:**
  - Rap lyric dataset entries are formatted with conditioning headers:
    `[Genre: <genre>] [Mood: <mood>] [Rhyme: <rhyme_scheme>] [Cadence: <cadence>]`
  - Allows steered text generation during inference based on style metadata.
- **Controllable Sampling:**
  - Temperature scaling, top-k filtering, nucleus (top-p) sampling, and repetition penalty to mitigate degeneracies and repetitive loops.
- **Ablation Studies:**
  - Configurable switches to toggle residual connections, layer normalization, causal masking, and positional embeddings to evaluate model degradation.
  - Hyperparameter grid sweeps across learning rates, layer counts, and embedding dimensions.
- **Analysis and Visualizations:**
  - Attention matrix extraction across layers for attention map inspection.
  - PCA projections of token embedding geometries before and after training.

## Repository Layout

```
.
├── model/
│   └── model.py                 # TransformerConfig, MaskedAttention, TransformerLayer, NanoTransformer
├── tokenizer/
│   ├── tokenization.py          # BPE pipeline with metadata prefixing
│   ├── char_tokenizer.py        # Character-level tokenizer
│   ├── char_stoi.json           # Character-to-index mapping
│   └── char_itos.json           # Index-to-character mapping
├── training/
│   └── training.ipynb           # Sequential training pipeline (Poetry corpus -> Rap corpus)
├── evaluation/
│   ├── eval_4.ipynb             # Inference benchmarking, decoding strategies, edge-case tests
│   └── dataset_analysis.ipynb   # Corpus statistics and token length distributions
├── experiments/
│   ├── scaling_experiments.ipynb      # Parameter scaling laws and architecture ablations
│   ├── attention_visualization.ipynb  # Self-attention map extraction
│   ├── embedding_visualization.ipynb  # PCA projections of learned embeddings
│   ├── failure_testing.ipynb          # Adversarial prompts and repetition edge-case tests
│   └── data_cleaning_experiment.ipynb # Impact of text sanitization on tokenization
├── data/
│   ├── poems/                   # Poetry training corpus
│   └── raps/                    # Cleaned rap lyric corpus with metadata annotations
└── outputs/                     # Memory-mapped binary token files (.bin)
```

## Setup and Usage

### Prerequisites

```bash
pip install torch tiktoken numpy pandas matplotlib scikit-learn
```

### Preprocessing and Tokenization

Generate the memory-mapped binary arrays for training:

```bash
python tokenizer/tokenization.py
```

### Training

The training workflow in `training/training.ipynb` loads memory-mapped data chunks using `np.memmap` for efficient GPU streaming:

- Optimizer: AdamW (`lr=3e-4`, weight decay, gradient clipping at 1.0)
- Staged training schedule: initial convergence on poetry corpus followed by fine-tuning on conditioned rap lyrics.

### Inference

```python
import torch
from model.model import NanoTransformer, TransformerConfig
from tokenizer.tokenization import BPETokenizer

config = TransformerConfig(vocab_size=50257, block_size=256, n_layer=4, n_head=4, n_embd=256)
model = NanoTransformer(config)
model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
model.eval()

tokenizer = BPETokenizer()
prompt = "[Genre: 90s Hip-Hop] [Mood: Reflective] The city lights"
input_ids = tokenizer.encode(prompt)

output_ids = model.generate(
    input_ids,
    max_new_tokens=150,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.15
)
print(tokenizer.decode(output_ids))
```
