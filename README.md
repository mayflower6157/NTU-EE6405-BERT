# Building BERT from Scratch (Modernized) with Python 3.12 and PyTorch 2.6+

Build BERT from scratch and compare different BERT variants using a modernized implementation.

# Building BERT with PyTorch from scratch

![img](https://uploads-ssl.webflow.com/60100d26d33c7cce48258afd/6244769a9ec65d641e367414_BERT%20with%20PyTorch.png)

This is the repository containing the code for a modernized implementation of BERT, featuring upgraded dependencies, CUDA 12.6 support, and a structured configuration for comparing different BERT variants.

[Building BERT with PyTorch from scratch](https://coaxsoft.com/blog/building-bert-with-pytorch-from-scratch)

## Installation

This repository supports both Conda and pip installation workflows for flexibility. You can install the required dependencies via one of the options below. Both are compatible with Python 3.12 and support GPU acceleration on CUDA 12.6.

### Option 0: Conda (Reproducible Research Setup)

This is the recommended setup for consistent environments, especially for GPU clusters and reproducible experiments.

```bash
mamba env create -f environment.yml
conda activate bert
```

This setup uses Python 3.12 and PyTorch 2.6 with CUDA 12.6.

### Option 1: Mac (Apple Silicon)

For Mac users with M1/M2/M3 chips, use the provided `environment_mac.yml` file for best compatibility:

```bash
mamba env create -f environment_mac.yml
conda activate bert-mac
```

This version uses the Metal (MPS) backend for GPU acceleration instead of CUDA.

You can verify MPS availability with the following code:

```python
import torch
print("MPS available:", torch.backends.mps.is_available())
```

For quick local testing and debugging, start with the Mac environment; for large-scale pretraining, switch to the CUDA environment on a GPU cluster.

### Option 2: Pip (Quick Setup)

```shell
conda create -n bert python=3.12
conda activate bert
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**Tip:** Use `environment.yml` when running on NTU GPU clusters or cloud machines for full reproducibility.

## Quick Start

Verify your environment and run a small tokenizer training snippet with the following example:

```python
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
sample_text = "Hello, BERT!"
encoded = tokenizer(sample_text)
print(encoded)
```
```python
import torch
import torchtext


print("🔹 PyTorch version:", torch.__version__)
print("🔹 Torchtext version:",torchtext.__version__)
print("🔹 CUDA available:", torch.cuda.is_available())
print("🔹 MPS available:", torch.backends.mps.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon MPS backend.")
else:
    device = torch.device("cpu")
    print("🧠 Falling back to CPU.")

# sanity check tensor
x = torch.rand(2, 2).to(device)
print(f"\nTensor created on: {x.device}")



```

This will ensure your tokenizer is working correctly before proceeding with full model training.
