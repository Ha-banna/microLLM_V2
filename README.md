# microLLM_V2

A lightweight, high-performance **Character-Level Transformer** built from scratch using **PyTorch**. This project implements a full Decoder-only architecture (similar to GPT) with Multi-Head Self-Attention, Feed-Forward networks, and Residual connections.

## 🚀 Features
*   **Architecture:** Multi-Head Self-Attention with $O(T^2)$ causal masking.
*   **Optimization:** Support for `torch.compile`, **Flash Attention**, and **Mixed Precision (AMP)** for maximum GPU utilization.
*   **Scalability:** Configurable embedding dimensions, layer depth, and block size.
*   **Device Agnostic:** Seamlessly runs on **CUDA** (NVIDIA), **ROCm** (AMD), or CPU.
*   **Cloud Ready:** Designed with a clean backend structure suitable for containerization via **Docker** and orchestration on **Kubernetes**.

---

## 📈 Model Architecture
The model follows the modern Transformer design:
1.  **Token & Position Embeddings:** Character-level mapping with learned positional encoding.
2.  **Transformer Blocks:** 
    *   **Layer Norm:** Pre-norm formulation for training stability.
    *   **Multi-Head Attention:** Parallel attention heads to capture diverse linguistic patterns.
    *   **Feed-Forward Network:** Position-wise MLP with ReLU/GELU activations.
3.  **Language Model Head:** Final linear layer mapping embeddings back to vocabulary logits.

---

## 🏃 Getting Started

### Prerequisites
*   Python 3.10+
*   Supported GPU (NVIDIA RTX 30+ or AMD ROCm-compatible)

### Installation
```bash
git clone https://github.com/your-username/microLLM_V2.git
cd microLLM_V2
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Training
Place your training data in a file named `wizard of oz.txt` (or update the script path) and run:
```bash
python final.py
```

---

## ⚙️ Configuration
You can tune the hyperparameters in `final.py` to match your VRAM capacity:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `n_embed` | 512 | Embedding dimension (width) |
| `n_layer` | 6 | Number of Transformer blocks (depth) |
| `n_head` | 8 | Number of attention heads |
| `block_size` | 256 | Context window length |
| `batch_size` | 64 | Number of sequences per batch |

---

## 🧪 Evaluation & Generation
The project includes a `.generate()` method that uses **Top-K sampling** or **Multinomial sampling** to produce text. It automatically handles context cropping to prevent positional embedding overflow.

---

## 📄 License
MIT License - feel free to use this for your own experiments!