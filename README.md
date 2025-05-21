
# 🦙  Llama 4: (text model)

This repository provides all the code and configurations required to train **Llama 4**, a transformer-based language model, on the **TinyStories** dataset. The model is designed to generate short, coherent, and child-friendly narratives. It is implemented in **PyTorch**, with optional **TensorFlow** support, and leverages **CUDA** acceleration.

> For more details on the official Llama 4 multimodal model by Meta, visit: [Meta AI Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

---
![](./expert.png)
## 📌 Project Overview

**Llama 4 (TinyStories Edition)** features approximately **771 million parameters**, optimized for training on a single GPU. The model is trained over **4 epochs**, each with **10,000 steps**, using a **sequence length of 128** and a **vocabulary size of 10,000**.

* ✅ Final training loss: **1.6609**
* ✅ Partial validation loss (26%): **1.4992**

Text quality improves dramatically from incoherent outputs in **Epoch 1** to fluent and emotionally rich narratives by **Epoch 4**.

---

## 📚 Dataset Details

### TinyStories Dataset

**Training Set:**

* Tokens: 464,965,814
* Processed in streaming mode
* Sequence Length: 128
* Batch Size: 16

**Validation Set:**

* Tokens: 4,673,588
* Non-streaming mode
* Total Batches: 292,092
* Sequence Length: 128
* Batch Size: 16

**Tokenizer:** Custom BPE tokenizer

* `pad_token_id = 0`
* Vocabulary Size: 10,000

---

## 🛠️ Model Configuration

```python
model_args = Llama4TextConfig(
    vocab_size=10000,
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=12,
    num_attention_heads=32,
    num_key_value_heads=8,
    moe_layers=[6],
    num_local_experts=6,
    max_position_embeddings=128,
    use_flash_attention=False,
    attention_dropout=0.1,
    initializer_range=0.02,
)
```

### Architecture

* Transformer-based with 12 layers
* One Mixture of Experts (MoE) layer at Layer 6
* 6 local experts
* 32 attention heads, 8 key-value heads
* Dropout: 0.1

---

## 🏋️ Training Configuration

```python
cfg = TrainingConfig(
    batch_size=16,
    seq_len=128,
    epochs=4,
    steps_per_epoch=10000,
    report_interval=1000,
    grad_clip_norm=1.0,
    learning_rate=1e-4,
    warmup_steps=10,
    max_lr=1e-4,
    min_lr=1e-5,
    log_interval=1000
)
```

* Gradient Clipping: 1.0
* Learning Rate: Warmup to `1e-4`, decay to `1e-5`
* Logging every 1000 steps

---

## 📊 Training Results

* Speed: \~3.25 steps/sec (CUDA-enabled GPU)

| Epoch | Avg Loss | Start Loss | End Loss             |
| ----- | -------- | ---------- | -------------------- |
| 1     | 2.4755   | 2.8820     | 1.9504               |
| 2     | \~       | \~         | 1.8973 (@ step 8000) |
| 3     | 1.7500   | 1.9057     | 1.7055               |
| 4     | 1.6609   | 1.7914     | 1.5687               |

* Partial Validation Loss: **1.4992** (at step 76,401 of 292,092)

---

## ✨ Sample Outputs

### Epoch 1

> "Jack was the bravest elephant..." (incoherent and inconsistent)

### Epoch 3

> "A yellow bird with an arrow in his hands." (clearer but still odd)

### Epoch 4

> *"Hello world! She saw lots of beautiful flowers, bright green grass, and bright blue birds. As Sarah stepped into the garden, she became very thirsty..."*

---

## 📦 Requirements

* Python ≥ 3.8
* PyTorch
* TensorFlow
* CUDA-enabled GPU
* Install dependencies:

```bash
pip install torch tensorflow tqdm
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/llama4-tinystories.git
cd llama4-tinystories
```

### 2. Prepare the Dataset

* Download TinyStories dataset and place it under `data/`
* Tokenize with `vocab_size=10000`

### 3. Configure Environment

* Ensure CUDA is available
* (Optional) Install TensorRT for fast inference

### 4. Run Training

```bash
python main.py
```

---

## ⚠️ Notes

* **Experimental Features:** PyTorch's `ComplexHalf` is still experimental
* **Inference:** TensorRT can boost performance if configured correctly
* **Vocabulary Limit:** 10,000 tokens is optimal for TinyStories but may limit general use

---

## 🔮 Future Work

* Expand vocabulary size for broader generalization
* Complete validation over full dataset
* Experiment with learning rate schedules and longer training
* Optimize validation efficiency (fewer batches or less frequency)

---

## 📄 License

Licensed under the [MIT License](LICENSE).
