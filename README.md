# 🧪 PyTorch Fast Prototyping Template — with Synthetic Data + Vibe Coding Guide

Welcome to the **Ultimate PyTorch Template** for fast machine learning prototyping. This repo is designed to help you quickly test deep learning architectures on rule-based synthetic datasets *before* investing time in real-world data pipelines.

---

## 🔍 Overview

This repo contains:

* ✅ A **synthetic dataset generator** based on a simple summation rule.
* 🧠 A **modular PyTorch training template** using an LSTM as the default model.
* ♻️ A **"vibe coding" strategy** to quickly convert this template into any ML model with the help of LLMs.

---

## 📦 Features

* **Synthetic Dataset Generator**
  Generate sequence data where each input is a vector of integers and the label is the sum. Easy to modify for other rule-based tasks.

* **Modular PyTorch Training Pipeline**
  Includes training, validation, inference, loss/accuracy plotting, and model saving.

* **Drop-in Model Conversion**
  Easily swap the LSTM model with another architecture using LLM-generated drop-in code blocks.

* **Minimal Template Edits**
  Promotes minimal, traceable changes to maintain code clarity and reduce breakage.

---

## 🧬 Synthetic Dataset

```python
# Example: [3, 5, 2, 1, 4] → target: 15
```

Change the rule (`sum(inputs)`) to anything: product, parity, XOR, etc.

```bash
python generate_dataset.py
```

Creates a CSV file with inputs and targets.

---

## 🚀 Run the Full Training Pipeline

Generate a synthetic dataset:

```bash
python src/synth_dataset_gen.py
```

Train the model:

```bash
python src/train.py
```

Run inference demo:

```bash
python src/infer.py
```

---

## 🔎 Inference Demo

At the end of training, the script runs a few demo predictions like:

```python
Input: [0, 2, 6, 3, 3] -> Predicted: 14
```

---

## 🧘‍♂️ Vibe Coding Strategy

Use this prompt to generate *drop-in code blocks* to convert the LSTM into any architecture:

> Please give me drop-in code snippets to change this LSTM-template model to a `<new_model>` to accomplish this goal: `<task>` with example data: `<csv snippet>`

Then, paste the snippets into Cursor (or your editor of choice) with a prompt like:

> Implement these suggested changes to convert the PyTorch template to `<new_model>`. Make minimal edits and preserve structure.

This approach prevents unnecessary edits and maximizes reuse.

---

## 🧠 Why This Template?

* Works out-of-the-box.
* Swappable model block.
* Helps isolate modeling issues *before* messy data is introduced.
* Encourages modular thinking and rapid iteration.
* Optimized for ML engineers, researchers, and educators alike.

---

## 🧩 Folder Structure

```
.
├── Datasets/
│   └── synth_i5_r0-9_n-1000.csv
├── models/
│   ├── model_latest.pth
│   └── model_<timestamp>.pth
├── logging/
│   ├── metrics_latest.png
│   └── metrics_<timestamp>.png
├── src/
│   ├── train.py
│   ├── synth_dataset_gen.py
│   ├── infer.py
│   └── __init__.py
├── tests/
│   ├── test_train.py
│   └── __init__.py
├── requirements.txt
└── README.md
```

---

## 📌 Dependencies

* PyTorch
* scikit-learn
* pandas
* matplotlib

Install with:

```bash
pip install torch scikit-learn pandas matplotlib
```

---

## ✅ Conclusion

This workflow helps you:

1. Prototype fast with synthetic data.
2. Validate architectures before deploying on real-world data.
3. Switch models effortlessly with LLM-powered vibe coding.

Happy experimenting! 🚀
