"""Script for running inference with a trained model."""

# pylint: disable=C3001,R0914,R0913,R0917,C0413
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.train import CustomModel, infer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  device = {DEVICE}")

MODEL_PATH = "models/model_latest.pth"  # Path to your trained model


def load_model(ckpt_path: str):
    """Loads a trained CustomModel and its configuration."""
    if not os.path.exists(ckpt_path):
        print(f"Error: Model checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # Load model configuration from checkpoint
    seq_len = ckpt.get("seq_len")
    hidden_size = ckpt.get("hidden_size")
    num_layers = ckpt.get("num_layers")
    num_classes = ckpt.get("num_classes")
    dropout = ckpt.get("dropout")
    vocab_size = ckpt.get("vocab_size")
    embedding_dim = ckpt.get("embedding_dim")

    if any(
        v is None
        for v in [
            seq_len,
            hidden_size,
            num_layers,
            num_classes,
            dropout,
            vocab_size,
            embedding_dim,
        ]
    ):
        print(
            "Error: Checkpoint is missing one or more required model parameters "
            "(seq_len, hidden_size, num_layers, num_classes, dropout, vocab_size, embedding_dim).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Instantiate CustomModel
    model_ = CustomModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    ).to(DEVICE)
    model_.load_state_dict(ckpt["model_state"])
    model_.eval()
    print(f"Model loaded from {ckpt_path} and set to evaluation mode.")
    return model_, seq_len


if __name__ == "__main__":
    DEMO_SAMPLES = [
        [0, 2, 6, 3, 3],  # Expected: 14
        [7, 8, 4, 2, 9],  # Expected: 30
        [1, 7, 5, 4, 1],  # Expected: 18
        [2, 2, 2, 2, 2],  # Expected: 10
        [9, 8, 7, 6, 5],  # Expected: 35
        [1, 1, 1, 1, 1],  # Expected: 5
        [5, 5, 5, 5, 5],  # Expected: 25
        [0, 0, 0, 0, 0],  # Expected: 0
        [3, 4, 5, 6, 7],  # Expected: 25
        [8, 1, 3, 7, 2],  # Expected: 21,
        [4, 4, 4, 4, 4],  # Expected: 20,
        [9, 0, 0, 0, 0],  # Expected: 9,
    ]
    EXPECTED_OUTPUTS = [14, 30, 18, 10, 35, 5, 25, 0, 25, 21, 20, 9]

    model, model_seq_len = load_model(MODEL_PATH)

    predictions, actual_samples_processed = [], []
    for i, sample_features in enumerate(DEMO_SAMPLES):
        if len(sample_features) != model_seq_len:
            print(
                f"Warning: Sample {i} has {len(sample_features)} features, "
                f"but model expects {model_seq_len}. Skipping this sample."
            )
            continue
        try:
            prediction = infer(model, sample_features, model_seq_len)
            predictions.append(prediction)
            actual_samples_processed.append(sample_features)
        except Exception as e:
            raise RuntimeError(
                f"Error during inference for sample {sample_features}: {e}"
            ) from e

    print("\n--- Results ---")
    for sample, pred, expected in zip(
        actual_samples_processed, predictions, EXPECTED_OUTPUTS
    ):
        if pred is not None:
            STATUS = "✅" if pred == expected else "❌"
            print(
                f"Input: {sample} -> Predicted: {pred}, Expected: {expected} {STATUS}"
            )
        else:
            print(f"Input: {sample} -> Prediction failed (see errors above)")
        print("-" * 30)
