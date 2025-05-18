"""Pytorch template"""

# pylint: disable=C3001,R0914,R0913,R0917,C0413
import os
import time
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATA_PATH = "Datasets/synth_i5_r0-9_n-1000.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  device = {DEVICE}")

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
DROPOUT = 0.1

SEQ_LEN = 5
VOCAB_SIZE = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EMBEDDING_DIM = VOCAB_SIZE
NUM_CLASSES = SEQ_LEN * VOCAB_SIZE


class ModelDataset(Dataset):
    """Dataset from a CSV file."""

    def __init__(self, df_: pd.DataFrame, input_cols: list[str], target_col: str):
        self.features = torch.tensor(df_[input_cols].values, dtype=torch.long)
        self.labels = torch.tensor(df_[target_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def collate_fn(batch):
    """
    Collates a batch of features and labels for model.
    features: (B, num_features)
    labels: (B)
    """
    features, labels = zip(*batch)
    features_tensor = torch.stack(features)
    labels_tensor = torch.stack(labels)
    return features_tensor, labels_tensor


class CustomModel(nn.Module):
    """An LSTM-based model for sequence classification with embedding."""

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x shape: (B, seq_len). Elements are token indices.
        The data pipeline typically prepares sequences of length seq_len.
        output shape: (B, num_classes) - raw logits
        """
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)  # embedded shape: (batch, seq_len, embedding_dim)

        # lstm_out shape: (batch, seq_len, lstm_hidden_size)
        # hn shape: (num_lstm_layers, batch, lstm_hidden_size)
        # cn shape: (num_lstm_layers, batch, lstm_hidden_size)
        lstm_out, (hn, cn) = self.lstm(embedded)

        # For sequence classification, we need a fixed-size representation for each input sequence.
        # The LSTM (lstm_out) produces a hidden state for each time step in the input sequence.
        # We select the hidden state from the *last time step* for every sequence in the batch.
        # This 'last_time_step_output' serves as a summary of the entire input sequence
        # and is used as input to the final classification layer.
        last_time_step_output = lstm_out[:, -1, :]  # shape: (batch, lstm_hidden_size)
        logits = self.fc(last_time_step_output)
        return logits


def custom_loss_function(logits, targets):
    """
    Wrapper function for calculating the loss.
    Instantiates and uses CrossEntropyLoss internally.
    """
    criterion = nn.CrossEntropyLoss()
    loss_ = criterion(logits, targets)
    return loss_


def train_epoch(model_, loader, optimizer_):
    """Trains the model for one epoch and computes loss and accuracy."""
    model_.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)

        optimizer_.zero_grad()
        logits = model_(features)  # (B, num_classes)
        loss = custom_loss_function(logits, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model_.parameters(), 1.0)  # Optional: gradient clip
        optimizer_.step()

        # loss.item() is avg loss for batch
        total_loss += loss.item() * features.size(0)

        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def eval_epoch(model_, loader):
    """Evaluates the model and computes loss and accuracy."""
    model_.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)

        logits = model_(features)
        loss = custom_loss_function(logits, targets)

        total_loss += loss.item() * features.size(0)

        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def infer(model_, feature_vector: list, seq_len: int):
    """
    Predicts the class for a single feature vector.
    'seq_len' here refers to the expected sequence length of the feature_vector,
    which should match the seq_len global constant used during training.
    """
    model_.eval()
    if len(feature_vector) != seq_len:
        raise ValueError(
            f"Input feature vector length {len(feature_vector)} does not match model's expected seq_len {seq_len}"
        )

    # Input features are now indices for the embedding layer
    features_tensor = (
        torch.tensor(feature_vector, dtype=torch.long).unsqueeze(0).to(DEVICE)
    )
    logits = model_(features_tensor)
    prediction_ = torch.argmax(logits, dim=1).item()
    return prediction_


if __name__ == "__main__":
    start_time = time.time()
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} samples from {DATA_PATH}")

    # Define input and target column names based on your CSV
    INPUT_COL_NAMES = [f"input_{i+1}" for i in range(SEQ_LEN)]
    TARGET_COL_NAME = "target"

    # 1. Split data
    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=None,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # 2. Dataset / DataLoader
    train_ds = ModelDataset(
        df_=train_df, input_cols=INPUT_COL_NAMES, target_col=TARGET_COL_NAME
    )
    val_ds = ModelDataset(
        df_=val_df, input_cols=INPUT_COL_NAMES, target_col=TARGET_COL_NAME
    )
    train_dl = DataLoader(
        dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_dl = DataLoader(
        dataset=val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # 3. Model / Optim
    model = CustomModel(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"\nStarting training for {EPOCHS} epochs on {DEVICE}...")
    for ep in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        tr_loss, tr_acc = train_epoch(model, train_dl, optimizer)
        vl_loss, vl_acc = eval_epoch(model, val_dl)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accuracies.append(tr_acc)
        val_accuracies.append(vl_acc)

        epoch_end_time = time.time()
        epoch_duration_seconds = int(epoch_end_time - epoch_start_time)
        epoch_minutes, epoch_seconds = divmod(epoch_duration_seconds, 60)

        print(
            f"Epoch {ep:02d}/{EPOCHS} │ "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} │ "
            f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f} │ "
            f"Time: {epoch_minutes}m {epoch_seconds}s"
        )

    # 5. Save
    os.makedirs("models", exist_ok=True)
    os.makedirs("logging", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    TS_MODEL_PATH = f"models/model_{ts}.pth"
    LATEST_MODEL_PATH = "models/model_latest.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seq_len": SEQ_LEN,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_classes": NUM_CLASSES,
            "dropout": DROPOUT,
            "vocab_size": VOCAB_SIZE,
            "embedding_dim": EMBEDDING_DIM,
        },
        TS_MODEL_PATH,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "seq_len": SEQ_LEN,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_classes": NUM_CLASSES,
            "dropout": DROPOUT,
            "vocab_size": VOCAB_SIZE,
            "embedding_dim": EMBEDDING_DIM,
        },
        LATEST_MODEL_PATH,
    )

    # 6. plot loss & Accuracy
    TS_METRICS_PATH = f"logging/metrics_{ts}.png"
    LATEST_METRICS_PATH = "logging/metrics_latest.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs_range = range(1, EPOCHS + 1)  # Corrected variable name
    ax1.plot(epochs_range, train_losses, label="Train Loss")
    ax1.plot(epochs_range, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accuracies, label="Train Accuracy")
    ax2.plot(epochs_range, val_accuracies, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    # Add a title to the figure
    script_name = os.path.basename(__file__)
    fig.suptitle(
        f"{script_name}\n{DATA_PATH}\nEpochs: {EPOCHS}",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle
    plt.savefig(TS_METRICS_PATH)
    plt.savefig(LATEST_METRICS_PATH)
    plt.close(fig)

    # runtime
    total_seconds = int(time.time() - start_time)
    minutes, seconds = divmod(total_seconds, 60)
    print(f"\nTotal runtime: {minutes}m {seconds}s")

    # 7. Demo
    print("\n--- Demo Inference ---")
    if SEQ_LEN == 5:
        demo_samples = [
            [0, 2, 6, 3, 3],  # Expected target: (0+2+6+3+3) = 14
            [7, 8, 4, 2, 9],  # Expected target: (7+8+4+2+9) = 30
            [1, 7, 5, 4, 1],  # Expected target: (1+7+5+4+1) = 18
        ]
        for sample_features in demo_samples:
            try:
                prediction = infer(model, sample_features, SEQ_LEN)
                print(f"Input: {sample_features} -> Predicted class: {prediction}")
            except ValueError as e:
                print(f"Error during demo prediction for {sample_features}: {e}")
    else:
        print(
            f"Demo samples are for seq_len=5. Current seq_len is {SEQ_LEN}. Skipping demo."
        )
