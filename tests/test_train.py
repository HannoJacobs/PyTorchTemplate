"""
Tests for the training pipeline components from src.train.
"""

# pylint: disable=E0401,E0611,E1101,C0413,R0913,R0914,C2801,W1203,C3001,R0917
import os
import sys
import copy

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.train import (
    ModelDataset,
    collate_fn,
    CustomModel,
    custom_loss_function,
    train_epoch,
    eval_epoch,
    infer,
    SEQ_LEN as SRC_seq_len,
    NUM_CLASSES as SRC_NUM_CLASSES,
    HIDDEN_SIZE as SRC_HIDDEN_SIZE,
    NUM_LAYERS as SRC_NUM_LAYERS,
    DROPOUT as SRC_DROPOUT,
    VOCAB_SIZE as SRC_VOCAB_SIZE,
    EMBEDDING_DIM as SRC_EMBEDDING_DIM,
)

# Define DEVICE locally for the test environment
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
print(f"🖥️ Test environment device = {DEVICE}")

# Test-specific constants for data generation
TEST_BATCH_SIZE = 2
TEST_NUM_SAMPLES = 4
TEST_LR = 1e-3


class TestModel:
    """Test class for the training pipeline components"""

    def test_model_dataset(self):
        """Test the ModelDataset class."""
        input_cols = [f"feature_{i}" for i in range(SRC_seq_len)]
        target_col = "target"
        # Human-readable data: features are simple ranges, targets are sums of features
        data = {col: [float(j) for j in range(TEST_NUM_SAMPLES)] for col in input_cols}
        # For row j, all features are j. Sum is SRC_seq_len * j.
        data[target_col] = [SRC_seq_len * i for i in range(TEST_NUM_SAMPLES)]
        df = pd.DataFrame(data)

        dataset = ModelDataset(df_=df, input_cols=input_cols, target_col=target_col)

        assert len(dataset) == TEST_NUM_SAMPLES, "Dataset length mismatch."

        features, label = dataset[0]
        assert isinstance(features, torch.Tensor), "Features should be a Tensor."
        assert features.shape == (
            SRC_seq_len,
        ), f"Feature shape mismatch. Expected ({SRC_seq_len},), got {features.shape}"
        assert features.dtype == torch.long, "Feature dtype mismatch for embedding."

        assert isinstance(label, torch.Tensor), "Label should be a Tensor."
        assert (
            label.shape == ()
        ), "Label should be a scalar tensor."  # torch.long, so shape is ()
        assert label.dtype == torch.long, "Label dtype mismatch."

    def test_collate_fn(self):
        """Test the collate_fn function with human-readable inputs."""
        # Define specific, human-readable features and labels for a batch
        # Assuming SRC_seq_len = 5 for this example. If it changes, these might need adjustment
        # or make SRC_seq_len smaller for this specific test case.
        # Let's use a fixed seq_len for this test's data for simplicity.
        test_seq_len_for_collate = 3
        if (
            SRC_seq_len < test_seq_len_for_collate
        ):  # adapt if global SRC_seq_len is smaller
            test_seq_len_for_collate = SRC_seq_len

        # Features are now token indices (long type)
        features1 = torch.tensor([1, 0, 2][:test_seq_len_for_collate], dtype=torch.long)
        label1 = torch.tensor(0, dtype=torch.long)  # Target class 0

        features2 = torch.tensor([2, 1, 0][:test_seq_len_for_collate], dtype=torch.long)
        label2 = torch.tensor(1, dtype=torch.long)  # Target class 1

        batch_data = [(features1, label1), (features2, label2)]

        # Call the collate function
        collated_features, collated_labels = collate_fn(batch_data)

        # Define the expected collated tensors
        expected_collated_features = torch.stack([features1, features2])
        expected_collated_labels = torch.stack([label1, label2])

        assert isinstance(
            collated_features, torch.Tensor
        ), "Collated features should be a Tensor."
        assert torch.equal(
            collated_features, expected_collated_features
        ), f"Collated features mismatch. Expected:\n{expected_collated_features}\nGot:\n{collated_features}"

        assert isinstance(
            collated_labels, torch.Tensor
        ), "Collated labels should be a Tensor."
        assert torch.equal(
            collated_labels, expected_collated_labels
        ), f"Collated labels mismatch. Expected:\n{expected_collated_labels}\nGot:\n{collated_labels}"

    def test_custom_model_init(self):
        """Test CustomModel initialization."""
        test_hidden_size = 16
        test_num_layers = 2
        test_vocab_size = 10
        test_embedding_dim = 5

        # Test with specified parameters
        model1 = CustomModel(
            hidden_size=test_hidden_size,
            num_layers=test_num_layers,
            num_classes=SRC_NUM_CLASSES,
            dropout=0.1,
            vocab_size=test_vocab_size,
            embedding_dim=test_embedding_dim,
        )
        assert isinstance(
            model1.embedding, nn.Embedding
        ), "Model missing Embedding layer."
        assert (
            model1.embedding.num_embeddings == test_vocab_size
        ), "Embedding vocab size mismatch."
        assert (
            model1.embedding.embedding_dim == test_embedding_dim
        ), "Embedding dimension mismatch."

        assert isinstance(model1.lstm, nn.LSTM), "Model missing LSTM layer."
        assert (
            model1.lstm.input_size == test_embedding_dim
        ), "LSTM input_size mismatch (should be embedding_dim)."
        assert model1.lstm.hidden_size == test_hidden_size, "LSTM hidden_size mismatch."
        assert model1.lstm.num_layers == test_num_layers, "LSTM num_layers mismatch."

        assert isinstance(model1.fc, nn.Linear), "Model missing Linear (fc) layer."
        assert (
            model1.fc.in_features == test_hidden_size
        ), "FC layer in_features mismatch."
        assert (
            model1.fc.out_features == SRC_NUM_CLASSES
        ), "FC layer out_features mismatch."

        # Test with default parameters (should use SRC_ constants)
        model2 = CustomModel(
            # hidden_size, num_layers, dropout, vocab_size, embedding_dim will use defaults from src.train
            num_classes=SRC_NUM_CLASSES,
        )
        assert (
            model2.lstm.hidden_size == SRC_HIDDEN_SIZE
        ), "Default LSTM hidden_size mismatch."
        assert (
            model2.lstm.num_layers == SRC_NUM_LAYERS
        ), "Default LSTM num_layers mismatch."
        assert (
            model2.embedding.num_embeddings == SRC_VOCAB_SIZE
        ), "Default Embedding vocab size mismatch."
        assert (
            model2.embedding.embedding_dim == SRC_EMBEDDING_DIM
        ), "Default Embedding dimension mismatch."
        assert (
            model2.fc.in_features == SRC_HIDDEN_SIZE
        ), "Default FC layer in_features mismatch."

    def test_custom_model_forward(self):
        """Test CustomModel forward pass."""
        model = CustomModel(
            num_classes=SRC_NUM_CLASSES,
        ).to(DEVICE)
        model.eval()  # Ensure deterministic behavior (dropout off)

        # Input is now token indices (long)
        dummy_input = torch.zeros(TEST_BATCH_SIZE, SRC_seq_len, dtype=torch.long).to(
            DEVICE
        )
        # Example: fill with valid token indices if SRC_VOCAB_SIZE > 0
        if SRC_VOCAB_SIZE > 0:
            dummy_input = dummy_input % SRC_VOCAB_SIZE

        output = model(dummy_input)

        assert isinstance(output, torch.Tensor), "Model output should be a Tensor."
        assert output.shape == (
            TEST_BATCH_SIZE,
            SRC_NUM_CLASSES,
        ), f"Model output shape mismatch. Expected ({TEST_BATCH_SIZE}, {SRC_NUM_CLASSES}), got {output.shape}"

    def test_custom_loss_function(self):
        """Test the custom_loss_function with verifiable scenarios."""
        # Scenario 1: High confidence, correct predictions -> low loss
        # Using a smaller num_classes for easier-to-define logits
        num_classes_test = 3
        logits1 = torch.tensor(
            [
                [10.0, 0.1, 0.1],  # Batch 0 strongly predicts class 0
                [0.1, 10.0, 0.1],
            ],  # Batch 1 strongly predicts class 1
            dtype=torch.float32,
        ).to(DEVICE)
        targets1 = torch.tensor([0, 1], dtype=torch.long).to(DEVICE)  # Correct targets

        loss1 = custom_loss_function(logits1, targets1)
        assert isinstance(loss1, torch.Tensor), "Loss1 should be a Tensor."
        assert loss1.shape == (), "Loss1 should be a scalar tensor."
        assert loss1.item() >= 0, "Loss1 should be non-negative."
        assert (
            loss1.item() < 0.1
        ), f"Expected low loss for confident correct predictions, got {loss1.item()}"

        # Scenario 2: Low confidence / incorrect predictions -> higher loss
        logits2 = torch.tensor(
            [
                [0.1, 0.1, 10.0],  # Batch 0 strongly predicts class 2
                [10.0, 0.1, 0.1],
            ],  # Batch 1 strongly predicts class 0
            dtype=torch.float32,
        ).to(DEVICE)
        # Targets are still 0 and 1, so predictions are confidently wrong.
        targets2 = torch.tensor([0, 1], dtype=torch.long).to(DEVICE)

        loss2 = custom_loss_function(logits2, targets2)
        assert isinstance(loss2, torch.Tensor), "Loss2 should be a Tensor."
        assert loss2.item() >= 0, "Loss2 should be non-negative."
        assert (
            loss2.item() > loss1.item()
        ), f"Expected loss2 ({loss2.item()}) to be higher than loss1 ({loss1.item()}) for incorrect predictions."
        assert (
            loss2.item() > 1.0
        ), f"Expected reasonably high loss for confident wrong predictions, got {loss2.item()}"

    def _get_dummy_dataloader(self):
        """Helper to create a dummy dataloader for train/eval tests with readable data."""
        input_cols = [f"feature_{i}" for i in range(SRC_seq_len)]
        target_col = "target"
        # Create easily verifiable features: feature_j for sample i will be (i+j) % VOCAB_SIZE
        data = {
            f"feature_{j}": [(i + j) % SRC_VOCAB_SIZE for i in range(TEST_NUM_SAMPLES)]
            for j in range(SRC_seq_len)
        }
        # Target for sample i is sum of its original features (before modulo if that was for generation)
        # For simplicity, let's make targets simple class indices for test.
        # For example, target is i % NUM_CLASSES
        data[target_col] = [i % SRC_NUM_CLASSES for i in range(TEST_NUM_SAMPLES)]

        df = pd.DataFrame(data)
        # ModelDataset expects input_cols for features that are integers.
        dataset = ModelDataset(df_=df, input_cols=input_cols, target_col=target_col)
        dataloader = DataLoader(
            dataset, batch_size=TEST_BATCH_SIZE, collate_fn=collate_fn
        )
        return dataloader

    def test_train_epoch(self):
        """Test the train_epoch function (basic run)."""
        model = CustomModel(
            num_classes=SRC_NUM_CLASSES,
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=TEST_LR)
        dataloader = self._get_dummy_dataloader()

        # Store initial parameters to check if they change
        initial_params = [param.clone() for param in model.parameters()]

        avg_loss, accuracy = train_epoch(model, dataloader, optimizer)

        assert isinstance(avg_loss, float), "Average loss should be a float."
        assert isinstance(accuracy, float), "Accuracy should be a float."
        assert 0.0 <= accuracy <= 1.0, "Accuracy should be between 0 and 1."
        assert model.training, "Model should be in training mode after train_epoch."

        # Check if model parameters changed
        params_changed = False
        for initial_p, trained_p in zip(initial_params, model.parameters()):
            if not torch.equal(initial_p, trained_p):
                params_changed = True
                break
        assert params_changed, "Model parameters should change after a training epoch."

    def test_eval_epoch(self):
        """Test the eval_epoch function (basic run)."""
        model = CustomModel(
            num_classes=SRC_NUM_CLASSES,
        ).to(DEVICE)
        dataloader = self._get_dummy_dataloader()

        # Store initial parameters to check they don't change (no optimizer step)
        initial_params_state_dict = copy.deepcopy(model.state_dict())

        avg_loss, accuracy = eval_epoch(model, dataloader)

        assert isinstance(avg_loss, float), "Average loss should be a float."
        assert isinstance(accuracy, float), "Accuracy should be a float."
        assert 0.0 <= accuracy <= 1.0, "Accuracy should be between 0 and 1."
        assert (
            not model.training
        ), "Model should be in evaluation mode after eval_epoch."

        # Check model parameters did not change
        for name, param in model.named_parameters():
            assert torch.equal(
                initial_params_state_dict[name], param
            ), f"Parameter {name} changed during eval_epoch."

    def test_infer(self):
        """Test the infer function."""
        model = CustomModel(
            num_classes=SRC_NUM_CLASSES,
        ).to(DEVICE)
        model.eval()

        # Test with a simple, human-readable valid input (list of ints for token indices)
        valid_feature_vector = [
            i % SRC_VOCAB_SIZE for i in range(SRC_seq_len)
        ]  # e.g., [0, 1, 2, 3, 4] if SRC_VOCAB_SIZE >= 5
        prediction = infer(model, valid_feature_vector, SRC_seq_len)
        assert isinstance(prediction, int), "Prediction should be an integer."
        assert 0 <= prediction < SRC_NUM_CLASSES, "Prediction out of class range."

        # Test with incorrect input dimension using a clear example
        invalid_feature_vector = [i % SRC_VOCAB_SIZE for i in range(SRC_seq_len + 1)]
        try:
            infer(model, invalid_feature_vector, SRC_seq_len)
            assert False, "ValueError not raised for incorrect input dimension."
        except ValueError as e:
            expected_msg_part = f"Input feature vector length {len(invalid_feature_vector)} does not match model's expected seq_len {SRC_seq_len}"
            assert expected_msg_part in str(
                e
            ), f"ValueError message mismatch. Got: {str(e)}"
        except Exception as e:
            assert False, f"Unexpected exception {type(e).__name__} raised: {e}"


if __name__ == "__main__":
    t = TestModel()

    t.test_model_dataset()
    t.test_collate_fn()
    t.test_custom_model_init()
    t.test_custom_model_forward()
    t.test_custom_loss_function()
    t.test_train_epoch()
    t.test_eval_epoch()
    t.test_infer()

    print("\nAll tests passed!")
