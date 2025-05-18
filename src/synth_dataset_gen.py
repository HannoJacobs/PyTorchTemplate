"""generate a synthetic dataset"""

# pylint: disable=E0401,E0611,E1101,C0413,R0913,R0914,C2801,W1203,C3001,R0917
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from tools.display_df import display_df  # pylint: disable=W0611

SEQ_LEN = 5
INPUT_TOKEN_MIN = 0
INPUT_TOKEN_MAX = 9

NUM_SAMPLES_TO_GENERATE = 1000
DATASET_FOLDER = "Datasets/"


def generate_sample():
    """Generate a rule-based sequence for a synthetic dataset"""
    inputs = [
        np.random.randint(INPUT_TOKEN_MIN, INPUT_TOKEN_MAX + 1) for _ in range(SEQ_LEN)
    ]
    target = sum(inputs)
    return inputs + [target]


def generate_dataset():
    """Generate and save a synthetic dataset using global config."""
    columns = [f"input_{i}" for i in range(1, SEQ_LEN + 1)] + ["target"]
    df_ = pd.DataFrame(columns=columns)

    for _ in range(NUM_SAMPLES_TO_GENERATE):
        row = generate_sample()
        df_.loc[len(df_)] = row

    os.makedirs(DATASET_FOLDER, exist_ok=True)
    output_filename = os.path.join(
        DATASET_FOLDER,
        f"synth_"
        f"i{SEQ_LEN}_"
        f"r{INPUT_TOKEN_MIN}-{INPUT_TOKEN_MAX}_"
        f"n-{NUM_SAMPLES_TO_GENERATE}.csv",
    )
    df_.to_csv(output_filename, index=False)
    print(f"Generated {NUM_SAMPLES_TO_GENERATE} samples to {output_filename}")
    return df_


if __name__ == "__main__":
    print(f"Creating {NUM_SAMPLES_TO_GENERATE:,} samples...")
    start_time = time.time()
    df = generate_dataset()
    print(f"Time taken: {(time.time() - start_time):.2f} seconds")

    # display_df(df, "df_display")
