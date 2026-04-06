
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from utils.utils import (set_seed, run_latent_sweep)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Model running module."
    )
    parser.add_argument("--base-path", type=str, required=True, help="Path to test and val dataset")
    parser.add_argument("--samples-path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to write the results")
    parser.add_argument("--variant", type=str, required=True, help="Default: SA-PGA")
    parser.add_argument("--train-size", type=int, required=True, help="Training size: 2720")
    parser.add_argument("--tag", type=str, required=True, help="Dataset tag: can change based on the seed")
    parser.add_argument("--epochs", type=int, default=500, help="Epochs for the initial latent sweep")
    parser.add_argument("--refine-epochs", type=int, default=250, help="Epochs for the refined latent sweep")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=50000, help="Random seed")
    parser.add_argument("--input-timesteps", type=int, default=4000)
    parser.add_argument("--ts-features", type=int, default=1, help='Do not change this value!')
    parser.add_argument("--n-floors", type=int, default=10, help='number of floors x number of EDP type')
    parser.add_argument("--save-models", action="store_true", help="Save the trained models")
    parser.add_argument("--refine-latent", action="store_true", help="Run the second-stage latent selection")
    return parser.parse_args()


def main():
    args = parse_args()
    run_latent_sweep(args)


if __name__ == "__main__":
    main()
