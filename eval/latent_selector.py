from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

"""
How to run: 
1) opend the command window
2) paste the following command:
``
python latent_selector.py \
--loss-path 'USER LOSS PATH' \
--output-dir 'USER OUTPUT LATENT AND LOSS DIRECTORY' \
--train-size 'TRAIN SIZE: HERE 2720 IS RECOMMENDED' \
--tag 'THE TAG SAVED FOR THE TRAIN SIZE'
``
IMP: Adapt it based on your setup and directory
"""

ALLOWED_VARIANTS = ["PLE", "SAE", "UAE", "SA-PGA"]


def parse_variants(raw: str | None):
    if raw is None or raw.strip() == "":
        return ALLOWED_VARIANTS
    items = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in items if x not in ALLOWED_VARIANTS]
    if bad:
        raise ValueError(f"Unsupported variants: {bad}. Allowed: {ALLOWED_VARIANTS}")
    return items


def extract_nmse(xls: pd.ExcelFile, sheet_name: str) -> float:
    df = xls.parse(sheet_name)
    for key in ["loss", "output_nmse_scalar", "nmse_scalar"]:
        if key in df.columns:
            return float(df[key].iloc[0])
    raise KeyError(f"No known loss column found in sheet '{sheet_name}'.")


def select_latents(loss_path: Path, variants, train_size: int, tag: str):
    selected_latent = {}
    selected_loss = {}

    for var in variants:
        # this jas been saved during the training
        xls = pd.ExcelFile(loss_path / f"{var}_{train_size}_{tag}.xlsx") 
        latent_range = xls.sheet_names
        # extract the loss from the required file
        loss = [extract_nmse(xls, ltn) for ltn in latent_range]
        # get the latent correspond to least loss value
        best_idx = int(np.argmin(loss)) 
        selected_loss[var] = [loss[best_idx]] # Store the selected loss
        selected_latent[var] = [latent_range[best_idx]] # Stor the corresponding latent range

    return selected_latent, selected_loss


def main():
    parser = argparse.ArgumentParser(
        description="LAtent selector"
    )
    parser.add_argument("--loss-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--variants", type=str, default="PLE,SAE,UAE,SA-PGA")
    args = parser.parse_args()

    loss_path = Path(args.loss_path)
    output_dir = Path(args.output_dir) # create if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_variants(args.variants)

    selected_latent, selected_loss = select_latents(
        loss_path=loss_path,
        variants=variants,
        train_size=args.train_size,
        tag=args.tag,
    )
    # save to excel file in the output directory
    pd.DataFrame(selected_loss).to_excel(
        output_dir / f"selected_loss_{args.train_size}_{args.tag}.xlsx",
        index=False,
    )
    # save to excel file in the output directory
    pd.DataFrame(selected_latent).to_excel(
        output_dir / f"selected_latent_{args.train_size}_{args.tag}.xlsx",
        index=False,
    )

    print(f"Saved selected latent/loss files to: {output_dir}")


if __name__ == "__main__":
    main()
