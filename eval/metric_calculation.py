import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from utils.utils import (
    load_base_input_data,
    load_base_output_data,
    create_variant_input,
    perform_prediction,
    calculate_metric_across_variants,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT_ROOT / "data" / "base"
SAMPLES_PATH = PROJECT_ROOT / "data" / "samples"
LATENT_PATH = PROJECT_ROOT / "results" / "selected_latent"
MODELS_ROOT = PROJECT_ROOT / "results" / "models"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eval_results_nrmse"

TRAIN_SIZE = int(input("Enter train size: "))
TAG = input("Enter tag: ")
SEED = int(input("Enter seed [50000]: "))

VARIANTS = ["PLE", "SAE", "UAE", "SA-PGA"]

def summarize_nrmse(m_d_total: dict, m_a_total: dict) -> dict:
    drift_per_floor = np.asarray(m_d_total["NRMSE"]).reshape(-1)
    accel_per_floor = np.asarray(m_a_total["NRMSE"]).reshape(-1)

    avg_drift = float(np.nanmean(drift_per_floor))
    avg_accel = float(np.nanmean(accel_per_floor))
    avg_total = float((avg_drift + avg_accel) / 2.0)

    out = {}
    for i, val in enumerate(drift_per_floor, start=1):
        out[f"drift_floor_{i}"] = float(val)
    for i, val in enumerate(accel_per_floor, start=1):
        out[f"accel_floor_{i}"] = float(val)

    out["avg_drift_nrmse"] = avg_drift
    out["avg_accel_nrmse"] = avg_accel
    out["avg_total_nrmse"] = avg_total
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (
        gm_test_samples,
        physical_samples_test,
        spectral_features_test_samples,
        rs_test_samples,
        pga_test,
        sa_t1_test,
    ) = load_base_input_data(base_path=str(BASE_PATH) + os.sep)

    y_true_EBF_drift, y_true_EBF_accel, _ = load_base_output_data(
        train_size=TRAIN_SIZE,
        tag=TAG,
        base_path=str(BASE_PATH) + os.sep,
        samples_path=str(SAMPLES_PATH) + os.sep,
    )

    latent_file = LATENT_PATH / f"selected_latent_{TRAIN_SIZE}_{TAG}.xlsx"
    latent = pd.read_excel(latent_file).iloc[0].to_dict()

    results_summary = {}

    for variant in VARIANTS:
        models_path = MODELS_ROOT / variant
        ltn = int(latent[variant])

        print(f"Start predicting {variant}")

        physical_samples_test_, n_struct_features = create_variant_input(
            variant=variant,
            physical_samples_test=physical_samples_test,
            spectral_features_test_samples=spectral_features_test_samples,
        )

        y_pred_EBF_drift, y_pred_EBF_accel, _, _ = perform_prediction(
            models_path=str(models_path) + os.sep,
            model_input=physical_samples_test_,
            gm_test_samples=gm_test_samples,
            RS_test_samples=rs_test_samples,
            variant=variant,
            train_size=TRAIN_SIZE,
            latent=ltn,
            tag=TAG,
        )

        (
            m_d_total,
            m_a_total,

        ) = calculate_metric_across_variants(
            y_true_EBF_drift,
            y_true_EBF_accel,
            y_pred_EBF_drift,
            y_pred_EBF_accel,
            variant=variant,
        )

        summary = summarize_nrmse(m_d_total, m_a_total)
        results_summary[variant] = summary

        print("=============NRMSE================")
        print(f"{variant} - Average acceleration is: {summary['avg_accel_nrmse']}")
        print(f"{variant} - Average drift is: {summary['avg_drift_nrmse']}")
        print(f"{variant} - Average total is: {summary['avg_total_nrmse']}")

    summary_df = pd.DataFrame(results_summary).T
    summary_df.to_excel(OUTPUT_DIR / f"nrmse_summary_{TRAIN_SIZE}_{TAG}.xlsx", index=True)

    drift_cols = [c for c in summary_df.columns if c.startswith("drift_floor_")]
    accel_cols = [c for c in summary_df.columns if c.startswith("accel_floor_")]

    summary_df[drift_cols].to_excel(OUTPUT_DIR / f"nrmse_drift_per_floor_{TRAIN_SIZE}_{TAG}.xlsx", index=True)
    summary_df[accel_cols].to_excel(OUTPUT_DIR / f"nrmse_accel_per_floor_{TRAIN_SIZE}_{TAG}.xlsx", index=True)

    print(f"Saved NRMSE files to: {OUTPUT_DIR}")