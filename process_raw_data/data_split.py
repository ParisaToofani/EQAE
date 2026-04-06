from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils.utils import (
    build_fixed_splits,
    dataset_creator,
    load_raw_data,
    make_one_to_many_pairs,
    make_val_and_test_pairs,
    resample_scaled_quakes,
    spectral_feature_parallel_actual,
    spectral_feature_parallel_nominal,
    _space_filling_select,
)


def create_train_dataset(
    data_root: str | Path,
    output_dir: str | Path,
    train_size: int = 2720,
    tag: str = None,
    seed_eq: int = 0, # modify based on your requirements
    seed_struct: int = 0, # modify based on your requirements
    ns_per_gm: int = 1, # modify based on your requirements
    test_sizes: tuple[int, int] = (100, 100),
    val_fracs: tuple[float, float] = (0.2, 0.2),
):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(data_root)
    # Generate spectral features based on nominal period
    spectral_features = Parallel(n_jobs=-1)(
        delayed(spectral_feature_parallel_nominal)(raw["acc_resp_spectrum"][idx, :])
        for idx in range(raw["acc_resp_spectrum"].shape[0])
    )
    spectral_features = np.array(spectral_features)
    # create initial train/test/val splits
    splits = build_fixed_splits(
        spec=spectral_features,
        struct=raw["physical_samples"],
        test_sizes=test_sizes,
        val_fracs=val_fracs,
        seed_test=42,
        seed_val=5000,
    )
    # get the indexes from the training pool
    e_tr = splits["E_tr_pool"][_space_filling_select(spectral_features[splits["E_tr_pool"]], train_size // max(ns_per_gm, 1), seed=seed_eq)]
    # map structures to ground motions
    train_pairs = make_one_to_many_pairs(e_tr, splits["S_tr_pool"], ns_per_gm=ns_per_gm, seed=seed_struct)
    val_pairs, test_pairs = make_val_and_test_pairs(splits["E_val"], 
                                                    splits["S_val"], 
                                                    splits["E_test"], 
                                                    splits["S_test"])
    # save the train set
    pd.DataFrame(train_pairs).to_hdf(output_dir / f"train_pairs_{train_size}_{tag}.h5", key="train")
    pd.DataFrame(test_pairs).to_hdf(output_dir / 'test_pairs.h5', key='test')
    pd.DataFrame(val_pairs).to_hdf(output_dir / 'val_pairs.h5', key='val')
    # process the ground motions
    scaled_eq_db = resample_scaled_quakes(raw["scaled_quakes"])
    # start creating the dataset for training
    physical_samples_train = dataset_creator(train_pairs, raw["physical_samples"], target="struct")
    physical_samples_test = dataset_creator(test_pairs, raw["physical_samples"], target ='struct')
    physical_samples_val = dataset_creator(val_pairs, raw["physical_samples"], target ='struct')
    #--------------------------------------------------
    peak_drift_per_floor_train = dataset_creator(train_pairs, raw["peak_drift_per_floor"], target="EDP")
    peak_drift_per_floor_test = dataset_creator(test_pairs, raw["peak_drift_per_floor"], target ='EDP')
    peak_drift_per_floor_val = dataset_creator(val_pairs, raw["peak_drift_per_floor"], target ='EDP')
    #--------------------------------------------------
    peak_accel_per_floor_train = dataset_creator(train_pairs, raw["peak_accel_per_floor"], target="EDP")
    peak_accel_per_floor_test = dataset_creator(test_pairs, raw["peak_accel_per_floor"], target ='EDP')
    peak_accel_per_floor_val = dataset_creator(val_pairs, raw["peak_accel_per_floor"], target ='EDP')
    #--------------------------------------------------
    eq_train = dataset_creator(train_pairs, scaled_eq_db, target="EQ")
    eq_test = dataset_creator(test_pairs, scaled_eq_db, target ='EQ')
    eq_val = dataset_creator(val_pairs, scaled_eq_db, target ='EQ')
    #--------------------------------------------------
    spectral_features_train = Parallel(n_jobs=-1)(
        delayed(spectral_feature_parallel_actual)(raw["acc_resp_spectrum"][idx[0], :], raw["physical_samples"][idx[1], 0])
        for idx in train_pairs
    )
    spectral_features_train = np.array(spectral_features_train)
    rs_train = dataset_creator(train_pairs, raw["acc_resp_spectrum"], target="Spec")

    spectral_features_test = Parallel(n_jobs=-1)(delayed(spectral_feature_parallel_actual)(raw["acc_resp_spectrum"][idx[0], :], raw["physical_samples"][idx[1], 0]) \
                                            for idx in test_pairs)
    spectral_features_test = np.array(spectral_features_test)
    rs_test = dataset_creator(test_pairs, raw["acc_resp_spectrum"], target ='Spec')

    spectral_features_val = Parallel(n_jobs=-1)(delayed(spectral_feature_parallel_actual)(raw["acc_resp_spectrum"][idx[0], :], raw["physical_samples"][idx[1], 0]) \
                                            for idx in val_pairs)
    spectral_features_val = np.array(spectral_features_val)
    rs_val = dataset_creator(val_pairs, raw["acc_resp_spectrum"], target ='Spec')
    # Start saving training data
    np.savez(output_dir / f"physical_samples_train_{train_size}_{tag}.npz", data=physical_samples_train)
    np.savez(output_dir / f"peak_drift_per_floor_train_{train_size}_{tag}.npz", data=peak_drift_per_floor_train)
    np.savez(output_dir / f"peak_accel_per_floor_train_{train_size}_{tag}.npz", data=peak_accel_per_floor_train)
    np.savez(output_dir / f"EQ_train_{train_size}_{tag}.npz", data=eq_train)
    np.savez(output_dir / f"spectral_features_train_{train_size}_{tag}.npz", data=spectral_features_train)
    np.savez(output_dir / f"RS_train_{train_size}_{tag}.npz", data=rs_train)
    # Start saving testing data
    np.savez(output_dir / f"physical_samples_test.npz", data=physical_samples_test)
    np.savez(output_dir / f"peak_drift_per_floor_test.npz", data=peak_drift_per_floor_test)
    np.savez(output_dir / f"peak_accel_per_floor_test.npz", data=peak_accel_per_floor_test)
    np.savez(output_dir / f"EQ_test.npz", data=eq_test)
    np.savez(output_dir / f"spectral_features_test.npz", data=spectral_features_test)
    np.savez(output_dir / f"RS_test.npz", data=rs_test)
    # Start saving validation data
    np.savez(output_dir / f"physical_samples_val.npz", data=physical_samples_val)
    np.savez(output_dir / f"peak_drift_per_floor_val.npz", data=peak_drift_per_floor_val)
    np.savez(output_dir / f"peak_accel_per_floor_val.npz", data=peak_accel_per_floor_val)
    np.savez(output_dir / f"EQ_val.npz", data=eq_val)
    np.savez(output_dir / f"spectral_features_val.npz", data=spectral_features_val)
    np.savez(output_dir / f"RS_val.npz", data=rs_val)
    return train_pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create training/testing and validation split files from raw data.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-size", type=int, default=2720)
    parser.add_argument("--tag")
    parser.add_argument("--seed-eq", type=int)
    parser.add_argument("--seed-struct", type=int)
    parser.add_argument("--ns-per-gm", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_pairs = create_train_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_size=args.train_size,
        tag=args.tag,
        seed_eq=args.seed_eq,
        seed_struct=args.seed_struct,
        ns_per_gm=args.ns_per_gm,
    )
    print(f"Saved {len(train_pairs)} train pairs")


if __name__ == "__main__":
    main()
