from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from tensorflow.keras import Model, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input, Reshape

from models.train import ( # import the model architecture for training
    train_model,
    train_model_PLE,
    train_model_SAE,
    train_model_UAE,
)

# required later in metrics calculation
NormType = Literal["std", "range", "iqr", "mean", "rms", "none"] 
StdType = Literal["y_true", "y_pred", "pooled"]

#======================================
# General setups 
#======================================

def set_seed(seed: int = 42) -> None:
    """
    Fix the seed for reproducability
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


#======================================
# metrics calculation
# IMP: each of these functions have their own 
# utilities either in plot creation or the 
# type of the model or output
#======================================

def nmse_dataset_vectorized(x_true: tf.Tensor, x_pred: tf.Tensor, eps: float = 1e-10) -> tf.Tensor:
    """
    Normalized loss, used during the training 
    !!! This is useful if the EDPs are of a different type
    """
    x_true = tf.cast(x_true, dtype=tf.float32)
    x_pred = tf.cast(x_pred, dtype=tf.float32)
    mse = tf.reduce_mean(tf.square(x_true - x_pred))
    variance_true = tf.reduce_mean(tf.square(x_true - tf.reduce_mean(x_true)))
    return mse / (variance_true + eps)


def make_nmse_scalar(train_var_scalar: float):
    """
    Normalized loss, used during the training 
    !!! This is useful if the EDPs are of a different type
    """
    train_var_scalar = tf.constant(train_var_scalar, dtype=tf.float32)
    eps = tf.constant(1e-12, dtype=tf.float32)

    def nmse_scalar(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        return mse / tf.maximum(train_var_scalar, eps)

    nmse_scalar.__name__ = "nmse_scalar"
    return nmse_scalar


def make_nmse_per_output(train_var_per_output: np.ndarray):
    """
    This is use for having an independent track 
    of loss for either drift or acceleration
    """
    train_var_per_output = tf.constant(train_var_per_output, dtype=tf.float32)
    eps = tf.constant(1e-12, dtype=tf.float32)

    def nmse_per_output(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        se = tf.square(y_pred - y_true)
        mse_per_out = tf.reduce_mean(se, axis=0)
        nmse_vec = mse_per_out / tf.maximum(train_var_per_output, eps)
        return tf.reduce_mean(nmse_vec)

    nmse_per_output.__name__ = "nmse_per_output"
    return nmse_per_output

#======================================
# Dataset creation
# IMP: each of these functions have their own 
# utilities either in plot creation or the 
# type of the model or output
#======================================

def struct_indices_selection(indices: np.ndarray, number_of_samples: int = 2720, replace: bool = True, seed: Optional[int] = None) -> np.ndarray:
    """
    Randomly select the structural indices from the structural database
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    return rng.choice(indices, size=number_of_samples, replace=replace)


def eq_indices_selection(
    indices: Optional[np.ndarray] = None,
    number_of_samples: int = 2720,
    spectral_features: Optional[np.ndarray] = None,
    sampling_type: str = "kmean",
    replace: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Different EQ index selections:
    1) random: completely at random
    2) Kmean: group the ground motion based on the similar characteristics
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    if sampling_type == "random":
        if indices is None:
            raise ValueError("indices are required for random selection")
        # here the default is false to enhance diversity across the ground motions
        return rng.choice(indices, size=number_of_samples, replace=replace) 
    if sampling_type != "kmean":
        raise ValueError("sampling_type must be 'random' or 'kmean'")
    if spectral_features is None:
        raise ValueError("spectral_features are required for kmeans selection")
    k = min(int(number_of_samples), spectral_features.shape[0])
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    km.fit(spectral_features)
    selected = []
    for i in range(k):
        members = np.where(km.labels_ == i)[0]
        if len(members) == 0:
            continue
        selected.append(rng.choice(members, 1, replace=replace)[0])
    return np.asarray(selected, dtype=int)


def create_pairs(eq_indices: Sequence[int], struct_indices: Sequence[int], se_ratio: int = 1) -> np.ndarray:
    """
    Match structures and groundmotions together
    """
    pairs: List[Tuple[int, int]] = []
    for eq_idx in eq_indices:
        sampled_struct = np.random.choice(struct_indices, size=se_ratio, replace=True)
        for s_idx in sampled_struct:
            pairs.append((int(eq_idx), int(s_idx)))
    return np.asarray(pairs, dtype=int)


def _cartesian_pairs(a: Sequence[int], b: Sequence[int]) -> np.ndarray:
    """
    Match structures and ground motions together (EBF style)
    """
    return np.asarray([(int(i), int(j)) for i in a for j in b], dtype=int)


def build_fixed_splits(
    spec: np.ndarray, # spectrum data
    struct: np.ndarray, # structural indices
    test_sizes: Tuple[int, int], # required test size (eq, struct)
    val_fracs: Tuple[float, float], # required val size (eq, struct)
    seed_test: int = 42, # your choice of the seed for the test set
    seed_val: int = 5000, # your choice of the seed for the validation set
) -> Dict[str, np.ndarray]:
    n_eq = spec.shape[0]
    n_struct = struct.shape[0]
    all_eq = np.arange(n_eq)
    all_struct = np.arange(n_struct)

    rng_test = np.random.default_rng(seed_test)
    e_test = np.sort(rng_test.choice(all_eq, size=min(test_sizes[0], n_eq), replace=False))
    s_test = np.sort(rng_test.choice(all_struct, size=min(test_sizes[1], n_struct), replace=False))

    e_pool = np.setdiff1d(all_eq, e_test)
    s_pool = np.setdiff1d(all_struct, s_test)

    rng_val = np.random.default_rng(seed_val)
    e_val = np.sort(rng_val.choice(e_pool, size=max(1, int(len(e_pool) * val_fracs[0])), replace=False))
    s_val = np.sort(rng_val.choice(s_pool, size=max(1, int(len(s_pool) * val_fracs[1])), replace=False))

    e_tr_pool = np.setdiff1d(e_pool, e_val)
    s_tr_pool = np.setdiff1d(s_pool, s_val)

    return {
        "E_test": e_test,
        "S_test": s_test,
        "E_val": e_val,
        "S_val": s_val,
        "E_tr_pool": e_tr_pool,
        "S_tr_pool": s_tr_pool,
    }


def make_val_and_test_pairs(e_val: Sequence[int], s_val: Sequence[int], e_test: Sequence[int], s_test: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    return _cartesian_pairs(e_val, s_val), _cartesian_pairs(e_test, s_test)


def make_one_to_many_pairs(e_tr: Sequence[int], s_tr_pool: Sequence[int], ns_per_gm: int = 2, seed: int = 19) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pairs = []
    for eq_idx in e_tr:
        selected = rng.choice(s_tr_pool, size=ns_per_gm, replace=False if ns_per_gm <= len(s_tr_pool) else True)
        pairs.extend((int(eq_idx), int(s_idx)) for s_idx in selected)
    return np.asarray(pairs, dtype=int)


def signal_interpolator(time_axis: np.ndarray, signal: np.ndarray, target_time: np.ndarray | float) -> np.ndarray:
    # interpolate the groundmotion to have consistent time steps
    f = interp1d(time_axis, signal, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(target_time)


def dataset_creator(pair_array: np.ndarray, input_samples: np.ndarray, target: str) -> np.ndarray:
    """
    This function has been designed to create the databases in the size
    and required format
    """
    target_samples = []
    for eq_idx, struct_idx in pair_array:
        if target == "struct":
            target_samples.append(input_samples[struct_idx, :])
        elif target == "EDP":
            target_samples.append(input_samples[eq_idx, struct_idx, :])
        elif target == "EQ":
            target_samples.append(input_samples[eq_idx, :, :])
        elif target == "Spec":
            target_samples.append(input_samples[eq_idx, :])
        else:
            raise ValueError(f"Unknown target: {target}")
    return np.asarray(target_samples)


def spectral_feature_parallel_nominal(spect: np.ndarray) -> List[float]:
    """
    Calculate the spectral features at 
    the nominal period: For this specific structure is T1= 0.77 s, change 
    based on your structure
    """
    t1 = 0.77
    times = np.arange(0, 5, 0.02) # spectral duration is set to be 5s with 0.02 time steps
    pga = spect[0] / 9.81
    s_a_0_2 = signal_interpolator(times, spect, 0.2) / 9.81
    s_a_t_1 = signal_interpolator(times, spect, t1) / 9.81
    s_a_1 = signal_interpolator(times, spect, 1) / 9.81
    s_a_2_5 = signal_interpolator(times, spect, 2.5) / 9.81
    return [pga, s_a_0_2, s_a_t_1, s_a_1, s_a_2_5]


def spectral_feature_parallel_actual(spect: np.ndarray, physical_sample_t1: float) -> List[float]:
    """
    Calculate the spectral features at 
    the actual period: This is defined in the database
    """
    times = np.arange(0, 5, 0.02)
    pga = spect[0] / 9.81
    s_a_0_2 = signal_interpolator(times, spect, 0.2) / 9.81
    s_a_t_1 = signal_interpolator(times, spect, physical_sample_t1) / 9.81
    s_a_1 = signal_interpolator(times, spect, 1) / 9.81
    s_a_2_5 = signal_interpolator(times, spect, 2.5) / 9.81
    return [pga, s_a_0_2, s_a_t_1, s_a_1, s_a_2_5]

# Store the data automatically
@dataclass
class BaseData:
    gm: np.ndarray
    physical: np.ndarray
    spectral: np.ndarray
    rs: np.ndarray
    drift: np.ndarray
    accel: np.ndarray

# Store the data automatically
@dataclass
class PreparedData:
    gm_train: np.ndarray
    gm_test: np.ndarray
    gm_val: np.ndarray
    physical_train: np.ndarray
    physical_test: np.ndarray
    physical_val: np.ndarray
    edp_train: np.ndarray
    edp_test: np.ndarray
    edp_val: np.ndarray
    n_struct_features: int

#======================================
# Load the da
#======================================
def load_raw_data(data_root: str | Path) -> Dict[str, np.ndarray]:
    """
    Load the required data
    """
    data_root = Path(data_root)
    scaled_quakes = scipy.io.loadmat(data_root / "scaled_quakes.mat")["scaled_quake_in_g"]
    steel_file = h5py.File(data_root / "SteelSamples_5.mat")
    physical_samples = pd.DataFrame(np.array(steel_file["physical_samples"]).T).to_numpy()
    peak_drift_per_floor = np.array(steel_file["y_driftTotal"])
    peak_accel_per_floor = np.array(steel_file["y_accelTotal"])
    acc_resp_spectrum = pd.read_hdf(data_root / "spectrum_data.h5").to_numpy()
    return {
        "scaled_quakes": scaled_quakes,
        "physical_samples": physical_samples,
        "peak_drift_per_floor": peak_drift_per_floor,
        "peak_accel_per_floor": peak_accel_per_floor,
        "acc_resp_spectrum": acc_resp_spectrum,
    }


def resample_scaled_quakes(scaled_quakes: np.ndarray, target_length: int = 4000, target_dt: float = 0.02) -> np.ndarray:
    """
    Making the ground motion length consistent
    """
    eq_general = {"EQ": [], "Time": []}
    num_gm = scaled_quakes.shape[1]
    for i in range(num_gm):
        eq_general["EQ"].append(scaled_quakes[0, i][:, 1])
        eq_general["Time"].append(scaled_quakes[0, i][:, 0])
    eq_general_df = pd.DataFrame(eq_general)

    resampled_signal = []
    for i in range(num_gm):
        time_array = eq_general_df.iloc[i].Time
        time_step = time_array[1] - time_array[0]
        signal = eq_general_df.iloc[i].EQ
        signal_length = len(signal)
        time_axis = np.linspace(0, signal_length * time_step, signal_length)
        target_time = np.linspace(0, target_length * target_dt, target_length)
        resampled_signal.append(signal_interpolator(time_axis, signal, target_time))
    return np.asarray(resampled_signal).reshape(num_gm, target_length, 1)


def load_split_npz(folder: str | Path, prefix: str, train_size: Optional[int] = None, tag: Optional[str] = None, log_transform: bool = False, divide_by_981: bool = False) -> np.ndarray:
    folder = Path(folder)
    if train_size is None:
        filename = f"{prefix}.npz"
    else:
        filename = f"{prefix}_{train_size}_{tag}.npz"
    arr = np.load(folder / filename)["data"]
    if divide_by_981:
        arr = arr / 9.81
    if log_transform:
        arr = np.log(arr)
    return arr


def load_base_input_data(base_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gm_test_samples = load_split_npz(base_path, "EQ_test")
    physical_samples_test = load_split_npz(base_path, "physical_samples_test")
    spectral_features_test_samples = load_split_npz(base_path, "spectral_features_test", log_transform=True)
    pga_test = spectral_features_test_samples[:, 0]
    sa_t1_test = spectral_features_test_samples[:, 2]
    rs_test_samples = load_split_npz(base_path, "RS_test", divide_by_981=True)
    return gm_test_samples, physical_samples_test, spectral_features_test_samples, rs_test_samples, pga_test, sa_t1_test


def prepare_training_data(base_path: str | Path, samples_path: str | Path, train_size: int, tag: str, variant: str) -> PreparedData:
    """
    Load all generated data
    train: used for training
    val: used for calibration and tuning
    test: used for high-level evaluation
    """
    gm_train = load_split_npz(samples_path, "EQ_train", train_size, tag)
    gm_test = load_split_npz(base_path, "EQ_test")
    gm_val = load_split_npz(base_path, "EQ_val")

    physical_train = load_split_npz(samples_path, "physical_samples_train", train_size, tag)
    physical_test = load_split_npz(base_path, "physical_samples_test")
    physical_val = load_split_npz(base_path, "physical_samples_val")

    spectral_train = load_split_npz(samples_path, "spectral_features_train", train_size, tag, log_transform=True)
    spectral_test = load_split_npz(base_path, "spectral_features_test", log_transform=True)
    spectral_val = load_split_npz(base_path, "spectral_features_val", log_transform=True)

    rs_train = load_split_npz(samples_path, "RS_train", train_size, tag, divide_by_981=True)
    rs_test = load_split_npz(base_path, "RS_test", divide_by_981=True)
    rs_val = load_split_npz(base_path, "RS_val", divide_by_981=True)

    drift_train = load_split_npz(samples_path, "peak_drift_per_floor_train", train_size, tag, log_transform=True)
    drift_test = load_split_npz(base_path, "peak_drift_per_floor_test", log_transform=True)
    drift_val = load_split_npz(base_path, "peak_drift_per_floor_val", log_transform=True)

    accel_train = load_split_npz(samples_path, "peak_accel_per_floor_train", train_size, tag, log_transform=True, divide_by_981=True)
    accel_test = load_split_npz(base_path, "peak_accel_per_floor_test", log_transform=True, divide_by_981=True)
    accel_val = load_split_npz(base_path, "peak_accel_per_floor_val", log_transform=True, divide_by_981=True)

    sa_t1_train = spectral_train[:, 2]
    sa_t1_test = spectral_test[:, 2]
    sa_t1_val = spectral_val[:, 2]

    if variant in {"PLE", "NH"}:
        n_struct_features = 8
    elif variant == "SA":
        physical_train = np.concatenate((physical_train, sa_t1_train.reshape(-1, 1)), axis=1)
        physical_test = np.concatenate((physical_test, sa_t1_test.reshape(-1, 1)), axis=1)
        physical_val = np.concatenate((physical_val, sa_t1_val.reshape(-1, 1)), axis=1)
        n_struct_features = 9
    elif variant == "SA-PGA":
        pga_train = spectral_train[:, 0]
        pga_test = spectral_test[:, 0]
        pga_val = spectral_val[:, 0]
        physical_train = np.concatenate((physical_train, sa_t1_train.reshape(-1, 1), pga_train.reshape(-1, 1)), axis=1)
        physical_test = np.concatenate((physical_test, sa_t1_test.reshape(-1, 1), pga_test.reshape(-1, 1)), axis=1)
        physical_val = np.concatenate((physical_val, sa_t1_val.reshape(-1, 1), pga_val.reshape(-1, 1)), axis=1)
        n_struct_features = 10
    elif variant == "RS":
        physical_train = np.concatenate((physical_train, rs_train), axis=1)
        physical_test = np.concatenate((physical_test, rs_test), axis=1)
        physical_val = np.concatenate((physical_val, rs_val), axis=1)
        n_struct_features = physical_train.shape[1]
    elif variant == "SPEC-DR":
        n_struct_features = physical_train.shape[1]
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    # space transformation ensure the stability of the neural network
    # when data is on the log scale for acceleration
    accel_min = np.min(accel_train, axis=0)
    accel_train = accel_train + accel_min
    accel_test = accel_test + accel_min
    accel_val = accel_val + accel_min

    edp_train = np.concatenate((drift_train, accel_train), axis=1)
    edp_test = np.concatenate((drift_test, accel_test), axis=1)
    edp_val = np.concatenate((drift_val, accel_val), axis=1)

    return PreparedData(
        gm_train=gm_train,
        gm_test=gm_test,
        gm_val=gm_val,
        physical_train=physical_train,
        physical_test=physical_test,
        physical_val=physical_val,
        edp_train=edp_train,
        edp_test=edp_test,
        edp_val=edp_val,
        n_struct_features=n_struct_features,
    )


#=============================================
# Run model modules
#=============================================
def load_data(base_path: Path, samples_path: Path, train_size: int, tag: str):
    gm_train_samples = np.load(samples_path / f"EQ_train_{train_size}_{tag}.npz")["data"]
    gm_test_samples = np.load(base_path / "EQ_test.npz")["data"]
    gm_val_samples = np.load(base_path / "EQ_val.npz")["data"]

    physical_samples_train = np.load(samples_path / f"physical_samples_train_{train_size}_{tag}.npz")["data"]
    physical_samples_test = np.load(base_path / "physical_samples_test.npz")["data"]
    physical_samples_val = np.load(base_path / "physical_samples_val.npz")["data"]

    spectral_features_train_samples = np.log(
        np.load(samples_path / f"spectral_features_train_{train_size}_{tag}.npz")["data"]
    )
    spectral_features_test_samples = np.log(np.load(base_path / "spectral_features_test.npz")["data"])
    spectral_features_val_samples = np.log(np.load(base_path / "spectral_features_val.npz")["data"])

    rs_train_samples = np.load(samples_path / f"RS_train_{train_size}_{tag}.npz")["data"] / 9.81
    rs_test_samples = np.load(base_path / "RS_test.npz")["data"] / 9.81
    rs_val_samples = np.load(base_path / "RS_val.npz")["data"] / 9.81

    peak_drift_per_floor_train = np.log(
        np.load(samples_path / f"peak_drift_per_floor_train_{train_size}_{tag}.npz")["data"]
    )
    peak_drift_per_floor_test = np.log(np.load(base_path / "peak_drift_per_floor_test.npz")["data"])
    peak_drift_per_floor_val = np.log(np.load(base_path / "peak_drift_per_floor_val.npz")["data"])

    peak_accel_per_floor_train = np.log(
        np.load(samples_path / f"peak_accel_per_floor_train_{train_size}_{tag}.npz")["data"] / 9.81
    )
    peak_accel_per_floor_test = np.log(np.load(base_path / "peak_accel_per_floor_test.npz")["data"] / 9.81)
    peak_accel_per_floor_val = np.log(np.load(base_path / "peak_accel_per_floor_val.npz")["data"] / 9.81)

    return {
        "gm_train_samples": gm_train_samples,
        "gm_test_samples": gm_test_samples,
        "gm_val_samples": gm_val_samples,
        "physical_samples_train": physical_samples_train,
        "physical_samples_test": physical_samples_test,
        "physical_samples_val": physical_samples_val,
        "spectral_features_train_samples": spectral_features_train_samples,
        "spectral_features_test_samples": spectral_features_test_samples,
        "spectral_features_val_samples": spectral_features_val_samples,
        "RS_train_samples": rs_train_samples,
        "RS_test_samples": rs_test_samples,
        "RS_val_samples": rs_val_samples,
        "peak_drift_per_floor_train": peak_drift_per_floor_train,
        "peak_drift_per_floor_test": peak_drift_per_floor_test,
        "peak_drift_per_floor_val": peak_drift_per_floor_val,
        "peak_accel_per_floor_train": peak_accel_per_floor_train,
        "peak_accel_per_floor_test": peak_accel_per_floor_test,
        "peak_accel_per_floor_val": peak_accel_per_floor_val,
    }


def prepare_variant_inputs(data: Dict, variant: str):
    variant = variant.upper()

    physical_samples_train = data["physical_samples_train"].copy()
    physical_samples_test = data["physical_samples_test"].copy()
    physical_samples_val = data["physical_samples_val"].copy()

    spectral_features_train_samples = data["spectral_features_train_samples"]
    spectral_features_test_samples = data["spectral_features_test_samples"]
    spectral_features_val_samples = data["spectral_features_val_samples"]

    sa_T1_train = spectral_features_train_samples[:, 2]
    sa_T1_test = spectral_features_test_samples[:, 2]
    sa_T1_val = spectral_features_val_samples[:, 2]

    if variant in {"PLE", "NH", "UAE", "SAE"}:
        n_struct_features = physical_samples_train.shape[1]

    elif variant == "SA-PGA":
        pga_train = spectral_features_train_samples[:, 0]
        pga_test = spectral_features_test_samples[:, 0]
        pga_val = spectral_features_val_samples[:, 0]

        physical_samples_train = np.concatenate(
            (physical_samples_train, sa_T1_train.reshape(-1, 1), pga_train.reshape(-1, 1)), axis=1
        )
        physical_samples_test = np.concatenate(
            (physical_samples_test, sa_T1_test.reshape(-1, 1), pga_test.reshape(-1, 1)), axis=1
        )
        physical_samples_val = np.concatenate(
            (physical_samples_val, sa_T1_val.reshape(-1, 1), pga_val.reshape(-1, 1)), axis=1
        )
        n_struct_features = physical_samples_train.shape[1]

    else:
        raise ValueError(f"Unsupported variant for input preparation: {variant}")
    # Acceleration space transformation for better stabilization of the model
    accel_min = np.min(data["peak_accel_per_floor_train"], axis=0)

    peak_accel_per_floor_train = data["peak_accel_per_floor_train"] + accel_min
    peak_accel_per_floor_test = data["peak_accel_per_floor_test"] + accel_min
    peak_accel_per_floor_val = data["peak_accel_per_floor_val"] + accel_min

    edp_train_samples = np.concatenate((data["peak_drift_per_floor_train"], peak_accel_per_floor_train), axis=1)
    edp_test_samples = np.concatenate((data["peak_drift_per_floor_test"], peak_accel_per_floor_test), axis=1)
    edp_val_samples = np.concatenate((data["peak_drift_per_floor_val"], peak_accel_per_floor_val), axis=1)

    return {
        "gm_train_samples": data["gm_train_samples"],
        "gm_test_samples": data["gm_test_samples"],
        "gm_val_samples": data["gm_val_samples"],
        "physical_samples_train": physical_samples_train,
        "physical_samples_test": physical_samples_test,
        "physical_samples_val": physical_samples_val,
        "EDP_train_samples": edp_train_samples,
        "EDP_test_samples": edp_test_samples,
        "EDP_val_samples": edp_val_samples,
        "n_struct_features": n_struct_features,
    }



def build_initial_latent_range():
    """
    Define the range of latent space to calibrate the model
    for different values, once calibrated one of each will be 
    selected based on the least tuning set error
    """
    return [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def build_refined_latent_range(latent_range: List[int], val_results_all: Dict[str, Dict]):
    """
    Ensure the latent selected by 'build_initial_latent_range'
    is the best by iterating over narrower range to search 
    if lower value exist
    """
    losses = []
    for ltn in latent_range:
        metrics = val_results_all[str(ltn)]
        nmse = metrics.get("nmse_scalar", metrics.get("output_nmse_scalar"))
        losses.append(nmse)

    best_idx = int(np.argmin(losses))
    selected_latent = latent_range[best_idx]

    if best_idx + 1 < len(latent_range):
        next_latent = latent_range[best_idx + 1]
    else:
        next_latent = selected_latent + 250

    step = max(1, int((next_latent - selected_latent) / 10))
    refined = list(np.arange(selected_latent + 1, next_latent, step))
    return refined


def choose_trainer(variant: str):
    """
    Select the trainer based on user defined variant
    """
    v = variant.upper()
    if v == "PLE":
        return train_model_PLE
    if v == "SAE":
        return train_model_SAE
    if v == "UAE":
        return train_model_UAE
    return train_model # this return anything rather than the above variants


def save_val_results_excel(path: Path, val_results_all: Dict[str, Dict]):
    """
    save the validation Results
    """
    if path.exists():
        mode = "a"
        if_sheet_exists = "replace"
    else:
        mode = "w"
        if_sheet_exists = None

    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, if_sheet_exists=if_sheet_exists) as writer:
        for key, value in val_results_all.items():
            pd.DataFrame(value, index=[0]).to_excel(writer, sheet_name=str(key), index=False)


def save_recon_excel(path: Path, recon_val_loss_all: Dict[str, float]):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        pd.DataFrame(recon_val_loss_all, index=[0]).to_excel(writer, sheet_name="recon_loss", index=False)


def maybe_save_model(model_obj, save_dir: Path, variant: str, train_size: int, latent_dim: int, tag: str, suffix: str = ""):
    # TODO: chamge the name of the function
    if model_obj is None:
        return
    model_path = save_dir / f"{variant}_{train_size}_{latent_dim}_{tag}{suffix}"
    model_obj.save(model_path)


def run_latent_sweep(args):
    """
    Train the models based on the user hyperparameter 
    and required model data
    """
    set_seed(args.seed)

    base_path = Path(args.base_path)
    samples_path = Path(args.samples_path)
    results_dir = Path(args.results_dir)
    models_dir = results_dir / "models" / args.variant
    excel_dir = results_dir / "excel"
    models_dir.mkdir(parents=True, exist_ok=True)
    excel_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_data(base_path, samples_path, args.train_size, args.tag)
    prepared = prepare_variant_inputs(raw_data, args.variant)

    input_timesteps = args.input_timesteps
    ts_features = args.ts_features
    n_floors = args.n_floors
    n_struct_features = prepared["n_struct_features"]

    trainer = choose_trainer(args.variant)
    latent_range = build_initial_latent_range()

    val_results_all = {}
    recon_val_loss_all = {}

    for ltn in latent_range:
        result = trainer(
            input_timesteps=input_timesteps,
            ts_features=ts_features,
            n_floors=n_floors,
            n_struct_features=n_struct_features,
            latent_dim=ltn,
            gm_train_samples=prepared["gm_train_samples"],
            physical_samples_train=prepared["physical_samples_train"],
            EDP_train_samples=prepared["EDP_train_samples"],
            gm_val_samples=prepared["gm_val_samples"],
            physical_samples_val=prepared["physical_samples_val"],
            EDP_val_samples=prepared["EDP_val_samples"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            all_edp=True,
        )

        if args.variant.upper() == "SAE":
            _, model, val_results = result
            recon_val_loss = np.nan
        elif args.variant.upper() == "UAE":
            _, ae_model, latent_model, reg_model, val_results = result
            model = reg_model
            recon_val_loss = np.nan
            if args.save_models:
                maybe_save_model(ae_model, models_dir, f"{args.variant}_AE", args.train_size, ltn, args.tag)
        else:
            _, model, _, val_results, recon_val_loss = result

        val_results_all[str(ltn)] = val_results
        recon_val_loss_all[str(ltn)] = recon_val_loss

        print(f"Latent {ltn} is calibrated")
        print("============================")

        if args.save_models:
            maybe_save_model(model, models_dir, args.variant, args.train_size, ltn, args.tag)

    if args.refine_latent:
        refined_range = build_refined_latent_range(latent_range, val_results_all)

        for ltn in refined_range:
            result = trainer(
                input_timesteps=input_timesteps,
                ts_features=ts_features,
                n_floors=n_floors,
                n_struct_features=n_struct_features,
                latent_dim=ltn,
                gm_train_samples=prepared["gm_train_samples"],
                physical_samples_train=prepared["physical_samples_train"],
                EDP_train_samples=prepared["EDP_train_samples"],
                gm_val_samples=prepared["gm_val_samples"],
                physical_samples_val=prepared["physical_samples_val"],
                EDP_val_samples=prepared["EDP_val_samples"],
                epochs=args.refine_epochs,
                batch_size=args.batch_size,
                all_edp=True,
            )

            if args.variant.upper() == "SAE":
                _, model, val_results = result
                recon_val_loss = np.nan
            elif args.variant.upper() == "UAE":
                _, ae_model, latent_model, reg_model, val_results = result
                model = reg_model
                recon_val_loss = np.nan
                if args.save_models:
                    maybe_save_model(ae_model, models_dir, f"{args.variant}_AE", args.train_size, ltn, args.tag, suffix="_v2")
            else:
                _, model, _, val_results, recon_val_loss = result

            val_results_all[str(ltn)] = val_results
            recon_val_loss_all[str(ltn)] = recon_val_loss

            print(f"Latent {ltn} is calibrated")
            print("============================")

            if args.save_models:
                maybe_save_model(model, models_dir, args.variant, args.train_size, ltn, args.tag, suffix="_v2")

    val_excel = excel_dir / f"{args.variant}_{args.train_size}_{args.tag}.xlsx"
    recon_excel = excel_dir / f"{args.variant}_recon_{args.train_size}_{args.tag}.xlsx"

    save_val_results_excel(val_excel, val_results_all)
    save_recon_excel(recon_excel, recon_val_loss_all)

    print(f"Validation results saved to: {val_excel}")
    print(f"Reconstruction results saved to: {recon_excel}")



#=============================================
# Post process the results
#=============================================

def to_ebf(y2d: np.ndarray, order: str = "building_major", n_eq=600, n_buildings=190) -> np.ndarray:
    n, f = y2d.shape # --> f corresponds to the number of floors
    try:
        n_eq, n_buildings = 600, 190
    except:
        raise ValueError(f"Unexpected sample count. Fixed the sample count!")

    if order == "building_major":
        return y2d.reshape(n_buildings, n_eq, f).transpose(1, 0, 2)
    if order == "eq_major":
        return y2d.reshape(n_eq, n_buildings, f)
    raise ValueError("order must be 'building_major' or 'eq_major'")

def _nan_safe_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x))

def _pearson_cc(x: np.ndarray, y: np.ndarray) -> float:
    x_mean = _nan_safe_mean(x)
    y_mean = _nan_safe_mean(y)
    dx = x - x_mean
    dy = y - y_mean
    num = np.nansum(dx * dy)
    den = np.sqrt(np.nansum(dx ** 2) * np.nansum(dy ** 2)) + 1e-12
    return float(num / den)

def _r2_score(x: np.ndarray, y: np.ndarray) -> float:
    ss_res = np.nansum((x - y) ** 2)
    ss_tot = np.nansum((x - _nan_safe_mean(x)) ** 2) + 1e-12
    return float(abs(1.0 - ss_res / ss_tot))

def _w1_distance(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    return float(wasserstein_distance(x[mask], y[mask]))

def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    mse = _nan_safe_mean((x - y) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)

def _w1_normalized(x: np.ndarray, y: np.ndarray) -> float:
    w1 = _w1_distance(x, y)
    if np.isnan(w1):
        return np.nan
    std = np.nanstd(x) + 1e-12
    return float(w1 / std)

def corr_mats_per_building(x: np.ndarray) -> np.ndarray:
    n_motions, n_buildings, n_features = x.shape
    corr = np.empty((n_buildings, n_features, n_features))
    for b in range(n_buildings):
        corr[b] = np.corrcoef(x[:, b, :], rowvar=False)
    return corr

def compute_metrics_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    assert y_true.shape == y_pred.shape
    n_eq, n_b, n_f = y_true.shape
    cc = np.full((n_b, n_f), np.nan, dtype=float)
    r2 = np.full((n_b, n_f), np.nan, dtype=float)
    w1 = np.full((n_b, n_f), np.nan, dtype=float)
    w1n = np.full((n_b, n_f), np.nan, dtype=float)
    nrmse = np.full((n_b, n_f), np.nan, dtype=float)

    for b in range(n_b):
        for f in range(n_f):
            t = y_true[:, b, f]
            p = y_pred[:, b, f]
            if np.all(np.isnan(t)) or np.all(np.isnan(p)):
                continue
            cc[b, f] = _pearson_cc(t, p)
            r2[b, f] = _r2_score(t, p)
            w1[b, f] = _w1_distance(t, p)
            w1n[b, f] = _w1_normalized(t, p)
            nrmse[b, f] = _rmse(t, p)/np.abs(np.nanmax(y_true)-np.nanmin(y_true))

    return {"CC": cc, "R2": r2, "W1": w1, "W1_norm": w1n, "NRMSE": nrmse}

def compute_aer_per_building_floor(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 0.0012) -> np.ndarray:
    return y_pred / (y_true + eps)

def trim_tail(x: np.ndarray, frac: float = 0.1, remove: str = "high") -> np.ndarray:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    k = int(frac * x.size)
    if k <= 0:
        return np.sort(x)
    x_sorted = np.sort(x)
    return x_sorted[:-k] if remove == "high" else x_sorted[k:]

def calculate_metric_across_variants(
    y_true_ebf_drift: np.ndarray,
    y_true_ebf_accel: np.ndarray,
    y_pred_ebf_drift: np.ndarray,
    y_pred_ebf_accel: np.ndarray,
):
    m_d = compute_metrics_per_output(y_true_ebf_drift, y_pred_ebf_drift)
    m_a = compute_metrics_per_output(y_true_ebf_accel, y_pred_ebf_accel)
    aer_drift = compute_aer_per_building_floor(y_true_ebf_drift, y_pred_ebf_drift)
    aer_accel = compute_aer_per_building_floor(y_true_ebf_accel, y_pred_ebf_accel)

    for key in list(m_d.keys()):
        m_d[key] = m_d[key].mean(axis=0).reshape(-1, 1)
        m_a[key] = m_a[key].mean(axis=0).reshape(-1, 1)

    y_true_all = np.concatenate([y_true_ebf_drift, y_true_ebf_accel], axis=-1)
    y_pred_all = np.concatenate([y_pred_ebf_drift, y_pred_ebf_accel], axis=-1)
    c_true = corr_mats_per_building(y_true_all)
    c_pred = corr_mats_per_building(y_pred_all)
    d = c_true - c_pred
    dd = d[:, :5, :5]
    aa = d[:, 5:, 5:]
    da = d[:, :5, 5:]
    frob_all = np.linalg.norm(d, axis=(1, 2)) / np.maximum(np.linalg.norm(c_true, axis=(1, 2)), 1e-12)
    frob_d = np.linalg.norm(dd, axis=(1, 2)) / np.maximum(np.linalg.norm(c_true[:, :5, :5], axis=(1, 2)), 1e-12)
    frob_a = np.linalg.norm(aa, axis=(1, 2)) / np.maximum(np.linalg.norm(c_true[:, 5:, 5:], axis=(1, 2)), 1e-12)
    frob_da = np.linalg.norm(da, axis=(1, 2)) / np.maximum(np.linalg.norm(c_true[:, :5, 5:], axis=(1, 2)), 1e-3)

    aer_d = {"r": np.nanmean(aer_drift)}
    aer_a = {"r": np.nanmean(aer_accel)}
    return m_d, m_a, aer_d, aer_a, frob_d, frob_a, frob_da, frob_all
