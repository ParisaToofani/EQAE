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

NormType = Literal["std", "range", "iqr", "mean", "rms", "none"]
StdType = Literal["y_true", "y_pred", "pooled"]


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# Losses / metrics used during training
# -----------------------------------------------------------------------------

def nmse_dataset_vectorized(x_true: tf.Tensor, x_pred: tf.Tensor, eps: float = 1e-10) -> tf.Tensor:
    x_true = tf.cast(x_true, dtype=tf.float32)
    x_pred = tf.cast(x_pred, dtype=tf.float32)
    mse = tf.reduce_mean(tf.square(x_true - x_pred))
    variance_true = tf.reduce_mean(tf.square(x_true - tf.reduce_mean(x_true)))
    return mse / (variance_true + eps)


def make_nmse_scalar(train_var_scalar: float):
    train_var_scalar = tf.constant(train_var_scalar, dtype=tf.float32)
    eps = tf.constant(1e-12, dtype=tf.float32)

    def nmse_scalar(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        return mse / tf.maximum(train_var_scalar, eps)

    nmse_scalar.__name__ = "nmse_scalar"
    return nmse_scalar


def make_nmse_per_output(train_var_per_output: np.ndarray):
    train_var_per_output = tf.constant(train_var_per_output, dtype=tf.float32)
    eps = tf.constant(1e-12, dtype=tf.float32)

    def nmse_per_output(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        se = tf.square(y_pred - y_true)
        mse_per_out = tf.reduce_mean(se, axis=0)
        nmse_vec = mse_per_out / tf.maximum(train_var_per_output, eps)
        return tf.reduce_mean(nmse_vec)

    nmse_per_output.__name__ = "nmse_per_output"
    return nmse_per_output


# -----------------------------------------------------------------------------
# Shape helpers and post-processing metrics
# -----------------------------------------------------------------------------

def to_ebf(y2d: np.ndarray, order: str = "building_major") -> np.ndarray:
    n, f = y2d.shape
    if n == 600 * 190:
        n_eq, n_buildings = 600, 190
    elif n == 680 * 200:
        n_eq, n_buildings = 680, 200
    else:
        raise ValueError(f"Unexpected sample count {n}. Expected 600*190 or 680*200.")

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


def _nrmse(x: np.ndarray, y: np.ndarray) -> float:
    mse = _nan_safe_mean((x - y) ** 2)
    rmse = np.sqrt(mse)
    value_range = (np.nanmax(x) - np.nanmin(x)) + 1e-12
    return float(rmse / value_range)


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
            nrmse[b, f] = _nrmse(t, p)

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
    variant: str = "SA-PGA",
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


# -----------------------------------------------------------------------------
# Sampling and dataset construction helpers
# -----------------------------------------------------------------------------

def struct_indices_selection(indices: np.ndarray, number_of_samples: int = 2720, replace: bool = True, seed: Optional[int] = None) -> np.ndarray:
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
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    if sampling_type == "random":
        if indices is None:
            raise ValueError("indices are required for random selection")
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


def _space_filling_select(x: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    k = int(min(k, x.shape[0]))
    if k <= 0:
        return np.array([], dtype=int)
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    km.fit(x)
    dist = np.linalg.norm(x[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    picked = np.argmin(dist, axis=0)
    return np.unique(np.sort(picked))


def create_pairs(eq_indices: Sequence[int], struct_indices: Sequence[int], se_ratio: int = 1) -> np.ndarray:
    pairs: List[Tuple[int, int]] = []
    for eq_idx in eq_indices:
        sampled_struct = np.random.choice(struct_indices, size=se_ratio, replace=True)
        for s_idx in sampled_struct:
            pairs.append((int(eq_idx), int(s_idx)))
    return np.asarray(pairs, dtype=int)


def _cartesian_pairs(a: Sequence[int], b: Sequence[int]) -> np.ndarray:
    return np.asarray([(int(i), int(j)) for i in a for j in b], dtype=int)


def build_fixed_splits(
    spec: np.ndarray,
    struct: np.ndarray,
    test_sizes: Tuple[int, int] = (680, 200),
    val_fracs: Tuple[float, float] = (0.1, 0.001),
    seed_test: int = 42,
    seed_val: int = 5000,
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
    f = interp1d(time_axis, signal, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(target_time)


def dataset_creator(pair_array: np.ndarray, input_samples: np.ndarray, target: str) -> np.ndarray:
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
    t1 = 0.77
    times = np.arange(0, 5, 0.02)
    pga = spect[0] / 9.81
    s_a_0_2 = signal_interpolator(times, spect, 0.2) / 9.81
    s_a_t_1 = signal_interpolator(times, spect, t1) / 9.81
    s_a_1 = signal_interpolator(times, spect, 1) / 9.81
    s_a_2_5 = signal_interpolator(times, spect, 2.5) / 9.81
    return [pga, s_a_0_2, s_a_t_1, s_a_1, s_a_2_5]


def spectral_feature_parallel_actual(spect: np.ndarray, physical_sample_t1: float) -> List[float]:
    times = np.arange(0, 5, 0.02)
    pga = spect[0] / 9.81
    s_a_0_2 = signal_interpolator(times, spect, 0.2) / 9.81
    s_a_t_1 = signal_interpolator(times, spect, physical_sample_t1) / 9.81
    s_a_1 = signal_interpolator(times, spect, 1) / 9.81
    s_a_2_5 = signal_interpolator(times, spect, 2.5) / 9.81
    return [pga, s_a_0_2, s_a_t_1, s_a_1, s_a_2_5]


# -----------------------------------------------------------------------------
# Data IO and preprocessing
# -----------------------------------------------------------------------------

@dataclass
class BaseData:
    gm: np.ndarray
    physical: np.ndarray
    spectral: np.ndarray
    rs: np.ndarray
    drift: np.ndarray
    accel: np.ndarray


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


def load_raw_data(data_root: str | Path) -> Dict[str, np.ndarray]:
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
        # Physical are raw
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


# -----------------------------------------------------------------------------
# Model training
# -----------------------------------------------------------------------------

def train_model(
    input_timesteps: int,
    ts_features: int,
    n_outputs: int,
    n_struct_features: int,
    latent_dim: int,
    gm_train_samples: np.ndarray,
    physical_samples_train: np.ndarray,
    edp_train_samples: np.ndarray,
    gm_val_samples: np.ndarray,
    physical_samples_val: np.ndarray,
    edp_val_samples: np.ndarray,
    epochs: int = 200,
    batch_size: int = 32,
    dim1: int = 2048,
    dim2: int = 1024,
    all_edp: bool = True,
):
    K.clear_session()

    ts_input = Input(shape=(input_timesteps, ts_features), name="ts_input")
    struct_input = Input(shape=(n_struct_features,), name="struct_input")

    x = Flatten()(ts_input)
    x = Dropout(0.15)(x)
    x = Dense(dim1, activation="relu", kernel_regularizer=regularizers.l2(5e-3))(x)
    x = Dropout(0.2)(x)
    x = Dense(dim2, activation="relu", kernel_regularizer=regularizers.l2(5e-3))(x)
    x = Dropout(0.2)(x)
    x = Dense(800, activation="relu", kernel_regularizer=regularizers.l2(5e-3))(x)
    latent = Dense(latent_dim, activation="elu", name="latent")(x)

    dec = Dense(800, activation="relu", kernel_regularizer=regularizers.l2(5e-3))(latent)
    dec = Dropout(0.2)(dec)
    dec = Dense(dim2, activation="relu", kernel_regularizer=regularizers.l2(5e-3))(dec)
    dec = Dropout(0.2)(dec)
    dec = Dense(dim1, activation="relu", kernel_regularizer=regularizers.l2(5e-3))(dec)
    dec = Dropout(0.15)(dec)
    dec = Dense(input_timesteps, activation="linear", kernel_regularizer=regularizers.l2(5e-3))(dec)
    reconstruction = Reshape((input_timesteps, ts_features), name="reconstruction")(dec)

    combined = Concatenate(name="combined")([latent, struct_input])
    combined_mean = tf.reduce_mean(combined, axis=1, keepdims=True)
    combined_std = tf.math.reduce_std(combined, axis=1, keepdims=True)
    combined = (combined - combined_mean) / tf.maximum(combined_std, 1e-12)
    reg = Dense(int(max(latent_dim, 2)), activation="selu")(combined)
    reg = Dense(int(max(latent_dim // 2, 1)), activation="selu")(reg)
    output = Dense(n_outputs, activation="linear", name="output")(reg)

    model = Model(inputs=[ts_input, struct_input], outputs=output)
    recon_model = Model(inputs=[ts_input, struct_input], outputs=reconstruction)

    train_var_scalar = float(np.var(edp_train_samples, ddof=0))
    train_var_per_output = np.var(edp_train_samples, axis=0, ddof=0).astype("float32")
    nmse_scalar = make_nmse_scalar(train_var_scalar)
    nmse_per_output = make_nmse_per_output(train_var_per_output)

    metrics = [nmse_scalar, nmse_per_output]
    if all_edp and n_outputs >= 10:
        def drift_nmse(y_true, y_pred):
            return nmse_scalar(y_true[:, :5], y_pred[:, :5])
        drift_nmse.__name__ = "drift_nmse"

        def accel_nmse(y_true, y_pred):
            return nmse_scalar(y_true[:, 5:10], y_pred[:, 5:10])
        accel_nmse.__name__ = "accel_nmse"
        metrics.extend([drift_nmse, accel_nmse])

    opt = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        global_clipnorm=1.0,
        use_ema=True,
        weight_decay=1e-7,
    )
    model.compile(optimizer=opt, loss=nmse_scalar, metrics={"output": metrics})

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
    ]

    history = model.fit(
        x=[gm_train_samples, physical_samples_train],
        y={"output": edp_train_samples},
        validation_data=([gm_val_samples, physical_samples_val], {"output": edp_val_samples}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )

    val_results = model.evaluate(
        x=[gm_val_samples, physical_samples_val],
        y={"output": edp_val_samples},
        verbose=0,
        return_dict=True,
    )

    def nmse_gm(x_true: tf.Tensor, x_pred: tf.Tensor, eps: float = 1e-10) -> tf.Tensor:
        x_true = tf.squeeze(x_true, axis=-1)
        x_pred = tf.squeeze(x_pred, axis=-1)
        err2 = tf.square(x_true - x_pred)
        mse_per_sample = tf.reduce_mean(err2, axis=[1])
        mse = tf.reduce_mean(mse_per_sample)
        nmse = mse / (gm_train_samples.var() + eps)
        return nmse

    recon_model.compile(optimizer="adam", loss=nmse_gm, run_eagerly=True)
    recon_val_loss = recon_model.evaluate([gm_val_samples, physical_samples_val], gm_val_samples, verbose=0)

    return history, model, recon_model, val_results, float(recon_val_loss)
