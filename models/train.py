
from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Concatenate, Reshape, LayerNormalization

from utils.utils import make_nmse_scalar, make_nmse_per_output, nmse_dataset_vectorized

def _make_slice_metric(nmse_scalar_fn, start: int, end: int, name: str):
    def metric(y_true, y_pred):
        return nmse_scalar_fn(y_true[:, start:end], y_pred[:, start:end])
    metric.__name__ = name
    return metric

def _make_common_metrics(edp_train_samples: np.ndarray, all_edp: bool = True):
    train_var_scalar = float(np.var(edp_train_samples, ddof=0))
    train_var_per_output = np.var(edp_train_samples, axis=0, ddof=0).astype("float32")
    nmse_scalar = make_nmse_scalar(train_var_scalar)
    nmse_per_output = make_nmse_per_output(train_var_per_output)

    metric_list = [nmse_scalar, nmse_per_output]
    if all_edp:
        metric_list.extend([
            _make_slice_metric(nmse_scalar, 0, 5, "drift_nmse"),
            _make_slice_metric(nmse_scalar, 5, 10, "accel_nmse"),
        ])
    return nmse_scalar, nmse_per_output, metric_list

def _make_callbacks():
    return [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
    ]

def _evaluate_reconstruction_model(
    recon_model: Model,
    gm_train_samples: np.ndarray,
    gm_val_samples: np.ndarray,
    physical_samples_train: np.ndarray,
    physical_samples_val: np.ndarray,
):
    def nmse_gm(x_true, x_pred, eps=1e-10):
        x_true = tf.squeeze(x_true, axis=-1)
        x_pred = tf.squeeze(x_pred, axis=-1)
        err2 = tf.square(x_true - x_pred)
        mse_per_sample = tf.reduce_mean(err2, axis=[1])
        mse = tf.reduce_mean(mse_per_sample)
        return mse / (gm_train_samples.var() + eps)

    recon_model.compile(optimizer="adam", loss=nmse_gm, run_eagerly=True)

    recon_train_loss = recon_model.evaluate([gm_train_samples, physical_samples_train], gm_train_samples, verbose=0)
    recon_val_loss = recon_model.evaluate([gm_val_samples, physical_samples_val], gm_val_samples, verbose=0)
    return recon_train_loss, recon_val_loss

def _print_val_results(title: str, val_results: Dict[str, float]) -> None:
    print(f"\n=========== FINAL METRICS: {title} ===========")
    print("\n--- Validation set ---")
    for k, v in val_results.items():
        print(f"{k:30s}: {float(v):.6f}")

def train_model(
    input_timesteps,
    ts_features,
    n_floors,
    n_struct_features=8,
    latent_dim=2,
    gm_train_samples=None,
    physical_samples_train=None,
    EDP_train_samples=None,
    gm_val_samples=None,
    physical_samples_val=None,
    EDP_val_samples=None,
    epochs=200,
    batch_size=32,
    dim1=2048,
    dim2=1024,
    all_edp=True,
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
    x = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(5e-3))(x)

    latent = Dense(latent_dim, activation="gelu", name="latent")(x)

    dec = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(5e-3))(latent)
    x = Dropout(0.2)(x)
    dec = Dense(dim2, activation="relu", kernel_regularizer=regularizers.l2(5e-3))(dec)
    x = Dropout(0.2)(x)
    dec = Dense(dim1, activation="relu", kernel_regularizer=regularizers.l2(5e-3))(dec)
    x = Dropout(0.15)(x)
    dec = Dense(input_timesteps, activation="linear", kernel_regularizer=regularizers.l2(5e-3))(dec)
    reconstruction = Reshape((input_timesteps, ts_features), name="reconstruction")(dec)

    combined = Concatenate(name="combined")([latent, struct_input])
    combined = LayerNormalization()(combined)
    reg = Dense(int(latent_dim), activation="selu")(combined)
    reg = Dense(max(1, int(latent_dim / 2)), activation="selu")(reg)
    output = Dense(n_floors, activation="linear", name="output")(reg)

    model = Model(inputs=[ts_input, struct_input], outputs=output)
    recon_model = Model(inputs=[ts_input, struct_input], outputs=reconstruction)

    nmse_scalar, _, metric_list = _make_common_metrics(EDP_train_samples, all_edp=all_edp)

    opt = tf.keras.optimizers.AdamW(learning_rate=1e-4, global_clipnorm=0.5, use_ema=True)

    if all_edp:
        model.compile(optimizer=opt, loss=nmse_scalar, metrics={"output": metric_list})
    else:
        model.compile(optimizer=opt, loss=nmse_scalar)

    history = model.fit(
        x=[gm_train_samples, physical_samples_train],
        y={"output": EDP_train_samples},
        validation_data=([gm_val_samples, physical_samples_val], {"output": EDP_val_samples}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=_make_callbacks(), # This ensure avoiding the overfitting
        verbose=0,
        shuffle=True,
    )

    val_results = model.evaluate(
        x=[gm_val_samples, physical_samples_val],
        y={"output": EDP_val_samples},
        verbose=0,
        return_dict=True,
    )

    # _print_val_results("SA-PGA", val_results)

    _, recon_val_loss = _evaluate_reconstruction_model(
        recon_model, gm_train_samples, gm_val_samples, physical_samples_train, physical_samples_val
    )
    return history, model, recon_model, val_results, recon_val_loss

def train_model_PLE(
    input_timesteps,
    ts_features,
    n_floors,
    n_struct_features=8,
    latent_dim=2,
    gm_train_samples=None,
    physical_samples_train=None,
    EDP_train_samples=None,
    gm_val_samples=None,
    physical_samples_val=None,
    EDP_val_samples=None,
    epochs=200,
    batch_size=32,
    dim1=2048,
    dim2=1024,
    all_edp=True,
):
    K.clear_session()

    ts_input = Input(shape=(input_timesteps, ts_features), name="ts_input")
    struct_input = Input(shape=(n_struct_features,), name="struct_input")

    x = Flatten()(ts_input)
    x = Dense(dim1, activation="relu")(x)
    x = Dense(dim2, activation="relu")(x)
    x = Dense(800, activation="relu")(x)

    latent = Dense(latent_dim, activation="gelu", name="latent")(x)

    latent_for_decoder = tf.stop_gradient(latent)
    dec = Dense(800, activation="relu")(latent_for_decoder)
    dec = Dense(dim2, activation="relu")(dec)
    dec = Dense(dim1, activation="relu")(dec)
    dec = Dense(input_timesteps, activation="relu")(dec)
    reconstruction = Reshape((input_timesteps, ts_features), name="reconstruction")(dec)

    combined = Concatenate(name="combined")([latent, struct_input])
    combined = LayerNormalization()(combined)
    reg = Dense(int(latent_dim), activation="gelu")(combined)
    reg = Dense(max(1, int(latent_dim / 2)), activation="gelu")(reg)
    output = Dense(n_floors, activation="linear", name="output")(reg)

    model = Model(inputs=[ts_input, struct_input], outputs=output)
    recon_model = Model(inputs=[ts_input, struct_input], outputs=reconstruction)

    nmse_scalar, _, metric_list = _make_common_metrics(EDP_train_samples, all_edp=all_edp)

    opt = tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        global_clipnorm=5,
        use_ema=True,
        weight_decay=1e-7, # further regularization to avoid overfitting
    )

    if all_edp:
        model.compile(optimizer=opt, loss=nmse_scalar, metrics={"output": metric_list})
    else:
        model.compile(optimizer=opt, loss=nmse_scalar)

    history = model.fit(
        x=[gm_train_samples, physical_samples_train],
        y={"output": EDP_train_samples},
        validation_data=([gm_val_samples, physical_samples_val], {"output": EDP_val_samples}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=_make_callbacks(),
        verbose=0,
        shuffle=True,
    )

    val_results = model.evaluate(
        x=[gm_val_samples, physical_samples_val],
        y={"output": EDP_val_samples},
        verbose=0,
        return_dict=True,
    )

    # _print_val_results("PLE", val_results)

    _, recon_val_loss = _evaluate_reconstruction_model(
        recon_model, gm_train_samples, gm_val_samples, physical_samples_train, physical_samples_val
    )

    return history, model, recon_model, val_results, recon_val_loss

def train_model_SAE(
    input_timesteps,
    ts_features,
    n_floors,
    n_struct_features=8,
    latent_dim=2,
    gm_train_samples=None,
    physical_samples_train=None,
    EDP_train_samples=None,
    gm_val_samples=None,
    physical_samples_val=None,
    EDP_val_samples=None,
    epochs=200,
    batch_size=32,
    dim1=2048,
    dim2=1024,
    all_edp=True,
):
    K.clear_session()

    ts_input = Input(shape=(input_timesteps, ts_features), name="ts_input")
    struct_input = Input(shape=(n_struct_features,), name="struct_input")

    x = Flatten()(ts_input)
    x = Dense(dim1, activation="relu")(x)
    x = Dense(dim2, activation="relu")(x)
    x = Dense(800, activation="relu", kernel_regularizer=regularizers.l2(5e-2))(x)

    latent = Dense(latent_dim, activation="elu", name="latent", kernel_regularizer=regularizers.l2(5e-2))(x)

    dec = Dense(800, activation="relu", kernel_regularizer=regularizers.l2(5e-2))(latent)
    dec = Dense(dim2, activation="relu")(dec)
    dec = Dense(dim1, activation="relu")(dec)
    dec = Dense(input_timesteps, activation="linear")(dec)
    reconstruction = Reshape((input_timesteps, ts_features), name="reconstruction")(dec)

    combined = Concatenate(name="combined")([latent, struct_input])
    reg = Dense(int(latent_dim), activation="elu", kernel_regularizer=regularizers.l2(1e-4))(combined)
    reg = Dense(max(1, int(latent_dim / 2)), activation="elu", kernel_regularizer=regularizers.l2(1e-4))(reg)
    output = Dense(n_floors, activation="linear", name="output", kernel_regularizer=regularizers.l2(1e-4))(reg)

    model = Model(inputs=[ts_input, struct_input], outputs=[reconstruction, output])

    class NMSE(tf.keras.losses.Loss):
        def __init__(self, var, eps=1e-8, name="nmse"):
            super().__init__(name=name)
            self.var = tf.constant(var, tf.float32)
            self.eps = eps
        def call(self, y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            return mse / (self.var + self.eps)

    rec_var   = gm_train_samples.var()         # or feature-wise if you prefer
    drift_var = EDP_train_samples.var()
    #--------------------------------------
    class NMSE_batch(tf.keras.losses.Loss):
        def __init__(self, eps=1e-8, name="nmse_batch"):
            super().__init__(name=name)
            self.eps = eps
        def call(self, y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            variance_true = tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_true)))
            return mse / (variance_true + self.eps)

    model.compile(
        optimizer=AdamW(1e-3),
        loss={"reconstruction": NMSE_batch(), "output": NMSE(drift_var)},
        loss_weights={"reconstruction": 0.5, "output": 0.5},
        metrics=[NMSE_batch()]
    )

    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',        # Monitor validation loss
        patience=25,               # Stop after 10 epochs with no improvement
        restore_best_weights=True # Roll back to the best weights after stopping
    )

    history = model.fit(
        x=[gm_train_samples, physical_samples_train],
        y={"reconstruction": gm_train_samples, "output": EDP_train_samples},
        validation_data=(
            [gm_val_samples, physical_samples_val],
            {"reconstruction": gm_val_samples, "output": EDP_val_samples},
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0,
    )

    val_results = model.evaluate(
        x=[gm_val_samples, physical_samples_val],
        y={"reconstruction": gm_val_samples, "output": EDP_val_samples},
        verbose=0,
        return_dict=True,
    )
    # _print_val_results("SAE", val_results)

    return history, model, val_results

def train_model_UAE(
    input_timesteps,
    ts_features,
    n_floors,
    n_struct_features=8,
    latent_dim=2,
    gm_train_samples=None,
    physical_samples_train=None,
    EDP_train_samples=None,
    gm_val_samples=None,
    physical_samples_val=None,
    EDP_val_samples=None,
    epochs=200,
    batch_size=32,
    dim1=2048,
    dim2=1024,
    all_edp=True,
):
    K.clear_session()

    ts_input = Input(shape=(input_timesteps, ts_features), name="ts_input")
    x = Flatten()(ts_input)
    x = Dense(dim1, activation="relu")(x)
    x = Dense(dim2, activation="relu")(x)
    x = Dense(800, activation="relu", kernel_regularizer=regularizers.l2(1e-7))(x)
    latent = Dense(latent_dim, activation="elu", name="latent")(x)

    dec = Dense(800, activation="relu")(latent)
    dec = Dense(dim2, activation="relu")(dec)
    dec = Dense(dim1, activation="relu")(dec)
    dec = Dense(input_timesteps, activation="relu")(dec)
    reconstruction = Reshape((input_timesteps, ts_features), name="reconstruction")(dec)

    ae_model = Model(inputs=ts_input, outputs=reconstruction, name="uae_autoencoder")
    ae_model.compile(optimizer=AdamW(1e-3), loss=nmse_dataset_vectorized)

    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',        # Monitor validation loss
    patience=25,               # Stop after 10 epochs with no improvement
    restore_best_weights=True # Roll back to the best weights after stopping
    )

    ae_model.fit(
        gm_train_samples,
        gm_train_samples,
        validation_data=(gm_val_samples, gm_val_samples),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0,
        shuffle=True,
    )

    latent_model = Model(inputs=ae_model.input, outputs=ae_model.get_layer("latent").output)
    train_latent = latent_model.predict(gm_train_samples, verbose=0)
    val_latent = latent_model.predict(gm_val_samples, verbose=0)

    reg_input = Input(shape=(latent_dim + n_struct_features,), name="uae_reg_input")
    reg = Dense(max(4, latent_dim), activation="relu", kernel_regularizer=regularizers.l2(0.0001))(reg_input)
    reg = Dense(max(2, latent_dim // 2), activation="relu", kernel_regularizer=regularizers.l2(0.0001))(reg)
    reg_output = Dense(n_floors, activation="linear", name="output")(reg)

    reg_model = Model(inputs=reg_input, outputs=reg_output, name="uae_regressor")

    nmse_scalar, _, metric_list = _make_common_metrics(EDP_train_samples, all_edp=all_edp)

    opt = tf.keras.optimizers.AdamW(learning_rate=0.001)

    if all_edp:
        reg_model.compile(optimizer=opt, loss=nmse_scalar, metrics=metric_list)
    else:
        reg_model.compile(optimizer=opt, loss=nmse_scalar)

    train_reg_input = np.concatenate((train_latent, physical_samples_train), axis=1)
    val_reg_input = np.concatenate((val_latent, physical_samples_val), axis=1)

    # Early stopping
    early_stop1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

    history = reg_model.fit(
        train_reg_input,
        EDP_train_samples,
        validation_data=(val_reg_input, EDP_val_samples),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop1],
        verbose=0,
        shuffle=True,
    )

    values = reg_model.evaluate(val_reg_input, EDP_val_samples, verbose=0, return_dict=False)
    val_results = {name: float(val) for name, val in zip(reg_model.metrics_names, values)}
    
    # _print_val_results("UAE", val_results)

    return history, ae_model, latent_model, reg_model, val_results

def train_by_variant(variant: str, **kwargs):
    variant_key = variant.upper()
    if variant_key == "SA-PGA":
        return train_model(**kwargs)
    if variant_key == "PLE":
        return train_model_PLE(**kwargs)
    if variant_key == "SAE":
        return train_model_SAE(**kwargs)
    if variant_key == "UAE":
        return train_model_UAE(**kwargs)
    raise ValueError(f"Unsupported variant: {variant}")
