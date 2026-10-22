"""Autoencoder: model building, losses, reconstruction errors and prediction."""

import numpy as np
from tensorflow.keras import Model
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


def predict_autoencoder(model, X_scaled, threshold=None):
    errors = reconstruction_errors(model, X_scaled)

    if threshold is None:
        y_pred = np.zeros(len(errors), dtype=int)
        prediction_mode = "native_autoencoder"
    else:
        y_pred = (errors > threshold).astype(int)
        prediction_mode = "percentile_threshold"

    native_pred = y_pred.copy()
    return y_pred, errors, native_pred, prediction_mode


def predict_autoencoder_by_activity(
    model,
    X,
    activity_codes,
    thresholds_by_activity,
    loss_name_by_activity=None,
    huber_delta_by_activity=None,
):
    if loss_name_by_activity is None:
        errors = reconstruction_errors(model, X)
    else:
        errors = reconstruction_errors_by_activity(
            model=model,
            X=X,
            activity_codes=activity_codes,
            loss_name_by_activity=loss_name_by_activity,
            huber_delta_by_activity=huber_delta_by_activity,
        )

    activity_codes = np.asarray(activity_codes)
    row_thresholds = np.array([thresholds_by_activity[a] for a in activity_codes], dtype=float)
    y_pred = (errors > row_thresholds).astype(int)
    native_pred = y_pred.copy()
    return y_pred, errors, native_pred, row_thresholds


def reconstruction_errors(model: Model, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X_recon = model(X, training=False).numpy()
    return np.mean(np.square(X - X_recon), axis=1)


def reconstruction_errors_by_activity(
    model: Model,
    X: np.ndarray,
    activity_codes,
    loss_name_by_activity: dict,
    huber_delta_by_activity: dict | None = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X_recon = model(X, training=False).numpy()

    activity_codes = np.asarray(activity_codes)
    huber_delta_by_activity = huber_delta_by_activity or {}

    scores = np.zeros(len(X), dtype=np.float32)

    for i, activity_code in enumerate(activity_codes):
        loss_name = str(loss_name_by_activity[activity_code]).lower()
        huber_delta = huber_delta_by_activity.get(activity_code, 1.0)

        x_true = X[i:i+1]
        x_pred = X_recon[i:i+1]

        if loss_name == "mse":
            scores[i] = float(np.mean(np.square(x_true - x_pred)))
        elif loss_name == "mae":
            scores[i] = float(np.mean(np.abs(x_true - x_pred)))
        elif loss_name == "huber":
            err = x_true - x_pred
            abs_err = np.abs(err)
            delta = float(huber_delta if huber_delta is not None else 1.0)
            huber = np.where(
                abs_err <= delta,
                0.5 * np.square(err),
                delta * (abs_err - 0.5 * delta),
            )
            scores[i] = float(np.mean(huber))
        else:
            raise ValueError(f"Unsupported loss_name for activity {activity_code}: {loss_name}")

    return scores


def set_seed(seed: int):
    """Make autoencoder training repeatable.

    Seeds Python, NumPy and TensorFlow, and switches TensorFlow to deterministic
    operations so two runs with the same seed give the same weights. Without the
    second step, TensorFlow's multi-threaded maths can add numbers up in a
    different order on each run, so results drift slightly between runs.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()


def get_ae_loss(config: dict):
    loss_name = str(config.get("loss_name", "mse")).lower()

    if loss_name == "mse":
        return "mse"
    if loss_name == "mae":
        return "mae"
    if loss_name == "huber":
        return Huber(delta=float(config.get("huber_delta", 1.0)))

    raise ValueError(f"Unsupported loss_name: {loss_name}")


def rowwise_reconstruction_loss(x_true, x_pred, loss_name: str, huber_delta: float = 1.0):
    loss_name = str(loss_name).lower()

    if loss_name == "mse":
        return tf.reduce_mean(tf.square(x_true - x_pred), axis=1)

    if loss_name == "mae":
        return tf.reduce_mean(tf.abs(x_true - x_pred), axis=1)

    if loss_name == "huber":
        huber = Huber(delta=float(huber_delta), reduction=tf.keras.losses.Reduction.NONE)
        return huber(x_true, x_pred)

    raise ValueError(f"Unsupported loss_name: {loss_name}")


def get_activity_col_map(cols):
    mapping = {}
    for i, col in enumerate(cols):
        if col.startswith("activity_"):
            activity_code = col.replace("activity_", "")
            mapping[activity_code] = i
    return mapping


class ActivityAwareAutoencoder(Model):
    def __init__(self, base_model, activity_loss_config, activity_col_map):
        super().__init__()
        self.base_model = base_model
        self.activity_loss_config = activity_loss_config
        self.activity_col_map = activity_col_map

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def _compute_activity_aware_loss(self, x, x_pred):
        batch_losses = []

        for activity_code, col_idx in self.activity_col_map.items():
            mask = tf.cast(tf.equal(x[:, col_idx], 1.0), tf.float32)

            loss_name = self.activity_loss_config[activity_code]["loss_name"]
            huber_delta = self.activity_loss_config[activity_code].get("huber_delta", 1.0)

            row_loss = rowwise_reconstruction_loss(
                x_true=x,
                x_pred=x_pred,
                loss_name=loss_name,
                huber_delta=huber_delta,
            )

            batch_losses.append(row_loss * mask)

        total_row_loss = tf.add_n(batch_losses)
        return tf.reduce_mean(total_row_loss)

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data

        with tf.GradientTape() as tape:
            x_pred = self.base_model(x, training=True)
            loss = self._compute_activity_aware_loss(x, x_pred)

        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        x_pred = self.base_model(x, training=False)
        loss = self._compute_activity_aware_loss(x, x_pred)
        return {"loss": loss}


def build_autoencoder(input_dim: int, config: dict, activity_col_map=None) -> Model:
    inputs = Input(shape=(input_dim,), name="input")
    x = Dense(config["hidden_dim_1"], activation="relu", name="enc_dense_1")(inputs)
    x = Dropout(config["dropout_rate"], name="enc_dropout_1")(x)
    x = Dense(config["hidden_dim_2"], activation="relu", name="bottleneck")(x)
    x = Dense(config["hidden_dim_1"], activation="relu", name="dec_dense_1")(x)
    outputs = Dense(input_dim, activation="linear", name="reconstruction")(x)

    base_model = Model(inputs=inputs, outputs=outputs, name="standard_autoencoder")

    if "loss_name_by_activity" in config:
        if activity_col_map is None:
            raise ValueError("activity_col_map is required when using loss_name_by_activity")

        missing_in_cols = [
            act for act in config["loss_name_by_activity"].keys()
            if act not in activity_col_map
        ]
        if missing_in_cols:
            raise ValueError(f"Missing activity columns for activities: {missing_in_cols}")

        activity_loss_config = {
            act: {
                "loss_name": str(config["loss_name_by_activity"][act]).lower(),
                "huber_delta": (
                    config.get("huber_delta_by_activity", {}).get(act, 1.0)
                    if str(config["loss_name_by_activity"][act]).lower() == "huber"
                    else None
                ),
            }
            for act in config["loss_name_by_activity"]
        }

        model = ActivityAwareAutoencoder(
            base_model=base_model,
            activity_loss_config=activity_loss_config,
            activity_col_map=activity_col_map,
        )
        model.compile(optimizer=Adam(learning_rate=config["learning_rate"]))
        return model

    model = base_model
    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss=get_ae_loss(config),
    )
    return model


def config_to_tag(config_id: int, config: dict) -> str:
    if "threshold_percentiles_by_activity" in config and config["threshold_percentiles_by_activity"] is not None:
        thr_map = config["threshold_percentiles_by_activity"]
        thr_str = "__".join(
            f"{k}-{str(v).replace('.', 'p')}" for k, v in sorted(thr_map.items())
        )
    else:
        thr_val = config.get("threshold_percentile")
        thr_str = "none" if thr_val is None else str(thr_val).replace(".", "p")

    if "loss_name_by_activity" in config and config["loss_name_by_activity"] is not None:
        loss_map = config["loss_name_by_activity"]
        loss_tag = "__".join(
            f"{k}-{loss_map[k]}" for k in sorted(loss_map)
        )
    else:
        loss_str = str(config.get("loss_name", "mse")).lower()
        huber_delta = config.get("huber_delta", None)

        if loss_str == "huber":
            loss_tag = f"huber_{str(huber_delta).replace('.', 'p')}"
        else:
            loss_tag = loss_str

    return (
        f"cfg_{config_id:03d}"
        f"__h1_{config['hidden_dim_1']}"
        f"__h2_{config['hidden_dim_2']}"
        f"__drop_{str(config['dropout_rate']).replace('.', 'p')}"
        f"__lr_{str(config['learning_rate']).replace('.', 'p')}"
        f"__bs_{config['batch_size']}"
        f"__loss_{loss_tag}"
        f"__thr_{thr_str}"
    )

