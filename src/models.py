"""Model builders for tabular, vision, and text tasks."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def set_global_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_linear_regression_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ]
    )
    return Pipeline([("prep", preprocessor), ("model", LinearRegression())])


def build_logistic_baseline_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    max_iter: int = 1000,
) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def regression_metrics(y_true: Iterable, y_pred: Iterable) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: Iterable, y_pred: Iterable) -> dict[str, float | str]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, digits=3),
    }


def build_mlp_regressor(input_dim: int, hidden: tuple[int, ...] = (128, 64)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs, name="tabular_mlp_regressor")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_mlp_classifier(
    input_dim: int,
    num_classes: int,
    hidden: tuple[int, ...] = (128, 64),
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="tabular_mlp_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_small_cnn(num_classes: int, input_shape: tuple[int, int, int] = (128, 128, 3)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="small_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (128, 128, 3),
    backbone: str = "mobilenetv2",
    trainable_top: int = 20,
) -> tf.keras.Model:
    if backbone == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base.trainable = True
    if trainable_top > 0:
        for layer in base.layers[:-trainable_top]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    if backbone == "mobilenetv2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    else:
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    features = base(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"transfer_{backbone}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_embedding_text_classifier(
    vocab_size: int,
    num_classes: int,
    max_len: int = 128,
    embedding_dim: int = 128,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(max_len,))
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="embedding_text_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
