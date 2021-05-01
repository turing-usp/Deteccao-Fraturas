from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

import fracture_detection.datasets.mura  # NOQA

AUTOTUNE = tf.data.AUTOTUNE


def _prepare_ds(
    ds: tf.data.Dataset,
    img_shape: Tuple[Optional[int], Optional[int], Optional[int]],
    batch_size: int = 8,
):
    def prepare_img(image, label):
        size = list(img_shape)[:2]
        return tf.image.resize(image, size), label

    return (
        ds.map(prepare_img, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )


def _train_calbacks(
    checkpoint_path: Path,
    save_freq: int = 5,
):
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    save_model_period_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path / "{epoch:04d}.h5",
        verbose=1,
        period=save_freq,
    )
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path / "final.h5",
    )
    save_log_calback = tf.keras.callbacks.CSVLogger(
        filename=checkpoint_path / "train.log"
    )
    early_stop_calback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", min_delta=0.005, patience=3
    )

    return [
        save_model_period_callback,
        save_model_callback,
        save_log_calback,
        early_stop_calback,
    ]


def train(
    model: tf.keras.Model,
    img_shape: Tuple[Optional[int], Optional[int], Optional[int]],
    checkpoint_path: Path,
    epochs: int = 20,
    batch_size: int = 8,
    save_freq: int = 5,
):
    (ds_train, ds_valid) = tfds.load(  # NOQA (false positve)
        "mura",
        split=["train", "valid"],
        shuffle_files=True,
        as_supervised=True,
    )

    ds_train = _prepare_ds(ds_train, img_shape=img_shape, batch_size=batch_size)
    ds_valid = _prepare_ds(ds_valid, img_shape=img_shape, batch_size=batch_size)

    callbacks = _train_calbacks(checkpoint_path, save_freq)

    return model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_valid,
        callbacks=callbacks,
    )
