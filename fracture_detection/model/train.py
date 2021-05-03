from typing import Optional, Tuple, Any, Dict

import tensorflow as tf
import tensorflow_datasets as tfds

import fracture_detection.datasets.mura  # NOQA
from fracture_detection.model.callbacks.mlflow import MLflowCallback

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


def train(
    model: tf.keras.Model,
    img_shape: Tuple[Optional[int], Optional[int], Optional[int]],
    epochs: int = 20,
    batch_size: int = 8,
    params_log: Optional[Dict[str, Any]] = None,
):
    (ds_train, ds_valid) = tfds.load(  # NOQA (false positve)
        "mura",
        split=["train", "valid"],
        shuffle_files=True,
        as_supervised=True,
    )

    ds_train = _prepare_ds(ds_train, img_shape=img_shape, batch_size=batch_size)
    ds_valid = _prepare_ds(ds_valid, img_shape=img_shape, batch_size=batch_size)

    params_log = params_log or {}
    params_log["batch_size"] = batch_size

    return model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_valid,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", min_delta=0.005, patience=3
            ),
            MLflowCallback(
                experiment_name="fraturing-test",
                params=params_log,
            ),
        ],
    )
