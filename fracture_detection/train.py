from os import PathLike
from typing import Optional, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
import fracture_detection.datasets.mura  # NOQA


AUTOTUNE = tf.data.AUTOTUNE


def generate_model(
    base_model: tf.keras.Model,
    img_shape: Tuple[Optional[int], Optional[int], Optional[int]],
    data_augmentation: bool = True,
):
    data_augmentation_layers = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(
        1.0 / 127.5, offset=-1
    )

    inputs = tf.keras.layers.Input(shape=img_shape)
    if data_augmentation:
        x = data_augmentation_layers(inputs)
    x = rescale(x)
    x = base_model(x, training=False)
    if len(x.shape) > 2:
        x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model


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
        # .cache()
        .batch(batch_size).prefetch(AUTOTUNE)
    )


def train(
    model: tf.keras.Model,
    img_shape: Tuple[Optional[int], Optional[int], Optional[int]],
    epochs: int = 20,
    batch_size: int = 8,
    checkpoint_path: PathLike = "./training",  # NOQA (tipagem de python ainda é uma desgraça)
):
    (ds_train, ds_valid) = tfds.load(  # NOQA (false positve)
        "mura",
        split=["train", "valid"],
        shuffle_files=True,
        as_supervised=True,
    )

    ds_train = _prepare_ds(ds_train, img_shape=img_shape, batch_size=batch_size)
    ds_valid = _prepare_ds(ds_valid, img_shape=img_shape, batch_size=batch_size)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1
    )

    return model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_valid,
        callbacks=[cp_callback],
    )
