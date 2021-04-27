from pathlib import Path
from typing import Optional, Tuple, Union

import tensorflow as tf
import tensorflow_datasets as tfds

import fracture_detection.datasets.mura  # NOQA

AUTOTUNE = tf.data.AUTOTUNE


def generate_model(
    base_model: tf.keras.Model,
    img_shape: Tuple[Optional[int], Optional[int], Optional[int]],
    freeze: Union[bool, int, float] = False,
    data_augmentation: bool = True,
):
    if isinstance(freeze, int):
        freeze_len = freeze
    elif isinstance(freeze, float):
        freeze_len = int(freeze*len(base_model.layers))
    else:  # isinstance(freeze, bool):
        if freeze:
            freeze_len = len(base_model.layers)
        else:
            freeze_len = 0

    data_augmentation_layers = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(
        1.0 / 127.5, offset=-1
    )

    # When you set layer.trainable = False, the BatchNormalization layer will
    # run in inference mode, and will not update its mean and variance statistics
    # https://www.tensorflow.org/tutorials/images/transfer_learning#important_note_about_batchnormalization_layers

    inputs = tf.keras.layers.Input(shape=img_shape)
    if data_augmentation:
        x = data_augmentation_layers(inputs)
    x = rescale(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    if freeze_len != len(base_model.layers):
        model.trainable = True
        base_model.trainable = True

        for layer in base_model.layers[:freeze_len]:
            layer.trainable = False

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

    cpp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path / "{epoch:04d}.h5",
        verbose=1,
        period=save_freq,
    )
    cpf_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path / "final.h5",
        period=save_freq,
    )
    cl_calback = tf.keras.callbacks.CSVLogger(
        filename=checkpoint_path / "train.log"
    )
    es_calback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0.005, patience=3
    )
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    return model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_valid,
        callbacks=[cpp_callback, cpf_callback, cl_calback, es_calback],
    )
