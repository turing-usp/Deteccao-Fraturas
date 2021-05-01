import tensorflow as tf
from typing import Optional, Tuple, Union, Callable

_data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])


def _freeze_model(
    model: tf.keras.Model,
    freeze: Union[bool, int, float] = False,
):
    # Obs:
    # When you set layer.trainable = False, the BatchNormalization layer will
    # run in inference mode, and will not update its mean and variance statistics
    # https://www.tensorflow.org/tutorials/images/transfer_learning#important_note_about_batchnormalization_layers

    if isinstance(freeze, int):
        freeze_len = freeze
    elif isinstance(freeze, float):
        freeze_len = int(freeze*len(model.layers))
    else:  # isinstance(freeze, bool):
        if freeze:
            freeze_len = len(model.layers)
        else:
            freeze_len = 0

    if freeze_len != len(model.layers):
        model.trainable = True

        for layer in model.layers[:freeze_len]:
            layer.trainable = False


def generate_model(
    base_model: tf.keras.Model,
    img_shape: Tuple[Optional[int], Optional[int], Optional[int]],
    freeze: Union[bool, int, float] = False,
    preprocess_input: Optional[Callable] = None,
    use_data_augmentation: bool = True,
):
    inputs = tf.keras.layers.Input(shape=img_shape)
    if use_data_augmentation:
        x = _data_augmentation(inputs)
    if preprocess_input is not None:
        x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    _freeze_model(base_model, freeze)

    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model
