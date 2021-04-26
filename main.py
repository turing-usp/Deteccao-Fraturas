from pathlib import Path

import tensorflow as tf

from fracture_detection.train import generate_model, train

IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

if __name__ == "__main__":
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )

    model = generate_model(base_model,
                           img_shape=IMG_SHAPE,
                           freeze=50)
    model.summary()

    base_model.trainable = True

    history = train(model,
                    epochs=5,
                    img_shape=IMG_SHAPE,
                    batch_size=8,
                    save_freq=5,
                    checkpoint_path=Path("./train_saves/mobilenetv2"))
