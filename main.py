import tensorflow as tf
from fracture_detection.train import generate_model, train

IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

if __name__ == "__main__":
    # base_model = tf.keras.applications.VGG19(
    #     input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    # )
    base_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=IMG_SHAPE),
            tf.keras.layers.Dense(128, activation="relu"),
        ]
    )
    base_model.summary()
    model = generate_model(base_model, img_shape=IMG_SHAPE)

    model.summary()
    train(model, epochs=20, img_shape=IMG_SHAPE)
