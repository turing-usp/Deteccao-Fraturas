import pickle

import tensorflow as tf

from fracture_detection.train import generate_model
from fracture_detection.train import train

IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

if __name__ == "__main__":
    base_model = tf.keras.applications.VGG19(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )
    model = generate_model(base_model, img_shape=IMG_SHAPE)

    model.summary()
    history = train(model, epochs=20, img_shape=IMG_SHAPE)

    with open('./train_history.pkl', 'wb') as f:
        pickle.dump(history, f)
