from pathlib import Path

import tensorflow as tf
import mlflow

from fracture_detection.model.train import train
from fracture_detection.model.generate import generate_model

IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

tf.random.set_seed(0x0404)  # meu aniversario :)

mlflow.set_tracking_uri("http://mlflow.grupoturing.com.br")
mlflow.set_experiment("fraturing")

if __name__ == "__main__":
    base_model = tf.keras.applications.VGG19(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )

    mlflow.log_param("ola", "Mundo")
    model = generate_model(
        base_model,
        img_shape=IMG_SHAPE,
        preprocess_input=tf.keras.applications.vgg19.preprocess_input,
        freeze=0.75,
    )
    model.summary()

    base_model.trainable = True

    history = train(
        model,
        epochs=50,
        img_shape=IMG_SHAPE,
        batch_size=32,
        save_freq=5,
        checkpoint_path=Path("./train_saves/vgg19_75p"),
    )

    mlflow.log_param("epochs", 50)
    mlflow.log_metric("accuracy", history["accuracy"])
