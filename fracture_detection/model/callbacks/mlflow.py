from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import tensorflow as tf


class MLflowCallback(tf.keras.callbacks.Callback):
    params_log: Optional[Dict[str, Any]]

    def __init__(self, experiment_name: str, params: Dict[str, Any] = None):
        super(MLflowCallback, self).__init__()
        self.params_log = params

        mlflow.set_tracking_uri(  # noqa: false positive
            "http://mlflow.grupoturing.com.br"
        )
        mlflow.set_experiment(experiment_name)  # noqa: false positive

    def on_epoch_end(self, epoch, logs=None):
        with TemporaryDirectory() as tmpdirname:
            filepath = Path(tmpdirname.name) / "model.h5"
            self.model.save(filepath, overwrite=True)

            with mlflow.start_run():
                mlflow.log_param("epoch", epoch)
                mlflow.log_params(self.params_log or {})
                mlflow.log_metrics(logs or {})
                mlflow.log_artifact(filepath)
