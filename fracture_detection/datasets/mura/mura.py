"""MURA dataset."""
from pathlib import Path
from typing import Dict, Iterator, Tuple, Union
import tensorflow_datasets as tfds


_DESCRIPTION = """
MURA (musculoskeletal radiographs) is a large dataset of bone X-rays.
Algorithms are tasked with determining whether an X-ray study is normal
or abnormal.

Musculoskeletal conditions affect more than 1.7 billion people worldwide,
and are the most common cause of severe, long-term pain and disability,
with 30 million emergency department visits annually and increasing.
We hope that our dataset can lead to significant advances in medical
imaging technologies which can diagnose at the level of experts,
towards improving healthcare access in parts of the world where access
to skilled radiologists is limited.

MURA is one of the largest public radiographic image datasets.
We're making this dataset available to the community and hosting a competition
to see if your models can perform as well as radiologists on the task.
"""

_CITATION = """
@misc{rajpurkar2018mura,
      title={MURA: Large Dataset for
             Abnormality Detection in Musculoskeletal Radiographs},
      author={Pranav Rajpurkar and Jeremy Irvin and Aarti Bagul and Daisy Ding
              and Tony Duan and Hershel Mehta and Brandon Yang and Kaylie Zhu
              and Dillon Laird and Robyn L. Ball and Curtis Langlotz and
              Katie Shpanskaya and Matthew P. Lungren and Andrew Y. Ng},
      year={2018},
      eprint={1712.06957},
      archivePrefix={arXiv},
      primaryClass={physics.med-ph}
}
"""

_KAGGLE_DATA = "cjinny/mura-v11"


class Mura(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for MURA dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the MURA metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(shape=(256, None, 3)),
                    "label": tfds.features.ClassLabel(names=["normal", "abnormal"]),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://stanfordmlgroup.github.io/competitions/mura/",
            citation=_CITATION,
        )

    @staticmethod
    def _split_generators(
        dl_manager: tfds.download.DownloadManager,
    ) -> Dict[str, Iterator[Tuple[str, Dict[str, Union[Path, str]]]]]:
        """Returns SplitGenerators."""

        path = dl_manager.download_kaggle_data(_KAGGLE_DATA)
        path /= "MURA-v1.1"

        return {
            "train": Mura._generate_examples(path / "train"),
            "valid": Mura._generate_examples(path / "valid"),
        }

    @staticmethod
    def _generate_examples(
        path: tfds.core.utils.gpath.PosixGPath,
    ) -> Iterator[Tuple[str, Dict[str, Union[Path, str]]]]:
        """Yields examples."""

        path = Path(path)
        for f in path.glob("**/*.png"):
            yield f.as_uri(), {
                "image": f,
                "label": "normal" if "positive" in f.as_uri() else "abnormal",
            }
