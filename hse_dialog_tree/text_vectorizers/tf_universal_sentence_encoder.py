import typing

import numpy as np
import tensorflow as tf
# Required for TfUniversalSentenceEncoder
# pylint: disable=unused-import
# noinspection PyUnresolvedReferences
import tensorflow_text  # noqa: F401

from .base import TextVectorizerBase

DEFAULT_PATH = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'


class TfUniversalSentenceEncoder(TextVectorizerBase):
    def __init__(self, path: str = DEFAULT_PATH):
        self._model = tf.saved_model.load(path)

    def vectorize(self, text: typing.Iterable[str]) -> np.ndarray:
        return self._model(text).numpy()

    @property
    def model(self):
        return self._model
