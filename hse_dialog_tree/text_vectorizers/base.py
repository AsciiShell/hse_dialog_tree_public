import abc
import typing

import numpy as np


class TextVectorizerBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def vectorize(self, text: typing.Iterable[str]) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def model(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.vectorize(*args, **kwargs)
