import typing

import numpy as np
from tqdm import tqdm

from .files import load_gzip_text


class DistributedDict:
    """Распределенный словарь текст - вектор
    Подаем на вход список файлов с текстами и векторами
    Читаем в память все фразы, для работы с векторами используем memmap
    """

    # pylint: disable=R0913
    def __init__(self, files_text: typing.List[str], files_vectors: typing.List[str], use_load=True,
                 shape=512, dtype=np.float32, offset=0, progress=True):
        progress = tqdm if progress else iter
        self.text = {}
        self.vectors = []
        for i, (f_text, f_vector) in enumerate(zip(progress(files_text), files_vectors)):
            phrases = load_gzip_text(f_text)
            if use_load:
                vectors = np.load(f_vector, mmap_mode='r')
            else:
                vectors = np.memmap(f_vector, dtype, mode='r', offset=offset, shape=(len(phrases), shape))
            for j, phrase in enumerate(phrases):
                self.text[phrase] = i, j
            self.vectors.append(vectors)

    def __getitem__(self, text):
        doc, row = self.text[text]
        return self.vectors[doc][row]

    def __len__(self):
        return len(self.vectors)
