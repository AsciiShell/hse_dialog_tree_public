import math


class Batch:
    def __init__(self, iterable, batch_size=1):
        self.iterable = iterable
        self.len = len(iterable)
        self.batch_size = batch_size

        self.iterable_len = math.ceil(self.len / self.batch_size)

    def __iter__(self):
        for ndx in range(0, self.len, self.batch_size):
            yield self.iterable[ndx:min(ndx + self.batch_size, self.len)]

    def __len__(self):
        return self.iterable_len


def batch_gen(iterable, batch_size=1):
    bucket = []
    for it in iterable:
        bucket.append(it)
        if len(bucket) >= batch_size:
            yield bucket
            bucket = []
    if len(bucket) > 0:
        yield bucket
