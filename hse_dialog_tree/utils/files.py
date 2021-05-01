import gzip
import json
import os
import pickle
import zipfile

import requests


def get_content(url, decode=True):
    r = requests.get(url)
    if r.ok:
        if decode:
            return json.loads(r.text)
        return r.text
    raise Exception(url)


def write_file(path, content):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    with open(path, 'wt') as f:
        if isinstance(content, (dict, list)):
            f.write(json.dumps(content, ensure_ascii=False))
        else:
            f.write(content)


HIGHEST_PROTOCOL = 4  # Google colab use python 3.6


def dump_pickle(obj, fname: str, *, protocol: int = HIGHEST_PROTOCOL):
    with zipfile.ZipFile(fname, 'w', compression=zipfile.ZIP_DEFLATED) as zfile:
        with zfile.open('data.pkl', 'w', force_zip64=True) as f:
            pickle.dump(obj, f, protocol=protocol)


def load_pickle(fname: str):
    with zipfile.ZipFile(fname, 'r') as zfile:
        with zfile.open('data.pkl', 'r') as f:
            return pickle.load(f)


def dump_gzip_text(texts, fname: str, *, compresslevel: int = 6):
    assert all([t.count('\n') == 0 for t in texts])
    with gzip.open(fname, 'wt', encoding='utf8', compresslevel=compresslevel) as gzfile:
        gzfile.write('\n'.join(texts))


def load_gzip_text(fname: str):
    with gzip.open(fname, 'rt', encoding='utf8') as gzfile:
        return gzfile.read().splitlines()


def load_gzip_text_iter(fname: str):
    with gzip.open(fname, 'rt', encoding='utf8') as gzfile:
        for line in gzfile:
            yield line.strip()
