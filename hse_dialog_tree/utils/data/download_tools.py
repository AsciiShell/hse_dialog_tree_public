import os
import shutil

from configargparse import ArgumentParser
from tqdm.auto import tqdm

from hse_dialog_tree.utils.files import get_content, write_file

URL_PREFIX = 'https://dracor.org/api/corpora'
DATA_PATH = '{dir}/{corpus}/{name}/{doc}'

parser = ArgumentParser()
group = parser.add_argument_group('Download Options')
group.add_argument('--data-root', type=str, default='data', help='Data root folder')
group.add_argument('--override', type=bool, default=False, help='Override existing files')


def get_all_corpus():
    return [x['name'] for x in get_content(URL_PREFIX)]


def download_corpus(directory, corpus, override=False):
    for row in tqdm(get_content('{url}/{corpus}'.format(url=URL_PREFIX, corpus=corpus))['dramas']):
        name = row['name']
        if not override and os.path.exists(
                DATA_PATH.format(dir=directory, corpus=corpus, name=name, doc='spoken-text.txt')):
            continue
        try:
            write_file(
                DATA_PATH.format(dir=directory, corpus=corpus, name=name, doc='meta.json'),
                get_content('{url}/{corpus}/play/{name}'.format(url=URL_PREFIX, corpus=corpus, name=name)))
            write_file(
                DATA_PATH.format(dir=directory, corpus=corpus, name=name, doc='tei.xml'),
                get_content('{url}/{corpus}/play/{name}/tei'.format(url=URL_PREFIX, corpus=corpus, name=name),
                            decode=False))
            write_file(
                DATA_PATH.format(dir=directory, corpus=corpus, name=name, doc='spoken-text-by-character.json'),
                get_content('{url}/{corpus}/play/{name}/spoken-text-by-character'.format(url=URL_PREFIX, corpus=corpus,
                                                                                         name=name)))
            write_file(
                DATA_PATH.format(dir=directory, corpus=corpus, name=name, doc='spoken-text.txt'),
                get_content('{url}/{corpus}/play/{name}/spoken-text'.format(url=URL_PREFIX, corpus=corpus, name=name),
                            decode=False))
        except Exception as e:  # pylint: disable=broad-except
            shutil.rmtree(DATA_PATH.format(dir=directory, corpus=corpus, name=name, doc=''), ignore_errors=True)
            print(e)


def main():
    args = parser.parse_args()
    for corpus in tqdm(get_all_corpus()):
        download_corpus(args.data_root, corpus, args.override)


if __name__ == '__main__':
    main()
