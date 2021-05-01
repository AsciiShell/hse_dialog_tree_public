import logging
import os
import typing

from bs4 import BeautifulSoup, element
from configargparse import ArgumentParser
from tqdm.auto import tqdm

from hse_dialog_tree.utils.files import dump_pickle

parser = ArgumentParser()
group = parser.add_argument_group('TEI file convert Options')
group.add_argument('--data-root', type=str, default='data', help='Data root folder')


def parse_scene(tree: element.Tag) -> typing.List[typing.Tuple[str, str]]:
    result = []
    for speech in tree.find_all('sp'):
        speaker = speech.find('speaker')
        if speaker is None:
            continue
        speaker = speaker.get_text()

        text = speech.find_all(['p', 'l'])
        if len(text) == 0:
            logging.debug(speech)
        text = ' '.join([x.get_text() for x in text])

        result.append((speaker, text))
    return result


def parse_text(doc: element.Tag) -> typing.List[typing.List[typing.Tuple[str, str]]]:
    result = []
    for scene in doc.find_all('div'):
        result.append(parse_scene(scene))
    return result


def parse_document(filename: str) -> typing.List[typing.List[typing.Tuple[str, str]]]:
    with open(filename, 'r') as tei:
        soup = BeautifulSoup(tei, 'lxml')
    return parse_text(soup)


def main():
    args = parser.parse_args()
    root = args.data_root
    for lang in os.listdir(root):
        all_data = {}
        all_data_path = os.path.join(root, lang, 'content_all.pkl.zip')
        lang_dir = os.path.join(root, lang)
        if os.path.exists(all_data_path) or not os.path.isdir(lang_dir):
            continue
        for book in tqdm(os.listdir(lang_dir)):
            path_book = os.path.join(lang_dir, book)
            data = parse_document(os.path.join(path_book, 'tei.xml'))
            dump_pickle(data, os.path.join(path_book, 'content.pkl.zip'))
            all_data[book] = data
        dump_pickle(all_data, all_data_path)


if __name__ == '__main__':
    main()
