"""
A collection of handy utilities
"""

from allennlp.commands.make_vocab import make_vocab_from_params
import codecs

from typing import List, Tuple, Dict

import os
import logging

from allennlp.common import Params
from allennlp.common.params import with_fallback

from io_.dataset_readers.load_dataset_reader import create_dataset_reader_params
from torch import Tensor

logger = logging.getLogger('mylog')

def get_ud_treebank_files(dataset_dir: str, treebanks: List[str] = []) -> Dict[str, Tuple[str, str, str]]:
    """
    Retrieves all treebank data paths in the given directory.
    :param dataset_dir: the directory where all treebank directories are stored
    :param treebanks: if not None or empty, retrieve just the subset of treebanks listed here
    :return: a dictionary mapping a treebank name to a list of train, dev, and test conllu files
    """
    datasets = {}
    treebanks = os.listdir(dataset_dir) if not treebanks else treebanks
    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]

        train_file = [file for file in conllu_files if file.endswith("train.conllu")]
        dev_file = [file for file in conllu_files if file.endswith("dev.conllu")]
        test_file = [file for file in conllu_files if file.endswith("test.conllu")]

        train_file = os.path.join(treebank_path, train_file[0]) if train_file else None
        dev_file = os.path.join(treebank_path, dev_file[0]) if dev_file else None
        test_file = os.path.join(treebank_path, test_file[0]) if test_file else None

        datasets[treebank] = (train_file, dev_file, test_file)
    return datasets

def get_ud_treebank_files_by_language(dataset_dir: str, languages: List[str] = []) -> Dict[str, Tuple[str, str, str]]:
    if languages:
        languages = [language.lower() for language in languages]
    treebanks = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f)) and f != 'vocab']
    datasets_by_language = {}
    for treebank in treebanks:
        if 'ud' in dataset_dir:
            language = treebank[3:].split('-')[0].lower()
        else:
            language = treebank
        if languages:
            if language not in languages:
                continue
        if language not in datasets_by_language:
            datasets_by_language[language] = {}
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]

        train_file = [file for file in conllu_files if file.endswith("train.conllu")]
        dev_file = [file for file in conllu_files if file.endswith("dev.conllu")]
        test_file = [file for file in conllu_files if file.endswith("test.conllu")]

        train_file = os.path.join(treebank_path, train_file[0]) if train_file else None
        dev_file = os.path.join(treebank_path, dev_file[0]) if dev_file else None
        test_file = os.path.join(treebank_path, test_file[0]) if test_file else None

        datasets_by_language[language][treebank] = (train_file, dev_file, test_file)
    return datasets_by_language

def merge_configs(configs: List[Params]) -> Params:
    """
    Merges a list of configurations together, with items with duplicate keys closer to the front of the list
    overriding any keys of items closer to the rear.
    :param configs: a list of AllenNLP Params
    :return: a single merged Params object
    """
    while len(configs) > 1:
        overrides, config = configs[-2:]
        configs = configs[:-2]

        configs.append(Params(with_fallback(preferred=overrides.params, fallback=config.params)))

    return configs[0]

def cache_vocab(args):
    """
    Caches the vocabulary given in the Params to the filesystem. Useful for large datasets that are run repeatedly.
    :param params: the AllenNLP Params
    :param vocab_config_path: an optional config path for constructing the vocab
    """

    params = {
        "vocabulary": {
            "directory_path": args.vocab_path,
            "non_padded_namespaces": ["upos", "xpos", "feats", "lemmas", "ner", "*tags", "*labels"],
            "tokens_to_add": {
                "upos": ["@@UNKNOWN@@"],
                "xpos": ["@@UNKNOWN@@"],
                "feats": ["@@UNKNOWN@@"],
                "lemmas": ["@@UNKNOWN@@"],
                "head_tags": ["@@UNKNOWN@@"]
            }
        }
    }
    if args.dataset == 'ud':
        params["train_data_path"] = "data/ud/multilingual/train.conllu"
        params["validation_data_path"] = "data/ud/multilingual/dev.conllu"
        params["test_data_path"] = "data/ud/multilingual/test.conllu"
    elif args.dataset == 'ontonotes':
        params["train_data_path"] = "data/ontonotes/all/train.conllu"
        params["validation_data_path"] = "data/ontonotes/all/dev.conllu"
        params["test_data_path"] = "data/ontonotes/all/test.conllu"
    else:
        raise ValueError
    dataset_reader_params = {
        "dataset_reader": create_dataset_reader_params(args)
    }
    params.update(dataset_reader_params)
    params = Params(params)

    if "vocabulary" not in params or "directory_path" not in params["vocabulary"]:
        return

    vocab_path = params["vocabulary"]["directory_path"]

    if os.path.exists(vocab_path):
        if os.listdir(vocab_path):
            return

        # Remove empty vocabulary directory to make AllenNLP happy
        try:
            os.rmdir(vocab_path)
        except OSError:
            pass

    params["vocabulary"].pop("directory_path", None)
    make_vocab_from_params(params, os.path.split(vocab_path)[0])

def clean_batch(args, batch, mode='train'):
    keys_to_keep = ['tokens', 'upos'] + [task for task in args.tasks]
    if 'deps' in args.tasks:
        keys_to_keep += ['head_tags', 'head_indices']
        keys_to_keep.remove('deps')
    batch_keys_to_remove = []
    for k, v in batch.items():
        if k not in keys_to_keep:
            if mode == 'train':
                batch_keys_to_remove.append(k)
            continue
        if isinstance(v, dict):
            for k_, v_ in v.items():
                if isinstance(v_, Tensor):
                    batch[k][k_] = v_.to(args.device)
        elif isinstance(v, Tensor):
            batch[k] = v.to(args.device)
    if mode == 'train':
        for k in batch_keys_to_remove:
            # For the lemmas task we need to keep the metadata of the words
            if k == 'metadata' and 'lemmas' in keys_to_keep:
                for d in batch['metadata']:
                    metadata_keys_to_remove = [x for x in d.keys() if x != 'words']
                    for k_ in metadata_keys_to_remove:
                        d.pop(k_, None)
            else:
                batch.pop(k, None)
    return batch
