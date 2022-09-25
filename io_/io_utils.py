"""
A collection of handy utilities
"""

from allennlp.commands.make_vocab import make_vocab_from_params

import os
import logging

from allennlp.common import Params

from io_.dataset_readers.load_dataset_reader import create_dataset_reader_params
from torch import Tensor

logger = logging.getLogger('mylog')

def cache_vocab(args, data_paths):
    """
    Caches the vocabulary given in the Params to the filesystem. Useful for large datasets that are run repeatedly.
    :param args: the config arguments.
    :param data_paths: the data paths.
    """

    params = {"vocabulary": {
        "directory_path": args.vocab_path,
        "non_padded_namespaces": ["upos", "xpos", "feats", "lemmas", "ner", "*tags", "*labels"],
        "tokens_to_add": {
            "upos": ["@@UNKNOWN@@"],
            "xpos": ["@@UNKNOWN@@"],
            "feats": ["@@UNKNOWN@@"],
            "lemmas": ["@@UNKNOWN@@"],
            "head_tags": ["@@UNKNOWN@@"]
        }
    }}
    if 'train' in data_paths:
        params["train_data_path"] = data_paths['train']
    if 'dev' in data_paths:
        params["validation_data_path"] = data_paths['dev']
    if 'test' in data_paths:
        params["test_data_path"] = data_paths['test']

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
    """
    Clean batch from unnecessary keys
    :param args: the config arguments.
    :param batch: the current batch.
    :param mode: 'train' or 'eval'.
    :return: cleaned batch.
    """
    # Set keys to keep
    keys_to_keep = ['tokens', 'upos'] + [task for task in args.tasks]
    # Set dependency parsing keys
    if 'deps' in args.tasks:
        keys_to_keep += ['head_tags', 'head_indices']
        keys_to_keep.remove('deps')
    # Construct keys to remove
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
    # Remove batch_keys_to_remove from batch
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
