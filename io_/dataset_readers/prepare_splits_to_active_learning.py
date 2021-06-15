from torch.utils.data import Dataset
from typing import Dict
import numpy as np

from io_ import AllennlpDataset


def prepare_splits(datasets: Dict[str, Dataset],
                   training_set_percentage=None,
                   training_set_size=None) -> Dict[str, Dataset]:
    train_samples_list = datasets['train'].iterable
    np.random.shuffle(train_samples_list)
    if training_set_percentage is not None:
        training_set_size = int(np.floor(len(train_samples_list) * training_set_percentage))
    elif training_set_size is None:
        raise ValueError('train_percentage and train_size must not contain both None')
    train_samples = train_samples_list[:training_set_size]
    datasets['train'] = AllennlpDataset(datasets['train'].vocab, train_samples)
    unlabeled_samples = train_samples_list[training_set_size:]
    datasets['unlabeled'] = AllennlpDataset(datasets['train'].vocab, unlabeled_samples)
    if 'dev' in datasets.keys():
        dev_samples_list = datasets['dev'].iterable
        dev_size = min(2 * training_set_size, len(dev_samples_list))
        dev_samples = dev_samples_list[:dev_size]
        datasets['dev'] = AllennlpDataset(datasets['dev'].vocab, dev_samples)
    return datasets

def add_samples(datasets: Dict[str, Dataset],
                sample_ids) -> Dict[str, Dataset]:
    if 'unlabeled' in datasets:
        instances_to_add = [x for x in datasets['unlabeled'].iterable if x['metadata']['sample_id'] in sample_ids]
        train_samples = datasets['train'].iterable + list(instances_to_add)
        datasets['train'] = AllennlpDataset(datasets['train'].vocab, train_samples)
        unlabeled_samples = [x for x in datasets['unlabeled'].iterable if x['metadata']['sample_id'] not in sample_ids]
        datasets['unlabeled'] = AllennlpDataset(datasets['train'].vocab, unlabeled_samples)
    return datasets
