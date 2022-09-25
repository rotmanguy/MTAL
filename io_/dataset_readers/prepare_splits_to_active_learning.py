from torch.utils.data import Dataset
from typing import Dict
import numpy as np

from io_.dataset_readers.convert_allenlp_reader_to_pytorch_dataset import AllennlpDataset


def prepare_splits(datasets: Dict[str, Dataset],
                   training_set_percentage=None,
                   training_set_size=None) -> Dict[str, Dataset]:
    # Preparing training samples
    train_samples_list = datasets['train'].iterable
    np.random.shuffle(train_samples_list)
    if training_set_percentage is not None:
        training_set_size = int(np.floor(len(train_samples_list) * training_set_percentage))
    elif training_set_size is None:
        raise ValueError('train_percentage and train_size must not contain both None')
    train_samples = train_samples_list[:training_set_size]
    datasets['train'] = AllennlpDataset(datasets['train'].vocab, train_samples)
    # Preparing unlabeled samples (those that we did not choose for training)
    unlabeled_samples = train_samples_list[training_set_size:]
    datasets['unlabeled'] = AllennlpDataset(datasets['train'].vocab, unlabeled_samples)
    # Preparing validation samples (we set them to be twice the initial training set size, if possible)
    if 'dev' in datasets.keys():
        dev_samples_list = datasets['dev'].iterable
        dev_size = min(2 * training_set_size, len(dev_samples_list))
        dev_samples = dev_samples_list[:dev_size]
        datasets['dev'] = AllennlpDataset(datasets['dev'].vocab, dev_samples)
    return datasets

def add_samples(datasets: Dict[str, Dataset], sample_ids) -> Dict[str, Dataset]:
    new_train_samples = [x for x in datasets['unlabeled'].iterable if x['metadata']['sample_id'] in sample_ids]
    unlabeled_samples = [x for x in datasets['unlabeled'].iterable if x['metadata']['sample_id'] not in sample_ids]
    # Add new train samples to current train samples
    new_train_samples = datasets['train'].iterable + new_train_samples
    # Update train samples
    datasets['train'] = AllennlpDataset(datasets['train'].vocab, new_train_samples)
    # Update unlabeled samples
    datasets['unlabeled'] = AllennlpDataset(datasets['train'].vocab, unlabeled_samples)
    return datasets
