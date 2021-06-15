import itertools
from typing import Union, List, Tuple

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Vocabulary
from torch.utils.data import Dataset
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance

class AllennlpDataset(Dataset):

    def __init__(self,
                 vocab: Vocabulary,
                 iterable: List=None,
                 reader: DatasetReader=None,
                 dataset_path: Union[str, List[str], Tuple[str]]=None
                 ):

        self.vocab = vocab
        self.reader = reader
        self.dataset_path = dataset_path
        if iterable is not None:
            self.iterable = iterable
        else:
            assert reader is not None and dataset_path is not None
            if type(dataset_path) in [list, tuple]:
                self.iterable = list(itertools.chain(*[self.reader.read(x) for x in dataset_path]))
            else:
                self.iterable = self.reader.read(dataset_path)
        self.iterator = (x for x in self.iterable)

        self._length = None

    def __len__(self):
        """
        This is gross but will go away in the next pytorch release,
        as they are introducing an `IterableDataset` class which doesn't
        need to have a length:
        https://pytorch.org/docs/master/data.html#torch.utils.data.IterableDataset
        """
        if self._length is None:
            self._length = 0
            for i in self.iterator:
                self._length += 1
            self.iterator = (x for x in self.iterable)
        return self._length

    def __getitem__(self, idx) -> Instance:
        get_next = next(self.iterator, None)
        if get_next is None:
            self.iterator = (x for x in self.iterable)
            get_next = next(self.iterator)
        return get_next

    # Function to tell the torch DataLoader how to batch up our custom data, i.e Instances
    def allennlp_collocate(self, batch):
        batch = Batch(batch)
        batch.index_instances(self.vocab)
        return batch.as_tensor_dict(batch.get_padding_lengths())