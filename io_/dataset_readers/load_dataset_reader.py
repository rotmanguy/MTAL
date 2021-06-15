from allennlp.common import Params
from allennlp.data import DatasetReader

def create_dataset_reader_params(args):
    dataset_reader_params = {
            "type": "universal-dependencies-reader",
            "tasks": args.tasks,
            "lazy": args.lazy,
            "token_indexers": {
                "tokens": {
                    "type": "single_id",
                    "lowercase_tokens": True
                },
                "bert": {
                    "type": "bert-pretrained-indexer",
                    "pretrained_model": args.pretrained_model,
                    "do_lowercase": args.do_lowercase,
                    "use_starting_offsets": True
                }
            }
        }
    return dataset_reader_params

def load_dataset_reader_(args):
    dataset_reader_params = create_dataset_reader_params(args)
    return DatasetReader.from_params(Params(dataset_reader_params))