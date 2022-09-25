from allennlp.data import Vocabulary
import argparse
import json
import logging
import os
from torch import cuda, device, distributed
from torch.utils.data import DataLoader
from utils.training import set_seed, train

from io_.dataset_readers.convert_allenlp_reader_to_pytorch_dataset import AllennlpDataset
from io_.dataset_readers.universal_dependencies import UniversalDependenciesDatasetReader # To register "universal-dependencies-reader"
from io_.dataset_readers.load_dataset_reader import load_dataset_reader_
from io_.dataset_readers.prepare_splits_to_active_learning import prepare_splits, add_samples
from io_.io_utils import cache_vocab

logger = logging.getLogger('mylog')

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--src_domain", type=str, help="source domain or language name")
    parser.add_argument('--dataset', type=str, default='ud', help='dataset name')
    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input directory where the data is saved.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # AL parameters
    parser.add_argument("--al_selection_method", default='entropy_based_confidence', type=str, help="active learning selection method")
    parser.add_argument("--num_al_iterations", type=int, default=5,
                        help="Number of iterations for al process.")
    ## please set training_set_size or training_set_percentage, but not both.
    parser.add_argument("--training_set_size", default=None, type=int,
                        help="Specify the training set size you wish")
    parser.add_argument("--training_set_percentage", default=None, type=float,
                        help="The starting training percentage. A float between 0 to 1.")
    parser.add_argument("--load_checkpoint_from_last_iter", default=False,
                        action="store_true", help="loading checkpoint from last AL iteration")
    parser.add_argument("--tasks", default=['deps', 'ner'], type=str, nargs="+", help="Tasks to solve")
    parser.add_argument("--task_for_scoring", type=str, default=None,
                        help="tasks to be judged by in the al method (only applicable for entropy_based_confidence and dropout_agreement)")
    parser.add_argument("--task_levels", default=None, type=int, nargs="+",
                        help="output layer of each task. Make sure to insert task levels in the same order of the tasks.")
    parser.add_argument("--task_weights_for_loss", default=None, type=float, nargs="+",
                        help="tasks weights for loss function. Make sure to insert task weights in the same order of the tasks.")
    parser.add_argument("--task_weights_for_selection", default=None, type=float, nargs="+",
                        help="task weights for AL selection (appeared as beta in the original paper). Make sure to insert task weights in the same order of the tasks.")
    parser.add_argument("--load_sample_ids_dir", default=None, type=str,
                        help="loading sample ids files from directory. Only valid if al_selection_method is from_file.")

    ## Training and Data parameters
    parser.add_argument("--saved_config_path", default=None, type=str,
                        help="path for pretrained config name")
    parser.add_argument("--load_pretrained_model", default=None, type=str, help="path for pretrained model")
    parser.add_argument("--vocab_path", default="data/ud/vocab/multilingual/vocabulary", type=str,
                        help="The vocabulary path")
    parser.add_argument("--bert_vocab_path", default= "config/archive/bert-base-multilingual-cased/vocab.txt", type=str,
                        help="The bert vocabulary path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--do_eval", default=False, action="store_true",
                        help="Whether to run eval (without training).")
    parser.add_argument("--do_lowercase", default=False, action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--lazy", default=False, action="store_true", help="Lazy load the dataset")

    # Model parameters
    parser.add_argument("--pretrained_model", type=str, default="bert-base-multilingual-cased",
                        help="pretrained model")
    parser.add_argument("--multitask_model_type", type=str, default="complex", choices=['simple', 'complex'],
                        help="choose multitask model type: simple or complex (with shared and unshared modules)")
    parser.add_argument("--mix_embedding", default=False, action="store_true",
                        help="if True perform weighted average over bert layers")
    parser.add_argument("--unfreeze_bert", default=False, action="store_true",
                        help="if True unfreeze bert layers and fine-tune it")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="label smoothing value")
    parser.add_argument("--dropout", type=float, default=0.33,
                        help="dropout value")
    parser.add_argument("--bert_dropout", type=float, default=0.15,
                        help="bert dropout (for MLM) value")
    parser.add_argument("--layer_dropout", type=float, default=0.0,
                        help="bert layer dropout value")

    # Hyperparameters
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    # GPU Parameters
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--device", default="0", type=int, help="CUDA device; set to -1 for CPU")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    # Server parameters
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    # Parse arguments
    args = parser.parse_args()

    # Create directory (if not exists)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Load saved config (if True)
    if args.saved_config_path is not None:
        with open(args.saved_config_path, 'r') as f:
            args.__dict__ .update(json.load(f))

    # Save config to file
    if not args.do_eval:
        with open(os.path.join(args.output_dir, 'cfg.txt'), 'w') as f:
            json.dump(args.__dict__, f)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Verify arguments
    if args.task_levels is not None:
        assert len(args.tasks) == len(args.task_levels)
        assert all([isinstance(level, int) for level in args.task_levels])

    if args.task_weights_for_loss is not None:
        assert len(args.tasks) == len(args.task_weights_for_loss)
        assert all([weight >= 0 for weight in args.task_weights_for_loss])
        assert sum(args.task_weights_for_loss) > 0
        if sum(args.task_weights_for_loss) != 1:
            args.task_weights_for_loss = [weight / sum(args.task_weights_for_loss)
                                          for weight in args.task_weights_for_loss]
    if args.task_weights_for_selection is not None:
        assert len(args.tasks) == len(args.task_weights_for_selection)
        assert all([weight >= 0 for weight in args.task_weights_for_selection])
        assert sum(args.task_weights_for_selection) > 0
        if sum(args.task_weights_for_selection) != 1:
            args.task_weights_for_selection = [weight / sum(args.task_weights_for_selection)
                                               for weight in args.task_weights_for_selection]

    # Setup CUDA, GPU & distributed training
    if args.device == -1:
        args.no_cuda = True
    if args.local_rank == -1 or args.no_cuda:
        device_ = device("cuda" if cuda.is_available() and not args.no_cuda else "cpu")
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    else:
        cuda.set_device(args.local_rank)
        device_ = device("cuda", args.local_rank)
        distributed.init_process_group(backend="nccl")

    # Set number of GPUs and device
    args.n_gpu = cuda.device_count() if not args.no_cuda else 0
    args.device = device_

    # If we train a single-task model we should config task_for_scoring accordingly
    if len(args.tasks) == 1 and args.task_for_scoring is None:
        args.task_for_scoring = args.tasks[0]

    if len(args.tasks) > 1 and \
            args.al_selection_method in ['entropy_based_confidence', 'dropout_agreement'] and \
            args.task_for_scoring is None:
        raise ValueError('al_selection_method should be set in multi-task mode when using '
                         '"entropy_based_confidence" or "dropout_agreement" as selection methods.')

    # Setup logging
    logging.basicConfig(filename=os.path.join(args.output_dir, args.src_domain + '_log.txt'),
                        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    fh = logging.FileHandler(os.path.join(args.output_dir, args.src_domain + '_log.txt'))
    logger.setLevel(logging.getLevelName('DEBUG'))
    logger.addHandler(fh)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    if args.local_rank not in [-1, 0]:
        distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank == 0:
        distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Load data paths
    data_paths = {}
    data_splits = ['train', 'dev', 'test']
    for data_split in data_splits:
        data_path = os.path.join(args.data_dir, args.dataset, args.src_domain, data_split + '.conllu')
        if os.path.exists(data_path):
            data_paths[data_split] = data_path
            logger.info(data_split + " path: %s" % data_path)

    # Loading the vocabulary
    try:
        cache_vocab(args, data_paths)
    except:
        logger.warning("Could not cache vocab")
    vocab = Vocabulary.from_files(args.vocab_path)
    logger.info("Vocab path: %s\n" % args.vocab_path)

    # Create datasets
    datasets = {}
    reader = load_dataset_reader_(args)
    for data_split in data_paths.keys():
        datasets[data_split] = AllennlpDataset(vocab=vocab,
                                               reader=reader,
                                               dataset_path=data_paths[data_split])
    # We simulate initial small training set
    # by setting training_set_percentage or training_set_size
    orig_training_set_size = float(len(datasets['train']))
    if args.training_set_percentage is not None:
        datasets = prepare_splits(datasets=datasets, training_set_percentage=args.training_set_percentage)
    elif args.training_set_size is not None:
        prepare_splits(datasets=datasets, training_set_size=args.training_set_size)
        args.training_set_percentage = len(datasets['train']) / (orig_training_set_size + 1e-8)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    args.training_samples_to_add = len(datasets['train'])

    iterations = range(args.num_al_iterations - 1, args.num_al_iterations) \
        if args.do_eval else range(args.num_al_iterations)

    # Running iterative AL
    for al_iter_num in iterations:
        dataloaders = {}
        for data_split in datasets.keys():
            # Prepare dataloaders
            if data_split == 'train' and not args.do_eval:
                dataloaders[data_split] = DataLoader(datasets[data_split],
                                                  shuffle=True,
                                                  batch_size=args.train_batch_size,
                                                  collate_fn=datasets[data_split].allennlp_collocate)
            else:
                dataloaders[data_split] = DataLoader(datasets[data_split],
                                                  shuffle=False,
                                                  batch_size=args.eval_batch_size,
                                                  collate_fn=datasets[data_split].allennlp_collocate)
        # Run single training iteration
        sample_ids = train(args, dataloaders, vocab, al_iter_num)
        # Prepare data for next AL iteration
        if sample_ids:
            datasets = add_samples(datasets, sample_ids)

if __name__ == "__main__":
    main()
