import csv
import json
import math
import sys

import torch
from tqdm import tqdm
from allennlp.data import Vocabulary
import argparse
import datetime
import io_
import models
import logging
import os
from torch.utils.data import RandomSampler, DataLoader, DistributedSampler, SequentialSampler
import utils
from io_.dataset_readers.load_dataset_reader import load_dataset_reader_
from io_.dataset_readers.prepare_splits_to_active_learning import prepare_splits, add_samples
from io_.io_utils import cache_vocab

logger = logging.getLogger('mylog')

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--src_domain", type=str, help="source domain or language name")
    parser.add_argument("--tgt_domain", type=str, help="target domain or language name")
    parser.add_argument("--al_scoring", type=str, help="active learning scoring method")
    parser.add_argument('--dataset', type=str, default='ud', help='dataset name')
    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input directory where the data is saved.")
    parser.add_argument("--unlabeled_input_dir", default=None, type=str,
                        help="The input directory where the unlabeled data is saved.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--saved_config_path", default=None, type=str,
                        help="Pretrained config name")
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
    parser.add_argument("--train_unlabeled_data", default=False, action="store_true",
                        help="Whether to train on unlabeled data).")
    parser.add_argument("--train_labeled_and_unlabeled_data", default=False, action="store_true",
                        help="Whether to train on unlabeled data).")
    parser.add_argument("--do_eval", default=False, action="store_true",
                        help="Whether to run eval (without training).")
    parser.add_argument("--do_lowercase", default=False, action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
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

    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--dropout", type=float, default=0.33,
                        help="dropout value")
    parser.add_argument("--bert_dropout", type=float, default=0.15,
                        help="bert dropout (for MLM) value")
    parser.add_argument("--layer_dropout", type=float, default=0.0,
                        help="bert layer dropout value")
    parser.add_argument("--word_dropout", type=float, default=0.2,
                        help="word dropout value")

    parser.add_argument("--pretrained_model", type=str, default="bert-base-multilingual-cased",
                        help="pretrained model")
    parser.add_argument("--bert_combine_layers", type=str, default="last", choices=["last", "all"],
                        help="last: extract only bert's last layer. all: extract all bert's layers")
    parser.add_argument("--mix_embedding", default=False, action="store_true",
                        help="if True perform weighted average over bert layers")
    parser.add_argument("--unfreeze_bert", default=False, action="store_true",
                        help="if True unfreeze bert layers and fine-tune it")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--strict", action="store_true", default=False, help="If true, pretrained model is continuing to train")
    parser.add_argument("--tasks", default=[], type=str, nargs="+", help="Tasks to solve")
    parser.add_argument("--task_levels", default=[], type=int, nargs="+", help="Height of each task")
    parser.add_argument("--task_weights", default=None, type=float, nargs="+", help="the weights of the tasks")
    parser.add_argument("--training_set_percentage", default=None, type=float,
                        help="The starting training percentage. A float between 0 to 1.")
    parser.add_argument("--training_set_size", default=None, type=int,  help="Specify the training set size you wish")
    parser.add_argument("--pretrained_training_set_size", default=None, type=int,  help="Specify the pretrained training set size you wish")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--config", default=[], type=str, nargs="+", help="Overriding configuration files")
    parser.add_argument("--device", default="0", type=int, help="CUDA device; set to -1 for CPU")
    parser.add_argument("--resume", type=str, help="Resume training with the given model")
    parser.add_argument("--lazy", default=False, action="store_true", help="Lazy load the dataset")
    parser.add_argument("--load_pretrained_model", default=None, type=str, help="path for pretrained model")
    parser.add_argument("--cleanup_archive", action="store_true", help="Delete the model archive")
    parser.add_argument("--replace_vocab", action="store_true", help="Create a new vocab and replace the cached one")
    parser.add_argument("--archive_bert", action="store_true", help="Archives the finetuned BERT model after training")
    parser.add_argument("--predictor", default="udify_predictor", type=str, help="The type of predictor to use")
    parser.add_argument("--task_for_scoring", type=str, default=None, help="tasks to be judged by in the al method (not applicable for joint entropy and random)")


    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # if os.path.exists(args.output_dir) and os.listdir(
    #         args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
    #             args.output_dir))

    if args.train_labeled_and_unlabeled_data and args.train_unlabeled_data:
        raise ValueError('train_labeled_and_unlabeled_data and train_unlabeled_data cannot be set both to True')

    if args.saved_config_path is not None:
        ##Todo check if output_dir and domain are the same
        with open(args.saved_config_path, 'r') as f:
            args.__dict__ .update(json.load(f))

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

        # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda"
                              "" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        #args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    if len(args.tasks) == 1 and args.task_for_scoring is None:
        args.task_for_scoring = args.tasks[0]

    # Setup logging
    logging.basicConfig(filename=os.path.join(args.output_dir, args.tgt_domain + '_log.txt'),
                        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    fh = logging.FileHandler(os.path.join(args.output_dir, args.tgt_domain + '_log.txt'))
    logger.setLevel(logging.getLevelName('DEBUG'))
    logger.addHandler(fh)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    utils.training.set_seed(args)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.unlabeled_input_dir is None:
        args.unlabeled_input_dir = os.path.join(args.data_dir, args.dataset)

    data_paths = {}
    data_splits = ['train', 'dev', 'test', 'unlabeled']
    for data_split in data_splits:
        data_path = os.path.join(args.data_dir, args.dataset, args.src_domain, data_split + '.conllu')
        if os.path.exists(data_path):
            data_paths[data_split] = data_path

    for data_split, path in data_paths.items():
        logger.info(data_split + " path: %s" % path)
    try:
        cache_vocab(args)
    except:
        # logger.warning("Could not cache vocab")
        pass
    vocab = Vocabulary.from_files(args.vocab_path)
    logger.info("Vocab path: %s\n" % args.vocab_path)

    datasets = {}
    reader = load_dataset_reader_(args)
    for data_split in data_paths.keys():
        datasets[data_split] = io_.dataset_readers.convert_allenlp_reader_to_pytorch_dataset.AllennlpDataset(vocab=vocab,
                                                                                                          reader=reader,
                                                                                                          dataset_path=data_paths[data_split])
    orig_training_set_size = float(len(datasets['train']))
    if args.training_set_percentage is not None:
        datasets = prepare_splits(datasets=datasets, training_set_percentage=args.training_set_percentage)
    elif args.training_set_size is not None:
        prepare_splits(datasets=datasets, training_set_size=args.training_set_size)
        args.training_set_percentage = len(datasets['train']) / (orig_training_set_size + + 1e-8)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.training_samples_to_add = len(datasets['train'])

    if args.training_set_percentage is not None:
        num_iters = math.floor(1.0 / args.training_set_percentage)
    else:
        num_iters = 1
    for al_iter_num in range(num_iters):
        dataloaders = {}
        for data_split in datasets.keys():
            if data_split == 'train' and not args.do_eval:
                dataloaders[data_split] = DataLoader(datasets[data_split],
                                                  shuffle=True,
                                                  batch_size=args.train_batch_size,
                                                  collate_fn=datasets[data_split].allennlp_collocate)
            else:
                dataloaders[data_split] = DataLoader(datasets[data_split],
                                                  shuffle=False,
                                                  batch_size=args.train_batch_size,
                                                  collate_fn=datasets[data_split].allennlp_collocate)

        if args.al_scoring == 'entropy':
            sample_ids_file = os.path.join(args.output_dir,
                                           args.tgt_domain + '_sample_ids_entropy_' + args.task_for_scoring + '_iter_' + str(
                                               al_iter_num) + '.csv')
        elif args.al_scoring == 'joint_entropy':
            sample_ids_file = os.path.join(args.output_dir,
                                           args.tgt_domain + '_sample_ids_entropy_ner_deps_iter_' + str(
                                               al_iter_num) + '.csv')
        else:
            sample_ids_file = os.path.join(args.output_dir,
                                           args.tgt_domain + '_sample_ids_random_iter_' + str(al_iter_num) + '.csv')
        with open(sample_ids_file, newline='') as f:
            reader = csv.reader(f)
            sample_ids = list(reader)
        sample_ids = [int(row[0]) for row in sample_ids]
        sentences = []
        for batch in tqdm(dataloaders['unlabeled'], desc="Calculating Scores for Unlabeled samples"):
            with torch.no_grad():
                for cur_meta in batch['metadata']:
                    if cur_meta['sample_id'] in sample_ids:
                        sentence = ' '.join(cur_meta['words'])
                        sentences.append(sentence)


if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except Exception as e:
    #     logger.error("Exception occurred in main: %s" % str(e))
    #     print("Exception occurred in main: %s" % str(e))
    #     sys.exit(1)