import logging
import random
import numpy as np
import torch
from torch.optim import LBFGS
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import Multitask_Model

logger = logging.getLogger('mylog')


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.cuda.manual_seed(args.seed)

def build_model(args, vocab, t_total, strict=False):
    model = Multitask_Model(vocab=vocab,
                            tasks=args.tasks,
                            task_levels=args.task_levels,
                            task_weights=args.task_weights,
                            dropout=args.dropout,
                            bert_dropout=args.bert_dropout,
                            layer_dropout=args.layer_dropout,
                            word_dropout=args.word_dropout,
                            mix_embedding=args.mix_embedding,
                            label_smoothing=args.label_smoothing,
                            unfreeze_bert=args.unfreeze_bert,
                            pretrained_model=args.pretrained_model,
                            bert_combine_layers=args.bert_combine_layers,
                            scale_temperature=args.scale_temperature)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    # Lower lr for BERT
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' not in n],
         "lr": args.learning_rate, "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n],
         "lr": 5e-5, "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' not in n],
         "lr": args.learning_rate, "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' in n],
         "lr": 5e-5, "weight_decay": 0.0}
    ]

    ## Equal lr
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n],
    #      "lr": args.learning_rate, "weight_decay": args.weight_decay},
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' in n],
    #     "lr": args.learning_rate, "weight_decay": 0.0},
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' not in n],
    #      "lr": args.learning_rate, "weight_decay": args.weight_decay},
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' not in n],
    #      "lr": args.learning_rate, "weight_decay": 0.0},
    # ]

    # # ## Hierarchical lr
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n and
    #                 any(nd in n for nd in ['layer.' + str(i) + '.' for i in range(0, 3)])],
    #      "lr": 5e-4, "weight_decay": 5e-2},
    #      {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n and
    #                  any(nd in n for nd in ['layer.' + str(i) + '.' for i in range(3, 6)])],
    #       "lr": 1e-4, "weight_decay": 1e-2},
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n and
    #                 any(nd in n for nd in ['layer.' + str(i) + '.' for i in range(6, 9)])],
    #      "lr": 5e-5, "weight_decay": 5e-3},
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n and
    #                 any(nd in n for nd in ['layer.' + str(i) + '.' for i in range(9, 12)])],
    #      "lr": 1e-5, "weight_decay": 1e-3},
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n and
    #                 not any(nd in n for nd in ['layer.' + str(i) + '.' for i in range(0, 12)])],
    #      "lr": args.learning_rate, "weight_decay": 0.0},
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' in n],
    #     "lr": args.learning_rate, "weight_decay": 0.0},
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' not in n],
    #      "lr": args.learning_rate, "weight_decay": args.weight_decay},
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' not in n],
    #      "lr": args.learning_rate, "weight_decay": 0.0},
    # ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
    dev_eval_dict = {}
    global_step = 0
    amp = None

    if args.load_pretrained_model:
        logger.info('Loading saved model from: %s' % args.load_pretrained_model)
        checkpoint = torch.load(args.load_pretrained_model, map_location=args.device)
        #if not load_only_bert:
        #    model.load_state_dict(checkpoint['model_state_dict'])
        #else:
        model_dict = model.state_dict()
        #bert_params = {k: v for k,v in checkpoint['model_state_dict'].items() if 'bert' in k}
        #model_dict.update(bert_params)
        shared_params = {k: v for k,v in checkpoint['model_state_dict'].items() if k in model_dict.keys()}
        model_dict.update(shared_params)
        model.load_state_dict(model_dict)

        if strict:
            logger.info('Loading saved optimizer and scheduler from: %s' % args.load_pretrained_model)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(args.device)
            dev_eval_dict = checkpoint['dev_eval_dict']
            global_step = dev_eval_dict['global_step']

    model.to(args.device)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        if strict:
            amp.load_state_dict(checkpoint['amp_state_dict'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        if args.fp16:
            from apex.parallel import DistributedDataParallel
            logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
            model = DistributedDataParallel(model)
        else:
            from torch.nn.parallel import DistributedDataParallel
            logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
            model = DistributedDataParallel(model,
                                            device_ids=[args.local_rank],
                                            output_device=args.local_rank,
                                            find_unused_parameters=True)
    return model, optimizer, scheduler, amp, dev_eval_dict, global_step

def apply_temperature(args, vocab, model_without_temperature):
    model = Multitask_Model(vocab=vocab,
                            tasks=args.tasks,
                            task_levels=args.task_levels,
                            task_weights=args.task_weights,
                            dropout=args.dropout,
                            bert_dropout=args.bert_dropout,
                            layer_dropout=args.layer_dropout,
                            word_dropout=args.word_dropout,
                            mix_embedding=args.mix_embedding,
                            label_smoothing=args.label_smoothing,
                            unfreeze_bert=args.unfreeze_bert,
                            pretrained_model=args.pretrained_model,
                            bert_combine_layers=args.bert_combine_layers,
                            scale_temperature=True)

    optimizer = AdamW([p for n, p in model.named_parameters() if 'temperature' in n], lr=0.001, eps=args.adam_epsilon)
    # optimizer = LBFGS([p for n, p in model.named_parameters() if 'temperature' in n], lr=0.01, max_iter=10)
    amp = None

    model_dict = model.state_dict()
    shared_params = {k: v for k,v in model_without_temperature.state_dict().items() if k in model_dict.keys()}
    model_dict.update(shared_params)
    model.load_state_dict(model_dict)
    model.to(args.device)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        if args.fp16:
            from apex.parallel import DistributedDataParallel
            logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
            model = DistributedDataParallel(model)
        else:
            from torch.nn.parallel import DistributedDataParallel
            logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
            model = DistributedDataParallel(model,
                                            device_ids=[args.local_rank],
                                            output_device=args.local_rank,
                                            find_unused_parameters=True)
    return model, optimizer, amp

def save_checkpoint(model, optimizer, amp, scheduler, dev_eval_dict, model_path):
    logger.info('Saving model to: %s' % model_path)
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    if amp is not None:
        state = {'model_state_dict': model_to_save.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'amp_state_dict': amp.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(),
                 'dev_eval_dict': dev_eval_dict}
    else:
        state = {'model_state_dict': model_to_save.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'amp_state_dict': amp,
                 'scheduler_state_dict': scheduler.state_dict(),
                 'dev_eval_dict': dev_eval_dict}
    torch.save(state, model_path)