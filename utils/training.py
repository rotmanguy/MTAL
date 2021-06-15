import logging
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import io_
from io_.dataset_readers.load_dataset_reader import load_dataset_reader_
from tqdm import trange, tqdm

from active_learning_utils.ece_calculation import set_temperature
from io_.io_utils import clean_batch
from utils.active_learning import active_learning_entropy, active_learning_joint_entropy, active_learning_random, \
    active_learning_dropout_entropy, active_learning_dropout_joint_entropy, active_learning_dropout_agreement, \
    active_learning_dropout_joint_agreement, active_learning_joint_max_entropy, active_learning_joint_min_entropy
from utils.conll18_ud_eval import write_results_file
from utils.evaluation import main_evaluation, predict_and_evaluate
from utils.utils import set_seed, build_model, save_checkpoint

logger = logging.getLogger('mylog')


def train(args, dataloaders, vocab, al_iter_num=0):
    """ Train the model """
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(dataloaders['train']) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(dataloaders['train']) // args.gradient_accumulation_steps * args.num_train_epochs


    model, optimizer, scheduler, amp, dev_eval_dict, global_step = build_model(args,
                                                                               vocab,
                                                                               t_total,
                                                                               strict=False)

    logger.info("Model Info: \n%s" % model)
    logger.info("Optimizer Info: \n%s" % optimizer)
    logger.info("Scheduler Info: \n%s" % scheduler)

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    if not args.do_eval:
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(dataloaders['train'].dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        patient = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
        set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
        for _ in train_iterator:
            epoch_iterator = tqdm(dataloaders['train'], desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = clean_batch(args, batch, mode='train')
                outputs = model(**batch)
                loss = outputs['loss']  # model outputs is a dictionary
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                        logger.info('step is %d \n' % step)
                        logger.info('glboal step is %d \n' % global_step)
                        dev_eval_dict, patient = main_evaluation(args, dataloaders, model, optimizer, scheduler, amp, dev_eval_dict, patient, global_step, args.src_domain, final_eval=False)
                    scheduler.step()  # Update learning rate schedule

                    if patient > 9:
                        epoch_iterator.close()
                        break
                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        if args.local_rank in [-1, 0]:
            tb_writer.close()

        logger.info('step is %d \n' % step)
        logger.info('global step is %d \n' % global_step)
        dev_eval_dict, _ = main_evaluation(args, dataloaders, model, optimizer, scheduler, amp, dev_eval_dict, patient, global_step, args.src_domain)
        best_model_path = os.path.join(args.output_dir, args.src_domain + '.tar.gz')
        if os.path.exists(best_model_path):
            args.load_pretrained_model = best_model_path
            model = None # to free space
            model, optimizer, scheduler, amp, dev_eval_dict, global_step = build_model(args, vocab,
                                                                                       t_total,
                                                                                       strict=False)
        else:
            logger.info('We did not find a saved model. Evaluating with current model')

    # logger.info('--- Optimizing Scaled Temperature ---')
    # model = set_temperature(args, vocab, model, dataloaders['dev'])
    # model_path = os.path.join(args.output_dir, args.src_domain + '.tar.gz')
    # save_checkpoint(model, optimizer, amp, scheduler, dev_eval_dict, model_path)

    logger.info('--- Performing final evaluation ---')
    for split in dataloaders.keys():
        eval_dict = predict_and_evaluate(args, model, dataloaders[split], split, global_step, args.src_domain, final_eval=True)
        res_filename = os.path.join(args.output_dir, args.src_domain + '_res_' + split + '_iter_' + str(al_iter_num) + '.conllu')
        write_results_file(eval_dict, res_filename)

    if args.tgt_domains is not None:
        for tgt_domain in args.tgt_domains:
            if tgt_domain == args.src_domain:
                continue
            domain_results_dir = os.path.join(args.output_dir, 'tgt_domains', tgt_domain)
            if not os.path.isdir(domain_results_dir):
                os.makedirs(domain_results_dir)
            tgt_data_paths = {}
            tgt_data_splits = ['test']
            for data_split in tgt_data_splits:
                data_path = os.path.join(args.data_dir, args.dataset, tgt_domain, data_split + '.conllu')
                if os.path.exists(data_path):
                    tgt_data_paths[data_split] = data_path
            tgt_datasets = {}
            reader = load_dataset_reader_(args)
            for data_split in tgt_data_paths.keys():
                tgt_datasets[data_split] = io_.dataset_readers.convert_allenlp_reader_to_pytorch_dataset.AllennlpDataset(
                    vocab=vocab,
                    reader=reader,
                    dataset_path=tgt_data_paths[data_split])
            tgt_dataloaders = {}
            for data_split in tgt_datasets.keys():
                tgt_dataloaders[data_split] = DataLoader(tgt_datasets[data_split],
                                                     shuffle=False,
                                                     batch_size=args.eval_batch_size,
                                                     collate_fn=tgt_datasets[data_split].allennlp_collocate)

            logger.info('--- Performing final evaluation on %s---' % tgt_domain)
            print('--- Performing final evaluation on %s---' % tgt_domain)
            for split in tgt_dataloaders.keys():
                eval_dict = predict_and_evaluate(args, model, tgt_dataloaders[split], split, global_step, tgt_domain, final_eval=True)
                res_filename = os.path.join(domain_results_dir, 'eval_' + split + '.csv')
                write_results_file(eval_dict, res_filename)

    if args.do_eval:
        return None
    sample_ids = []
    if 'unlabeled' in dataloaders:
        if args.al_scoring == 'joint_entropy':
            sample_ids = active_learning_joint_entropy(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_scoring == 'joint_max_entropy':
            sample_ids = active_learning_joint_max_entropy(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_scoring == 'joint_min_entropy':
            sample_ids = active_learning_joint_min_entropy(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_scoring == 'dropout_joint_entropy':
            sample_ids = active_learning_dropout_joint_entropy(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_scoring == 'dropout_joint_agreement':
            sample_ids = active_learning_dropout_joint_agreement(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_scoring == 'entropy':
            sample_ids = active_learning_entropy(args, model, dataloaders['unlabeled'], args.task_for_scoring, al_iter_num)
        elif args.al_scoring == 'dropout_entropy':
            sample_ids = active_learning_dropout_entropy(args, model, dataloaders['unlabeled'], args.task_for_scoring, al_iter_num)
        elif args.al_scoring == 'dropout_agreement':
            sample_ids = active_learning_dropout_agreement(args, model, dataloaders['unlabeled'], args.task_for_scoring, al_iter_num)
        else: ## random
            sample_ids = active_learning_random(args, model, dataloaders['unlabeled'], al_iter_num)
    return sample_ids