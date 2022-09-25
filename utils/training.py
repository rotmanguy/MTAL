from copy import deepcopy
import logging
import os
from tensorboardX import SummaryWriter
from torch import distributed
from torch.nn.utils import clip_grad_norm_
from tqdm import trange, tqdm

from io_.io_utils import clean_batch
from active_learning.al_methods import run_active_learning
from utils.conll18_ud_eval import write_results_file
from utils.evaluation import main_evaluation, predict_and_evaluate
from utils.utils import set_seed, build_model

logger = logging.getLogger('mylog')


def train(args, dataloaders, vocab, al_iter_num=0):
    # Setting number of training steps
    if args.max_steps > 0:
        num_training_steps = args.max_steps
        args.num_train_epochs = args.max_steps // (len(dataloaders['train']) // args.gradient_accumulation_steps) + 1
    else:
        num_training_steps = len(dataloaders['train']) // args.gradient_accumulation_steps * args.num_train_epochs
    if args.load_checkpoint_from_last_iter:
        num_training_steps *= args.num_al_iterations

    # Continue training model from last AL iteration if load_from_last_iter_bool is True
    set_back_to_old_val = None
    load_from_last_iter_bool = args.load_pretrained_model is None and \
                               not args.do_eval and al_iter_num > 0 and \
                               args.load_checkpoint_from_last_iter
    if load_from_last_iter_bool:
        set_back_to_old_val = deepcopy(args.load_pretrained_model)
        args.load_pretrained_model = os.path.join(args.output_dir, args.src_domain + '.tar.gz')

    # Loading training objects
    model, optimizer, scheduler, amp, dev_eval_dict, global_step = build_model(args,
                                                                               vocab,
                                                                               num_training_steps,
                                                                               strict=False)
    if load_from_last_iter_bool:
        args.load_pretrained_model = set_back_to_old_val

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
                        distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", num_training_steps)

        patient = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
        set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
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
                        clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

                    # Running evaluation
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

        # Running evaluation for one last time during training
        logger.info('step is %d \n' % step)
        logger.info('global step is %d \n' % global_step)
        dev_eval_dict, _ = main_evaluation(args, dataloaders, model, optimizer, scheduler, amp, dev_eval_dict, patient, global_step, args.src_domain)

        # Saving and loading best model
        best_model_path = os.path.join(args.output_dir, args.src_domain + '.tar.gz')
        if os.path.exists(best_model_path):
            set_back_to_old_val = deepcopy(args.load_pretrained_model)
            args.load_pretrained_model = best_model_path
            model = None # to free space
            model, optimizer, scheduler, amp, dev_eval_dict, global_step = build_model(args, vocab,
                                                                                       num_training_steps,
                                                                                       strict=True)
            args.load_pretrained_model = set_back_to_old_val
        else:
            logger.info('We did not find a saved model. Evaluating with current model')

    # Eval!
    logger.info('--- Performing final evaluation ---')
    for split in dataloaders.keys():
        eval_dict = predict_and_evaluate(args, model, dataloaders[split], split, global_step, args.src_domain, final_eval=True)
        res_filename = os.path.join(args.output_dir, args.src_domain + '_res_' + split + '_iter_' + str(al_iter_num) + '.conllu')
        write_results_file(eval_dict, res_filename)

    # Run AL step (choosing unlabeled samples for annotation)
    sample_ids = run_active_learning(args, model, dataloaders, al_iter_num)
    return sample_ids
