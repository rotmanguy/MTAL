import logging
import numpy as np
import os
import torch
from tqdm import tqdm

from active_learning.al_selection_utils import enable_dropout, select_by_independent_selection, select_by_pareto, select_by_rrf

from active_learning.al_utils import compute_accuracy, compute_statistics, save_confidence_per_task, save_sample_ids
from active_learning.entropy_utils import compute_sequence_label_entropy, compute_dependency_entropy
from io_.io_utils import clean_batch

logger = logging.getLogger('mylog')

sample_ids = []

def run_active_learning(args, model, dataloaders, al_iter_num):
    if 'unlabeled' in dataloaders:
        if args.al_selection_method == 'entropy_based_confidence':
            sample_ids = entropy_based_confidence(args, model, dataloaders['unlabeled'], args.task_for_scoring, al_iter_num)
        elif args.al_selection_method == 'dropout_agreement':
            sample_ids = dropout_agreement(args, model, dataloaders['unlabeled'], args.task_for_scoring, al_iter_num)
        elif args.al_selection_method == 'average_entropy_based_confidence':
            sample_ids = average_entropy_based_confidence(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_selection_method == 'average_dropout_agreement':
            sample_ids = average_dropout_agreement(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_selection_method == 'maximum_entropy_based_confidence':
            sample_ids = maximum_entropy_based_confidence(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_selection_method == 'minimum_entropy_based_confidence':
            sample_ids = minimum_entropy_based_confidence(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_selection_method == 'pareto_entropy_based_confidence':
            sample_ids = pareto_entropy_based_confidence(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_selection_method == 'rrf_entropy_based_confidence':
            sample_ids = rrf_entropy_based_confidence(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_selection_method == 'independent_selection_entropy_based_confidence':
            sample_ids = independent_selection_entropy_based_confidence(args, model, dataloaders['unlabeled'], al_iter_num)
        elif args.al_selection_method == 'from_file':
            sample_ids = load_from_file(args, al_iter_num)
        else: ## random
            sample_ids = random_selection(args, model, dataloaders['unlabeled'], al_iter_num)
    else:
        sample_ids = []
    return sample_ids

def entropy_based_confidence(args, model, dataloader, task, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d" % len(dataloader.dataset))
    if task not in args.tasks:
        raise ValueError(task + ' should be in args.tasks')

    # Set model in eval mode
    model.eval()

    # Initialize dictionaries
    al_scores = {}
    accuracy_scores = {}

    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Decode
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)

            # Compute normalized entropy
            if task == 'deps':
                entropy = compute_dependency_entropy(outputs['head_preds'], outputs['tag_preds'], lengths)
            else:
                entropy = compute_sequence_label_entropy(outputs[task + '_preds'], lengths)

            # Compute confidence
            confidence = 1 - entropy
            confidence = list(confidence.cpu().numpy())

            # Compute accuracy
            if task == 'deps':
                accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
            else:
                accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)

            # Update dictionaries
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))

    # Choose unlabeled samples with the lowest confidence
    sample_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]

    # Compute statistics and save metadata to disk
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    save_sample_ids(al_iter_num, args, sample_ids)
    save_confidence_per_task(args, 'unlabeled', al_scores, al_iter_num)

    return sample_ids

def dropout_agreement(args, model, dataloader, task, al_iter_num, k=10):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d" % len(dataloader.dataset))
    if task not in args.tasks:
        raise ValueError(task + ' should be in args.tasks')
    if k < 2:
        raise ValueError('number of dropout models (k) should be larger than 1.')

    # Set model in eval mode
    model.eval()
    enable_dropout(model)

    # Initialize dictionaries
    al_scores = {}
    accuracy_scores = {}

    for batch in tqdm(dataloader, desc="Calculating Scores for unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Gather predictions from all tasks.
            # For deps, aside from gathering head predictions we also gather tag predictions (for future implementation).
            predictions = [[]] if task != 'deps' else [[], []]
            for i in range(k):
                # Decode per model
                outputs = model(**batch)
                if hasattr(model, "module"):
                    outputs = model.module.decode(outputs)
                else:
                    outputs = model.decode(outputs)

                # Get predictions
                if task == 'deps':
                    predictions[0].append(outputs['predicted_heads'])
                    predictions[1].append(outputs['predicted_head_tags'])
                else:
                    predictions[0].append(outputs['predicted_' + task])

                # Get lengths only once
                if i == 0:
                    lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)

            # Compute confidence (average agreement between all k dropout models)
            confidence = np.array([0.0 for _ in range(len(lengths))])
            for i in range(k - 1):
                for j in range(i + 1, k):
                    predicted_head_i = predictions[0][i]
                    predicted_head_j = predictions[0][j]
                    acc_i_j = compute_accuracy(predicted_head_i, predicted_head_j, lengths)
                    acc_j_i = compute_accuracy(predicted_head_j, predicted_head_i, lengths)
                    confidence += (acc_i_j + acc_j_i)
            confidence /= (k * (k - 1))
            confidence = list(confidence)

            # Update dictionary
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))

    # Disable dropout and compute accuracy for non-denoised outputs
    model.eval()
    for batch in tqdm(dataloader, desc="Calculating Scores for unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Decode
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)

            # Compute accuracy
            if task == 'deps':
                accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
            else:
                accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)

            # Update dictionary
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))

    # Choose unlabeled samples with the lowest confidence
    sample_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]

    # Compute statistics and save metadata to disk
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    save_sample_ids(al_iter_num, args, sample_ids)
    save_confidence_per_task(args, 'unlabeled', al_scores, al_iter_num)

    return sample_ids

def average_entropy_based_confidence(args, model, dataloader, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than 1.')

    # Set model in eval mode
    model.eval()

    # Initialize dictionaries
    al_scores = {}
    accuracy_scores = {}
    sample_to_confidences = {}

    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Decode
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)

            # Compute normalized entropy
            entropies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_entropy = compute_dependency_entropy(outputs['head_preds'], outputs['tag_preds'], lengths)
                else:
                    cur_entropy = compute_sequence_label_entropy(outputs[task + '_preds'], lengths)
                entropies.append(cur_entropy)
            entropies = torch.stack(entropies, axis=0)

            # Compute confidence
            confidences = 1 - entropies
            if args.task_weights_for_selection is None:
                confidence = torch.mean(confidences, axis=0).cpu().numpy()
            else:
                confidence = torch.sum(confidences * torch.tensor(args.task_weights_for_selection).unsqueeze(-1).to(args.device), axis=0).cpu().numpy()

            # Compute accuracy
            accuracies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
                else:
                    cur_accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
                accuracies.append(cur_accuracy)
            accuracies = np.stack(accuracies, axis=0)
            accuracy = np.mean(accuracies, axis=0)

            # Update dictionaries
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))
            sample_to_confidences.update(dict(zip(outputs['sample_id'], confidences.cpu().numpy().transpose())))

    # Choose unlabeled samples with the lowest confidence
    sample_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]

    # Compute statistics and save metadata to disk
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    save_sample_ids(al_iter_num, args, sample_ids)
    save_confidence_per_task(args, 'unlabeled', sample_to_confidences, al_iter_num)

    return sample_ids

def average_dropout_agreement(args, model, dataloader, al_iter_num, k=10):
    logger.info("***** Scoring unlabeled Samples *****" )
    logger.info("Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than 1.')
    if k < 2:
        raise ValueError('number of dropout models (k) should be larger than 1.')

    # Set model in eval mode
    model.eval()
    enable_dropout(model)

    # Initialize dictionaries
    al_scores = {}
    accuracy_scores = {}
    sample_to_confidences = {}

    for batch in tqdm(dataloader, desc="Calculating Scores for unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Gather predictions from all tasks.
            # For deps, aside from gathering head predictions we also gather tag predictions (for future implementation).
            predictions = {task: [[]] if task != 'deps' else [[], []] for task in args.tasks}
            for i in range(k):
                # Decode per model
                outputs = model(**batch)
                if hasattr(model, "module"):
                    outputs = model.module.decode(outputs)
                else:
                    outputs = model.decode(outputs)

                # Get predictions
                for task in args.tasks:
                    if task == 'deps':
                        predictions[task][0].append(outputs['predicted_heads'])
                        predictions[task][1].append(outputs['predicted_head_tags'])
                    else:
                        predictions[task][0].append(outputs['predicted_' + task])

                # Get lengths only once
                if i == 0:
                    lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)

            # Compute confidence
            confidences = []
            for task in args.tasks:
                cur_confidence = np.array([0.0 for _ in range(len(lengths))])
                for i in range(k - 1):
                    for j in range(i + 1, k):
                        predicted_head_i = predictions[task][0][i]
                        predicted_head_j = predictions[task][0][j]
                        acc_i_j = compute_accuracy(predicted_head_i, predicted_head_j, lengths)
                        acc_j_i = compute_accuracy(predicted_head_j, predicted_head_i, lengths)
                        cur_confidence += (acc_i_j + acc_j_i)
                cur_confidence /= (k * (k - 1))
                confidences.append(cur_confidence)
            confidences = np.stack(confidences, axis=0)
            if args.task_weights_for_selection is None:
                confidence = np.mean(confidences, axis=0)
            else:
                confidence = np.sum(confidences * np.expand_dims(np.array(args.task_weights_for_selection), axis=1), axis=0)

            # Update dictionaries
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            sample_to_confidences.update(dict(zip(outputs['sample_id'], confidences.transpose())))

    # Disable dropout and compute accuracy for non-denoised outputs
    model.eval()
    for batch in tqdm(dataloader, desc="Calculating Scores for unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Decode
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)
            lengths = list(lengths.cpu().numpy())

            # Compute accuracy
            accuracies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
                else:
                    cur_accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
                accuracies.append(cur_accuracy)
            accuracies = np.stack(accuracies, axis=0)
            accuracy = np.mean(accuracies, axis=0)

            # Update dictionary
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))

    # Choose unlabeled samples with the lowest confidence
    sample_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]

    # Compute statistics and save metadata to disk
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    save_sample_ids(al_iter_num, args, sample_ids)
    save_confidence_per_task(args, 'unlabeled', sample_to_confidences, al_iter_num)

    return sample_ids

def maximum_entropy_based_confidence(args, model, dataloader, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than 1.')

    # Set model in eval mode
    model.eval()

    # Initialize dictionaries
    al_scores = {}
    accuracy_scores = {}
    sample_to_confidences = {}

    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Decode
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)

            # Compute confidence
            entropies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_entropy = compute_dependency_entropy(outputs['head_preds'], outputs['tag_preds'], lengths)
                else:
                    cur_entropy = compute_sequence_label_entropy(outputs[task + '_preds'], lengths)
                entropies.append(cur_entropy)
            entropies = torch.stack(entropies, axis=0)
            confidences = 1 - entropies
            if args.task_weights_for_selection is None:
                confidence = torch.max(confidences, axis=0).values.cpu().numpy()
            else:
                confidence = torch.max(confidences * torch.tensor(args.task_weights_for_selection).unsqueeze(-1).to(args.device), axis=0).values.cpu().numpy()

            # Compute accuracy
            accuracies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
                else:
                    cur_accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
                accuracies.append(cur_accuracy)
            accuracies = np.stack(accuracies, axis=0)
            accuracy = np.mean(accuracies, axis=0)

            # Update dictionaries
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))
            sample_to_confidences.update(dict(zip(outputs['sample_id'], confidences.cpu().numpy().transpose())))

    # Choose unlabeled samples with the lowest confidence
    sample_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]

    # Compute statistics and save metadata to disk
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    save_sample_ids(al_iter_num, args, sample_ids)
    save_confidence_per_task(args, 'unlabeled', sample_to_confidences, al_iter_num)

    return sample_ids

def minimum_entropy_based_confidence(args, model, dataloader, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than 1.')

    # Set model in eval mode
    model.eval()

    # Initialize Dictionaries
    al_scores = {}
    accuracy_scores = {}
    sample_to_confidences = {}

    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Decode
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)

            # Compute confidence
            entropies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_entropy = compute_dependency_entropy(outputs['head_preds'], outputs['tag_preds'], lengths)
                else:
                    cur_entropy = compute_sequence_label_entropy(outputs[task + '_preds'], lengths)
                entropies.append(cur_entropy)
            entropies = torch.stack(entropies, axis=0)
            confidences = 1 - entropies
            if args.task_weights_for_selection is None:
                confidence = torch.min(confidences, axis=0).values.cpu().numpy()
            else:
                confidence = torch.min(confidences * torch.tensor(args.task_weights_for_selection).unsqueeze(-1).to(args.device), axis=0).values.cpu().numpy()

            # Compute accuracy
            accuracies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
                else:
                    cur_accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
                accuracies.append(cur_accuracy)
            accuracies = np.stack(accuracies, axis=0)
            accuracy = np.mean(accuracies, axis=0)

            # Update dictionaries
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))
            sample_to_confidences.update(dict(zip(outputs['sample_id'], confidences.cpu().numpy().transpose())))

    # Choose unlabeled samples with the lowest confidence
    sample_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]

    # Compute statistics and save metadata to disk
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    save_sample_ids(al_iter_num, args, sample_ids)
    save_confidence_per_task(args, 'unlabeled', sample_to_confidences, al_iter_num)

    return sample_ids

def pareto_entropy_based_confidence(args, model, dataloader, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than 1.')

    # Set model in eval mode
    model.eval()

    # Initialize dictionary
    sample_to_confidences = {}

    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Decode
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)

            # Compute confidence
            entropies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_entropy = compute_dependency_entropy(outputs['head_preds'], outputs['tag_preds'], lengths)
                else:
                    cur_entropy = compute_sequence_label_entropy(outputs[task + '_preds'], lengths)
                entropies.append(cur_entropy)
            entropies = torch.stack(entropies, axis=0)
            confidences = 1 - entropies

            # Update dictionary
            sample_to_confidences.update(dict(zip(outputs['sample_id'], confidences.cpu().numpy().transpose())))

    # Choose unlabeled samples from the minimal Pareto frontier
    sample_ids = select_by_pareto(args, sample_to_confidences)

    # Compute statistics and save metadata to disk
    save_sample_ids(al_iter_num, args, sample_ids)
    save_confidence_per_task(args, 'unlabeled', sample_to_confidences, al_iter_num)

    return sample_ids

def rrf_entropy_based_confidence(args, model, dataloader, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than 1.')

    # Set model in eval mode
    model.eval()

    # Initialize dictionary
    sample_to_confidences = {}

    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Decode
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)

            # Compute confidence
            entropies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_entropy = compute_dependency_entropy(outputs['head_preds'], outputs['tag_preds'], lengths)
                else:
                    cur_entropy = compute_sequence_label_entropy(outputs[task + '_preds'], lengths)
                entropies.append(cur_entropy)
            entropies = torch.stack(entropies, axis=0)
            confidences = 1 - entropies

            # Update dictionary
            sample_to_confidences.update(dict(zip(outputs['sample_id'], confidences.cpu().numpy().transpose())))

    # Choose unlabeled samples according to the RRF method
    sample_ids = select_by_rrf(args, sample_to_confidences)

    # Compute statistics and save metadata to disk
    save_sample_ids(al_iter_num, args, sample_ids)
    save_confidence_per_task(args, 'unlabeled', sample_to_confidences, al_iter_num)

    return sample_ids

def independent_selection_entropy_based_confidence(args, model, dataloader, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than 1.')

    # Set model in eval mode
    model.eval()

    # Initialize dictionary
    sample_to_confidences = {}

    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Decode
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)

            # Compute confidence
            entropies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_entropy = compute_dependency_entropy(outputs['head_preds'], outputs['tag_preds'], lengths)
                else:
                    cur_entropy = compute_sequence_label_entropy(outputs[task + '_preds'], lengths)
                entropies.append(cur_entropy)
            entropies = torch.stack(entropies, axis=0)
            confidences = 1 - entropies

            # Update dictionary
            sample_to_confidences.update(dict(zip(outputs['sample_id'], confidences.cpu().numpy().transpose())))

    # Choose unlabeled samples independently from each task
    sample_ids = select_by_independent_selection(args, sample_to_confidences)

    # Compute statistics and save metadata to disk
    save_sample_ids(al_iter_num, args, sample_ids)
    save_confidence_per_task(args, 'unlabeled', sample_to_confidences, al_iter_num)

    return sample_ids

def load_from_file(args, al_iter_num):
    """
    Loading sample ids from a pre-saved file (please specify the directory of the file: 'args.load_sample_ids_dir')
    """

    assert args.load_sample_ids_dir is not None

    # Load file
    sample_ids_file = [os.path.join(args.load_sample_ids_dir, f) for f in os.listdir(args.load_sample_ids_dir) if
                        args.src_domain + '_sample_ids' in f and 'iter_' + str(al_iter_num) in f][0]

    # Choose unlabeled samples from a file (according to their order of appearance)
    sample_ids = []
    i = 0
    with open(sample_ids_file, 'r') as f:
        for line in f.readlines():
            sample_id = int(line.strip())
            sample_ids.append(sample_id)
            i += 1
            if i >= args.training_samples_to_add:
                break

    # Save metadata to disk
    save_sample_ids(al_iter_num, args, sample_ids)

    return sample_ids

def random_selection(args, model, dataloader, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("Num examples = %d", len(dataloader.dataset))
    model.eval()

    # Shuffle samples in a random order
    all_samples = []
    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            all_samples += [x['sample_id'] for x in batch['metadata']]
    np.random.shuffle(all_samples)

    # Choose unlabeled samples randomly
    sample_ids = all_samples[:args.training_samples_to_add]

    # Save metadata to disk
    save_sample_ids(al_iter_num, args, sample_ids)

    return sample_ids