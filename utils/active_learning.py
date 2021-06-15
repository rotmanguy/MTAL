import csv
import logging
import os

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from tqdm import tqdm

from active_learning_utils.al_utils import compute_sequence_label_entropy, compute_dependency_entropy
from io_.io_utils import clean_batch

logger = logging.getLogger('mylog')

def active_learning_joint_entropy(args, model, dataloader, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than one')
    model.eval()
    al_scores = {}
    accuracy_scores = {}
    lengths_confidence = []
    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)
            entropies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_entropy = compute_dependency_entropy(outputs['head_preds'], outputs['tag_preds'], lengths)
                else:
                    cur_entropy = compute_sequence_label_entropy(outputs[task + '_preds'], lengths)
                entropies.append(cur_entropy)
            entropies = torch.stack(entropies, axis=0)
            entropy = torch.mean(entropies, axis=0)
            confidence = 1 - entropy
            confidence = confidence.cpu().numpy()
            accuracies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
                else:
                    cur_accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
                accuracies.append(cur_accuracy)
            accuracies = np.stack(accuracies, axis=0)
            accuracy = np.mean(accuracies, axis=0)
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))
            lengths = list(lengths.cpu().numpy())
            lengths_confidence += [(length_, confidence_) for length_, confidence_ in zip(lengths, confidence)]

    # check = [(np.mean([l_.item() for l_, e_ in sorted([(l, e/(l**div)) for (l,e) in lengths_entropy], key=lambda item: item[1], reverse=True)][:args.training_samples_to_add]), div) for div in [1, 0.75, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]]
    average_length = [(np.mean([l_.item() for l_, c_ in sorted([(l, c) for (l, c) in lengths_confidence], key=lambda item: item[1])][:args.training_samples_to_add]))]
    logger.info("------------- average_length -------")
    logger.info(average_length)
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    samples_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]
    sample_ids_file = os.path.join(args.output_dir, args.src_domain + '_sample_ids_joint_entropy_'+ '_'.join(args.tasks) + '_iter_' + str(al_iter_num) + '.csv')
    with open(sample_ids_file, 'w', newline='') as myfile:
        myfile.write("\n".join([str(id) for id in samples_ids]))
    return samples_ids

def active_learning_joint_max_entropy(args, model, dataloader, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than one')
    model.eval()
    al_scores = {}
    accuracy_scores = {}
    lengths_confidence = []
    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)
            entropies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_entropy = compute_dependency_entropy(outputs['head_preds'], outputs['tag_preds'], lengths)
                else:
                    cur_entropy = compute_sequence_label_entropy(outputs[task + '_preds'], lengths)
                entropies.append(cur_entropy)
            entropies = torch.stack(entropies, axis=0)
            entropy = torch.max(entropies, axis=0).values
            confidence = 1 - entropy
            confidence = confidence.cpu().numpy()
            accuracies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
                else:
                    cur_accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
                accuracies.append(cur_accuracy)
            accuracies = np.stack(accuracies, axis=0)
            accuracy = np.mean(accuracies, axis=0)
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))
            lengths = list(lengths.cpu().numpy())
            lengths_confidence += [(length_, confidence_) for length_, confidence_ in zip(lengths, confidence)]

    # check = [(np.mean([l_.item() for l_, e_ in sorted([(l, e/(l**div)) for (l,e) in lengths_entropy], key=lambda item: item[1], reverse=True)][:args.training_samples_to_add]), div) for div in [1, 0.75, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]]
    average_length = [(np.mean([l_.item() for l_, c_ in sorted([(l, c) for (l, c) in lengths_confidence], key=lambda item: item[1])][:args.training_samples_to_add]))]
    logger.info("------------- average_length -------")
    logger.info(average_length)
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    samples_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]
    sample_ids_file = os.path.join(args.output_dir, args.src_domain + '_sample_ids_joint_max_entropy_' + '_'.join(args.tasks) + '_iter_' + str(al_iter_num) + '.csv')
    with open(sample_ids_file, 'w', newline='') as myfile:
        myfile.write("\n".join([str(id) for id in samples_ids]))
    return samples_ids

def active_learning_joint_min_entropy(args, model, dataloader, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than one')
    model.eval()
    al_scores = {}
    accuracy_scores = {}
    lengths_confidence = []
    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)
            entropies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_entropy = compute_dependency_entropy(outputs['head_preds'], outputs['tag_preds'], lengths)
                else:
                    cur_entropy = compute_sequence_label_entropy(outputs[task + '_preds'], lengths)
                entropies.append(cur_entropy)
            entropies = torch.stack(entropies, axis=0)
            entropy = torch.min(entropies, axis=0).values
            confidence = 1 - entropy
            confidence = confidence.cpu().numpy()
            accuracies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
                else:
                    cur_accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
                accuracies.append(cur_accuracy)
            accuracies = np.stack(accuracies, axis=0)
            accuracy = np.mean(accuracies, axis=0)
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))
            lengths = list(lengths.cpu().numpy())
            lengths_confidence += [(length_, confidence_) for length_, confidence_ in zip(lengths, confidence)]

    # check = [(np.mean([l_.item() for l_, e_ in sorted([(l, e/(l**div)) for (l,e) in lengths_entropy], key=lambda item: item[1], reverse=True)][:args.training_samples_to_add]), div) for div in [1, 0.75, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]]
    average_length = [(np.mean([l_.item() for l_, c_ in sorted([(l, c) for (l, c) in lengths_confidence], key=lambda item: item[1])][:args.training_samples_to_add]))]
    logger.info("------------- average_length -------")
    logger.info(average_length)
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    samples_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]
    sample_ids_file = os.path.join(args.output_dir, args.src_domain + '_sample_ids_joint_min_entropy_' + '_'.join(args.tasks) + '_iter_' + str(al_iter_num) + '.csv')
    with open(sample_ids_file, 'w', newline='') as myfile:
        myfile.write("\n".join([str(id) for id in samples_ids]))
    return samples_ids

def active_learning_dropout_joint_entropy(args, model, dataloader, al_iter_num):
    def enable_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than one')
    model.eval()
    enable_dropout(model)
    al_scores = {}
    accuracy_scores = {}
    lengths_confidence = []
    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            preds = {task: [[]] if task != 'deps' else [[], []] for task in args.tasks}
            for _ in range(10):
                outputs = model(**batch)
                if hasattr(model, "module"):
                    outputs = model.module.decode(outputs)
                else:
                    outputs = model.decode(outputs)
                for task in args.tasks:
                    if task == 'deps':
                        preds[task][0].append(outputs['head_preds'])
                        preds[task][1].append(outputs['tag_preds'])
                    else:
                        preds[task][0].append(outputs[task + '_preds'])

            for task in args.tasks:
                for i, pred in enumerate(preds[task]):
                    preds[task][i] = torch.mean(torch.stack(pred), axis=0)

            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)
            entropies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_entropy = compute_dependency_entropy(preds[task][0], preds[task][1], lengths)
                else:
                    cur_entropy = compute_sequence_label_entropy(preds[task][0], lengths)
                entropies.append(cur_entropy)
            entropies = torch.stack(entropies, axis=0)
            entropy = torch.mean(entropies, axis=0)
            confidence = 1 - entropy
            confidence = confidence.cpu().numpy()

            accuracies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
                else:
                    cur_accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
                accuracies.append(cur_accuracy)
            accuracies = np.stack(accuracies, axis=0)
            accuracy = np.mean(accuracies, axis=0)
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))
            lengths = list(lengths.cpu().numpy())
            lengths_confidence += [(length_, confidence_) for length_, confidence_ in zip(lengths, confidence)]

    #check = [(np.mean([l_.item() for l_, e_ in sorted([(l, e/(l**div)) for (l,e) in lengths_entropy], key=lambda item: item[1], reverse=True)][:args.training_samples_to_add]), div) for div in [1, 0.75, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]]
    average_length = [(np.mean([l_.item() for l_, c_ in sorted([(l, c) for (l, c) in lengths_confidence], key=lambda item: item[1])][:args.training_samples_to_add]))]
    logger.info("------------- average_length -------")
    logger.info(average_length)
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    samples_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]
    sample_ids_file = os.path.join(args.output_dir, args.src_domain + '_sample_ids_dropout_joint_entropy_' + '_'.join(args.tasks) + '_iter_' + str(al_iter_num) + '.csv')
    with open(sample_ids_file, 'w', newline='') as myfile:
        myfile.write("\n".join([str(id) for id in samples_ids]))
    model.eval()
    return samples_ids

def active_learning_entropy(args, model, dataloader, task, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d" % len(dataloader.dataset))
    if task not in args.tasks:
        raise ValueError(task + ' should be in args.tasks')
    model.eval()
    al_scores = {}
    accuracy_scores = {}
    lengths_confidence = []
    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)
            if task == 'deps':
                entropy = compute_dependency_entropy(outputs['head_preds'], outputs['tag_preds'], lengths)
            else:
                entropy = compute_sequence_label_entropy(outputs[task + '_preds'], lengths)
            confidence = 1 - entropy
            confidence = list(confidence.cpu().numpy())
            if task == 'deps':
                accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
            else:
                accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))
            lengths = list(lengths.cpu().numpy())
            lengths_confidence += [(length_, confidence_) for length_, confidence_ in zip(lengths, confidence)]

    #check = [(np.mean([l_.item() for l_, e_ in sorted([(l, e/(l**div)) for (l,e) in lengths_entropy], key=lambda item: item[1], reverse=True)][:args.training_samples_to_add]), div) for div in [1, 0.75, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]]
    average_length = [(np.mean([l_.item() for l_, c_ in sorted([(l, c) for (l, c) in lengths_confidence], key=lambda item: item[1])][:args.training_samples_to_add]))]
    logger.info("------------- average_length -------")
    logger.info(average_length)
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    samples_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]
    sample_ids_file = os.path.join(args.output_dir, args.src_domain + '_sample_ids_entropy_' + task + '_iter_' + str(al_iter_num) + '.csv')
    with open(sample_ids_file, 'w', newline='') as myfile:
        myfile.write("\n".join([str(id) for id in samples_ids]))
    return samples_ids

def active_learning_dropout_agreement(args, model, dataloader, task, al_iter_num):
    def enable_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("  Num examples = %d" % len(dataloader.dataset))
    if task not in args.tasks:
        raise ValueError(task + ' should be in args.tasks')
    model.eval()
    enable_dropout(model)
    al_scores = {}
    lengths_confidence = []
    n = 10
    for batch in tqdm(dataloader, desc="Calculating Scores for unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            predicted_tags = [[]] if task != 'deps' else [[], []]
            for i in range(n):
                outputs = model(**batch)
                if hasattr(model, "module"):
                    outputs = model.module.decode(outputs)
                else:
                    outputs = model.decode(outputs)
                if task == 'deps':
                    predicted_tags[0].append(outputs['predicted_heads'])
                    predicted_tags[1].append(outputs['predicted_head_tags'])
                else:
                    predicted_tags[0].append(outputs['predicted_' + task])
                if i == 0:
                    lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)
            if task == "deps":
                confidence = np.array([0.0 for _ in range(len(lengths))])
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        predicted_head_i = predicted_tags[0][i]
                        predicted_head_j = predicted_tags[0][j]
                        acc_i_j = compute_accuracy(predicted_head_i, predicted_head_j, lengths)
                        acc_j_i = compute_accuracy(predicted_head_j, predicted_head_i, lengths)
                        confidence += (acc_i_j + acc_j_i)
                confidence /= (n * (n - 1))
            else:
                confidence = np.array([0.0 for _ in range(len(lengths))])
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        predicted_tags_i = predicted_tags[0][i]
                        predicted_tags_j = predicted_tags[0][j]
                        acc_i_j = compute_accuracy(predicted_tags_i, predicted_tags_j, lengths)
                        acc_j_i = compute_accuracy(predicted_tags_j, predicted_tags_i, lengths)
                        confidence += (acc_i_j + acc_j_i)
                confidence /= (n * (n - 1))

            confidence = list(confidence)
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            lengths_confidence += [(length_, confidence_) for length_, confidence_ in zip(lengths, confidence)]

    model.eval()
    accuracy_scores = {}
    for batch in tqdm(dataloader, desc="Calculating Scores for unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)
            if task == 'deps':
                accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
            else:
                accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))

    #check = [(np.mean([l_.item() for l_, e_ in sorted([(l, e/(l**div)) for (l,e) in lengths_entropy], key=lambda item: item[1], reverse=True)][:args.training_samples_to_add]), div) for div in [1, 0.75, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]]
    average_length = [(np.mean([l_.item() for l_, c_ in sorted([(l, c) for (l, c) in lengths_confidence], key=lambda item: item[1])][:args.training_samples_to_add]))]
    logger.info("------------- average_length -------")
    logger.info(average_length)
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    samples_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]
    sample_ids_file = os.path.join(args.output_dir, args.src_domain + '_sample_ids_entropy_' + task + '_iter_' + str(al_iter_num) + '.csv')
    with open(sample_ids_file, 'w', newline='') as myfile:
        myfile.write("\n".join([str(id) for id in samples_ids]))
    return samples_ids

def active_learning_dropout_joint_agreement(args, model, dataloader, al_iter_num):
    def enable_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    logger.info("***** Scoring unlabeled Samples *****" )
    logger.info("Num examples = %d", len(dataloader.dataset))
    if len(args.tasks) < 2:
        raise ValueError('number of task should be larger than one')
    model.eval()
    enable_dropout(model)
    al_scores = {}
    lengths_confidence = []
    n = 10

    for batch in tqdm(dataloader, desc="Calculating Scores for unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            predicted_tags = {task: [[]] if task != 'deps' else [[], []] for task in args.tasks}
            for i in range(n):
                outputs = model(**batch)
                if hasattr(model, "module"):
                    outputs = model.module.decode(outputs)
                else:
                    outputs = model.decode(outputs)
                for task in args.tasks:
                    if task == 'deps':
                        predicted_tags[task][0].append(outputs['predicted_heads'])
                        predicted_tags[task][1].append(outputs['predicted_head_tags'])
                    else:
                        predicted_tags[task][0].append(outputs['predicted_' + task])
                if i == 0:
                    lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)
            confidences = []
            for task in args.tasks:
                cur_confidence = np.array([0.0 for _ in range(len(lengths))])
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        predicted_head_i = predicted_tags[task][0][i]
                        predicted_head_j = predicted_tags[task][0][j]
                        acc_i_j = compute_accuracy(predicted_head_i, predicted_head_j, lengths)
                        acc_j_i = compute_accuracy(predicted_head_j, predicted_head_i, lengths)
                        cur_confidence += (acc_i_j + acc_j_i)
                cur_confidence /= (n * (n - 1))
                confidences.append(cur_confidence)
            confidences = np.stack(confidences, axis=0)
            confidence = np.mean(confidences, axis=0)
            al_scores.update(dict(zip(outputs['sample_id'], confidence)))
            lengths = list(lengths.cpu().numpy())
            lengths_confidence += [(length_, confidence_) for length_, confidence_ in zip(lengths, confidence)]

    model.eval()
    accuracy_scores = {}
    for batch in tqdm(dataloader, desc="Calculating Scores for unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)
            lengths = list(lengths.cpu().numpy())
            accuracies = []
            for task in args.tasks:
                if task == 'deps':
                    cur_accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
                else:
                    cur_accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
                accuracies.append(cur_accuracy)
            accuracies = np.stack(accuracies, axis=0)
            accuracy = np.mean(accuracies, axis=0)
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))

    #check = [(np.mean([l_.item() for l_, e_ in sorted([(l, e/(l**div)) for (l,e) in lengths_entropy], key=lambda item: item[1], reverse=True)][:args.training_samples_to_add]), div) for div in [1, 0.75, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]]
    average_length = [(np.mean([l_.item() for l_, c_ in sorted([(l, c) for (l, c) in lengths_confidence], key=lambda item: item[1])][:args.training_samples_to_add]))]
    logger.info("------------- average_length -------")
    logger.info(average_length)
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    samples_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]
    sample_ids_file = os.path.join(args.output_dir, args.src_domain + '_sample_ids_entropy_' + '_'.join(args.tasks) + '_iter_' + str(al_iter_num) + '.csv')
    with open(sample_ids_file, 'w', newline='') as myfile:
        myfile.write("\n".join([str(id) for id in samples_ids]))
    return samples_ids

def active_learning_dropout_entropy(args, model, dataloader, task, al_iter_num):
    def enable_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("Num examples = %d", len(dataloader.dataset))
    if task not in args.tasks:
        raise ValueError(task + ' should be in args.tasks')

    model.eval()
    enable_dropout(model)
    al_scores = {}
    accuracy_scores = {}
    lengths_confidence = []
    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            preds = [[]] if task != 'deps' else [[], []]
            for _ in range(10):
                outputs = model(**batch)
                if hasattr(model, "module"):
                    outputs = model.module.decode(outputs)
                else:
                    outputs = model.decode(outputs)
                if task == 'deps':
                    if 'head_preds' in outputs:
                        preds[0].append(outputs['head_preds'])
                    if 'tag_preds' in outputs:
                        preds[1].append( outputs['tag_preds'])
                else:
                    preds[0].append(outputs[task + '_preds'])
            lengths = torch.tensor(([len(sent) for sent in outputs['words']]), device=args.device)
            if task == 'deps':
                preds[0] = torch.mean(torch.stack(preds[0]), axis=0)
                preds[1] = torch.mean(torch.stack(preds[1]), axis=0)
            else:
                preds[0] = torch.mean(torch.stack(preds[0]), axis=0)
            if task == 'deps':
                entropy = compute_dependency_entropy(preds[0], preds[1], lengths)
            else:
                entropy = compute_sequence_label_entropy(preds[0], lengths)
            confidence = 1 - entropy
            confidence = list(confidence.cpu().numpy())
            if task == 'deps':
                accuracy = compute_accuracy(outputs['head_indices'], outputs['predicted_heads'], lengths)
            else:
                accuracy = compute_accuracy(outputs[task + '_tags'], outputs['predicted_' + task], lengths)
            al_scores.update(dict(zip(outputs['sample_id'], entropy)))
            accuracy_scores.update(dict(zip(outputs['sample_id'], accuracy)))
            lengths = list(lengths.cpu().numpy())
            lengths_confidence += [(length_, confidence_) for length_, confidence_ in zip(lengths, confidence)]

    #check = [(np.mean([l_.item() for l_, e_ in sorted([(l, e/(l**div)) for (l,e) in lengths_entropy], key=lambda item: item[1], reverse=True)][:args.training_samples_to_add]), div) for div in [1, 0.75, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]]
    average_length = [(np.mean([l_.item() for l_, c_ in sorted([(l, c) for (l, c) in lengths_confidence], key=lambda item: item[1])][:args.training_samples_to_add]))]
    logger.info("------------- average_length -------")
    logger.info(average_length)
    compute_statistics(args, 'unlabeled', al_scores, accuracy_scores, al_iter_num)
    samples_ids = [k for k, v in sorted(al_scores.items(), key=lambda item: item[1])][:args.training_samples_to_add]
    sample_ids_file = os.path.join(args.output_dir, args.src_domain + '_sample_ids_dropout_entropy_' + task + '_iter_' + str(al_iter_num) + '.csv')
    with open(sample_ids_file, 'w', newline='') as myfile:
        myfile.write("\n".join([str(id) for id in samples_ids]))
    model.eval()
    return samples_ids

def active_learning_random(args, model, dataloader, al_iter_num):
    logger.info("***** Scoring Unlabeled Samples *****")
    logger.info("Num examples = %d", len(dataloader.dataset))
    model.eval()
    all_samples = []
    for batch in tqdm(dataloader, desc="Calculating Scores for Unlabeled samples"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            all_samples += [x['sample_id'] for x in batch['metadata']]
    np.random.shuffle(all_samples)
    samples_ids = all_samples[:args.training_samples_to_add]
    sample_ids_file = os.path.join(args.output_dir, args.src_domain + '_sample_ids_random_iter_' + str(al_iter_num) + '.csv')
    with open(sample_ids_file, 'w', newline='') as myfile:
        myfile.write("\n".join([str(id) for id in samples_ids]))
    return samples_ids

def compute_statistics(args, data_split, al_scores, accuracy_scores, step_num=None):
    all_ids = al_scores.keys()
    confidence = np.array([al_scores[id] for id in all_ids])
    accuracy = np.array([accuracy_scores[id] for id in all_ids])
    plot_accuracy_vs_confidence(args, data_split, confidence, accuracy, step_num)
    sentence_calibration_errors(args, data_split, confidence, accuracy, step_num)

def compute_accuracy(y_true, y_preds, lengths):
    assert len(y_true) == len(y_preds)
    accuracies = []
    for cur_y_true, cur_y_pred, cur_len in zip(y_true, y_preds, lengths):
        cur_y_true = cur_y_true[: int(cur_len.item())]
        cur_y_pred = cur_y_pred[: int(cur_len.item())]
        accuracy = accuracy_score(cur_y_true, cur_y_pred)
        accuracies.append(accuracy)
    return np.array(accuracies)

def get_file_to_save_name(args, data_split, ending_name, step_num=None):
    scale_temperature_str = '_tmp_scaling' if args.scale_temperature else ''
    task_for_scoring_str = '_task_for_scoring_' + args.task_for_scoring if args.task_for_scoring is not None else ''
    sub_folder = args.al_scoring + scale_temperature_str
    if not os.path.isdir(os.path.join(args.output_dir, sub_folder)):
        os.mkdir(os.path.join(args.output_dir, sub_folder))
    if step_num != None:
        sub_folder = os.path.join(sub_folder, str(step_num))
        if not os.path.isdir(os.path.join(args.output_dir, sub_folder)):
            os.mkdir(os.path.join(args.output_dir, sub_folder))
    file_to_save = os.path.join(args.output_dir, sub_folder, args.src_domain + '_' + data_split + '_' +
                                args.al_scoring + task_for_scoring_str + '_' + '_'.join(args.tasks) + '_' + ending_name)
    return file_to_save

def plot_accuracy_vs_confidence(args, data_split, confidence, accuracy, step_num=None):
    # Calculate the point density
    xy = np.vstack([confidence, accuracy])
    try:
        z = gaussian_kde(xy)(xy) / 100.
    except:
        z = np.ones_like(confidence)
    count_dict = {}
    for tup in zip(confidence, accuracy):
        if tup in count_dict:
            count_dict[tup] += 1
        else:
            count_dict[tup] = 1
    len_x = len(confidence)
    for k, v in count_dict.items():
        if v > 10:
            v += 500
        count_dict[k] = v / len_x
    #z = np.array([count_dict[tup] for tup in zip(x, y)])
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = confidence[idx], accuracy[idx], z[idx]

    fig, ax_joint = plt.subplots()
    sc = ax_joint.scatter(x, y, c=z, alpha=0.3, s=10, edgecolors=None, cmap=cm.Reds)

    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x, y)  # perform linear regression
    y_pred = linear_regressor.predict(x)  # make predictions
    ax_joint.plot(x, y_pred, color='black')
    ax_joint.set_ylim([0,1])
    r_square = r2_score(y, y_pred)
    ax_joint.set_title("R squared = %.2f" % r_square, fontsize=20)
    # Set labels on joint
    ax_joint.set_ylim((np.min(y), np.max(y)))
    ax_joint.set_xlabel('Confidence')
    ax_joint.set_ylabel('Accuracy')
    ax_joint.set_facecolor('xkcd:powder blue')
    plt.colorbar(sc)
    # plt.show()

    file_to_save = get_file_to_save_name(args, data_split, 'confidence_vs_accuracy_sentences.png', step_num)
    fig.savefig(file_to_save)
    plt.close()

    file_to_save_r_square = get_file_to_save_name(args, data_split, 'confidence_vs_accuracy_sentences_rsquare.csv', step_num)
    with open(file_to_save_r_square, 'w') as f:
        f.write("R square, %s\n" % (r_square))

def sentence_calibration_errors(args, data_split, confidence, accuracy, step_num=None):
    """
    Computes several metrics used to measure calibration error:
        - Expected Calibration Error (ECE): \sum_k |acc(k) - conf(k)| / n
        - Maximum Calibration Error (MCE): max_k |acc(k) - conf(k)|
        - Overconfidence Calibration Error (OCE): \sum_k conf(k) * max(conf(k) - acc(k), 0) / n
    """

    delta = abs(accuracy - confidence)
    expected_error = delta.mean()
    max_error = max(delta)
    oc_expected_error = (confidence * np.where(confidence > accuracy, confidence - accuracy, 0)).mean()

    file_to_save = get_file_to_save_name(args, data_split, 'calibration_errors.csv', step_num)
    with open(file_to_save, 'w') as f:
        f.write("expected_error, %s\n" % (expected_error))
        f.write("max_error, %s\n" % (max_error))
        f.write("overconfident_expected_error, %s\n" % (oc_expected_error))


# ner entropy / parser entropy / sum of entropies
# two single classifiers with sum of entropies
# how many ids are present in both multi-task  and single-task
# multi-task with two classifiers / multi-task with joint model
# cross active learning individual

# add number of tokens per model / number of sentences (two graphs)

# compute accuracy vs confidence for droput-droput and model-dropout