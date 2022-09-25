"""
A collection of handy utilities
"""
import os
from argparse import Namespace
from copy import deepcopy
from typing import Dict, Any, List
import logging

import allennlp
import numpy as np
import torch
from tqdm import tqdm

from io_.io_utils import clean_batch
from utils.conll18_ud_eval import evaluate, load_conllu_file, log_results
from utils.seqeval import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils.utils import save_checkpoint

logger = logging.getLogger('mylog')

def geo_mean_overflow(arr):
    log_arr = np.log(arr)
    return np.exp(log_arr.sum()/len(log_arr))


def sanitize(x: Any) -> Any:
    """
    Sanitize turns PyTorch and Numpy types into basic Python types, so they
    can be serialized into JSON.
    """
    if isinstance(x, (str, float, int, bool)):
        # x is already serializable
        return x
    elif isinstance(x, torch.Tensor):
        # tensor needs to be converted to a list (and moved to cpu if necessary)
        return x.cpu().tolist()
    elif isinstance(x, np.ndarray):
        # array needs to be converted to a list
        return x.tolist()
    elif isinstance(x, np.number):  # pylint: disable=no-member
        # NumPy numbers need to be converted to Python numbers
        return x.item()
    elif isinstance(x, dict):
        # Dicts need their values sanitized
        return {key: sanitize(value) for key, value in x.items()}
    elif isinstance(x, allennlp.data.Token):
        # Tokens get sanitized to just their text.
        return x.text
    elif isinstance(x, (list, tuple)):
        # Lists and Tuples need their values sanitized
        return [sanitize(x_i) for x_i in x]
    elif x is None:
        return "None"
    elif hasattr(x, 'to_json'):
        return x.to_json()
    else:
        raise ValueError(f"Cannot sanitize {x} of type {type(x)}. "
                         "If this is your own custom class, add a `to_json(self)` method "
                         "that returns a JSON-like object.")

def create_output_lines(lines: str, outputs: Dict) -> List[str]:
    multiword_map = None
    if outputs["multiword_ids"]:
        multiword_ids = [[id_] + [int(x) for x in id_.split("-")] for id_ in outputs["multiword_ids"]]
        multiword_forms = outputs["multiword_forms"]
        multiword_map = {start: (id_, form) for (id_, start, end), form in zip(multiword_ids, multiword_forms)}
    output_lines = []
    for i, line in enumerate(lines):
        line_ = [str(l) for l in line[:-1]]

        # Handle multiword tokens
        if multiword_map and i + 1 in multiword_map:
            id_, form = multiword_map[i + 1]
            row = f"{id_}\t{form}" + "".join(["\t_"] * 8)
            output_lines.append(row)

        row = "\t".join(line_) + "".join(["\t_"] * 2) + "\t" + str(line[-1])
        output_lines.append(row)
    return output_lines

def convert_dict_prediction_to_str(args: Namespace, outputs: Dict) -> str:
    # Converting predictions into a string, so that we could write them to the disk
    word_count = len([word for word in outputs["words"]])
    lines = zip(*[outputs[k] if (k in outputs and outputs[k] != "None") else ["_"] * word_count
                  for k in ["ids",
                            "words",
                            "predicted_lemmas" if "lemmas" in args.tasks else "lemmas",
                            "predicted_upos" if "upos" in args.tasks else "upos_tags",
                            "predicted_xpos" if "xpos" in args.tasks else "xpos_tags",
                            "predicted_feats" if "feats" in args.tasks else "feats",
                            "predicted_heads" if "deps" in args.tasks else "head_indices",
                            "predicted_head_tags" if "deps" in args.tasks else "head_tags",
                            "predicted_ner" if "ner" in args.tasks else "ner_tags"
                            ]])
    output_lines = create_output_lines(lines, outputs)
    return "\n".join(output_lines) + "\n\n"

def convert_dict_labels_to_str(outputs: Dict) -> str:
    # Converting gold labels into a string, so that we could write them to the disk
    word_count = len([word for word in outputs["words"]])
    lines = zip(*[outputs[k] if (k in outputs and outputs[k] != "None") else ["_"] * word_count
                  for k in ["ids",
                            "words",
                            "lemmas",
                            "upos_tags",
                            "xpos_tags",
                            "feats",
                            "head_indices",
                            "head_tags",
                            "ner_tags"
                            ]])
    output_lines = create_output_lines(lines, outputs)
    return "\n".join(output_lines) + "\n\n"

def load_gold_file(args, gold_filename, data_split):
    # Loading the gold (labeled) file from disk
    if type(gold_filename) in [list, tuple]:
        tmp_gold_filename = os.path.join(args.output_dir, 'gold_' + data_split + '.conllu')
        with open(tmp_gold_filename, 'w') as outfile:
            for fname in gold_filename:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
        gold_file = load_conllu_file(tmp_gold_filename)
        os.remove(tmp_gold_filename)
    else:
        gold_file = load_conllu_file(gold_filename)
    return gold_file

def predict_and_evaluate(args, model, dataloader, data_split, global_step, domain, final_eval=False):
    # Predict and Eval!
    logger.info("predict and eval global step %d" % global_step)
    logger.info("***** Running evaluation %s *****", data_split)
    logger.info("  Num examples = %d", len(dataloader.dataset))

    # Loading the gold (labeled) file
    dataset_path_exists = False
    if dataloader.dataset.dataset_path is not None:
        gold_filename = dataloader.dataset.dataset_path
        logger.info("gold file path: %s", gold_filename)
        dataset_path_exists = True
    else:
        gold_filename = os.path.join(args.output_dir, domain + '_' + data_split + '.conllu')
        gold_file = open(gold_filename, "w")

    # Creating the predicted file
    tmp_pred_filename = os.path.join(args.output_dir, domain + '_pred_' + data_split + '.conllu')
    tmp_pred_file = open(tmp_pred_filename, "w")

    model.eval()
    extra_labels_gold = {}
    extra_labels_predict = {}
    extra_evaluation = {}

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            # Predict
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)

            # Gather main predictions (and labels if not dataset_path_exists)
            outputs = sanitize(outputs)
            batch_size = len(outputs['words'])
            predictions = []
            if not dataset_path_exists:
                labels = []
            for i in range(batch_size):
                output = {}
                for k, v in outputs.items():
                    if isinstance(v, list):
                        output[k] = v[i]
                prediction = convert_dict_prediction_to_str(args, output)
                predictions.append(prediction)
                if not dataset_path_exists:
                    label = convert_dict_labels_to_str(output)
                    labels.append(label)
            # Writing predicted lines to disk
            tmp_pred_file.writelines(predictions)
            # Writing gold lines to disk if not dataset_path_exists
            if not dataset_path_exists:
                gold_file.writelines(labels)

            ## Gathering gold and predicted labels for the additional tasks
            for task in args.tasks:
                if task in ["deps", "UFeats", "AllTags", "Lemmas", "CLAS", "MLAS", "BLEX"]:
                    continue
                task_upper = task.upper()
                if task_upper not in extra_labels_gold:
                    extra_labels_gold[task_upper] = []
                if task_upper not in extra_labels_predict:
                    extra_labels_predict[task_upper] = []
                if task_upper not in extra_evaluation:
                    extra_evaluation[task_upper] = {'loss': 0.0}

                sent_lengths = [len(x) for x in outputs['words']]
                extra_labels_gold[task_upper] += [x[:sent_lengths[i]] for i, x in enumerate(outputs[task + '_tags'])]
                extra_labels_predict[task_upper] += [x[:sent_lengths[i]] for i, x in enumerate(outputs['predicted_' + task])]
                extra_evaluation[task_upper]['loss'] += outputs["loss_dict"][task]

    # Done writing predicted file
    tmp_pred_file.close()

    # Done writing gold file
    if not dataset_path_exists:
        gold_file.close()

    # Writing classification report
    for task in extra_labels_gold.keys():
        task_upper = task.upper()
        extra_evaluation[task_upper]['aligned_accuracy'] = accuracy_score(extra_labels_gold[task_upper], extra_labels_predict[task_upper])
        extra_evaluation[task_upper]['precision'] = precision_score(extra_labels_gold[task_upper], extra_labels_predict[task_upper])
        extra_evaluation[task_upper]['recall'] = recall_score(extra_labels_gold[task_upper], extra_labels_predict[task_upper])
        extra_evaluation[task_upper]['f1'] = f1_score(extra_labels_gold[task_upper], extra_labels_predict[task_upper])
        cls_report = classification_report(extra_labels_gold[task_upper], extra_labels_predict[task_upper], digits=2)
        with open(os.path.join(args.output_dir, domain + '_' + task + '_res_' + data_split + '.conllu'), "w") as text_file:
            text_file.write(cls_report)

    # Eval by comparing the gold vs pred file
    tmp_pred_file = load_conllu_file(tmp_pred_filename)
    gold_file = load_gold_file(args, gold_filename, data_split)
    evaluation = evaluate(gold_file, tmp_pred_file)

    # Update the evaluation dictionary
    evaluation.update(extra_evaluation)
    evaluation['global_step'] = global_step

    # Log evaluation
    log_results(evaluation)

    # Remove temporary file
    if not final_eval:
        os.remove(tmp_pred_filename)
    if not dataset_path_exists:
        os.remove(gold_filename)

    return evaluation

def main_evaluation(args, dataloaders, model, optimizer, scheduler, amp, dev_eval_dict, patient, global_step, domain, final_eval=False):
    # Perform prediction and evaluation
    curr_dev_eval_dict = predict_and_evaluate(args, model, dataloaders['dev'], 'dev', global_step, domain, final_eval)

    # Checking if current model is the best one
    if not bool(dev_eval_dict):
        is_best = True # if dict is empty then set to True
    else:
        # Computing geometric average over task metrics
        dev_eval_metrics = []
        curr_dev_eval_metrics = []
        for task in args.tasks:
            if task == 'deps':
                metric_name = 'LAS'
            else:
                metric_name = task.upper()
            metric = dev_eval_dict[metric_name]['f1'] if dev_eval_dict[metric_name]['f1'] != 0.0 else 1e-8
            cur_metric = curr_dev_eval_dict[metric_name]['f1'] if curr_dev_eval_dict[metric_name]['f1'] != 0.0 else 1e-8
            dev_eval_metrics.append(metric)
            curr_dev_eval_metrics.append(cur_metric)
        is_best = geo_mean_overflow(dev_eval_metrics) <= geo_mean_overflow(curr_dev_eval_metrics)

    if is_best:
        logger.info('New best score on step: %d. Saving model.' % curr_dev_eval_dict['global_step'])
        dev_eval_dict = deepcopy(curr_dev_eval_dict)
        model_path = os.path.join(args.output_dir, args.src_domain + '.tar.gz')
        save_checkpoint(model, optimizer, amp, scheduler, dev_eval_dict, model_path)
        patient = 0
    else:
        patient += 1
        logger.info('Score did not improve on step: %d. '
              'Best score so far was achieved in step: %d' % (global_step, dev_eval_dict['global_step']))

    logger.info('\n')
    return dev_eval_dict, patient
