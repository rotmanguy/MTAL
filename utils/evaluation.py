"""
A collection of handy utilities
"""
import os
from argparse import Namespace
from copy import deepcopy
from typing import Dict, Any
import logging

import allennlp
import numpy as np
import torch
from tqdm import tqdm

from io_.io_utils import clean_batch
from utils.conll18_ud_eval import evaluate, load_conllu_file, write_results_file, print_results
from utils.seqeval import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils.utils import save_checkpoint

logger = logging.getLogger('mylog')

def geo_mean_overflow(arr):
    log_arr = np.log(arr)
    return np.exp(log_arr.sum()/len(log_arr))


def sanitize(x: Any) -> Any:  # pylint: disable=invalid-name,too-many-return-statements
    """
    Sanitize turns PyTorch and Numpy types into basic Python types so they
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

def convert_dict_prediction_to_str(args: Namespace, outputs: Dict) -> str:
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

    multiword_map = None
    if outputs["multiword_ids"]:
        multiword_ids = [[id] + [int(x) for x in id.split("-")] for id in outputs["multiword_ids"]]
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
    return "\n".join(output_lines) + "\n\n"

def convert_dict_labels_to_str(outputs: Dict) -> str:
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

    multiword_map = None
    if outputs["multiword_ids"]:
        multiword_ids = [[id] + [int(x) for x in id.split("-")] for id in outputs["multiword_ids"]]
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
    return "\n".join(output_lines) + "\n\n"

def load_gold_file(args, gold_filename, data_split):
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
    # Eval!
    logger.info("predict and eval global step %d" % global_step)
    logger.info("***** Running evaluation %s *****", data_split)
    logger.info("  Num examples = %d", len(dataloader.dataset))
    dataset_path_exists = False
    if dataloader.dataset.dataset_path is not None:
        gold_filename = dataloader.dataset.dataset_path
        logger.info("gold file path: %s", gold_filename)
        dataset_path_exists = True
        # if final_eval:
        #     gold_filename_below_10 = gold_filename.replace(args.dataset, args.dataset + '_below_10')
        #     gold_filename_above_10 = gold_filename.replace(args.dataset, args.dataset + '_above_10')
    else:
        gold_filename = os.path.join(args.output_dir, domain + '_' + data_split + '.conllu')
        gold_file = open(gold_filename, "w")
        # if final_eval:
        #     gold_filename_below_10 = os.path.join(args.output_dir, domain + '_' + data_split + '_below_10' + '.conllu')
        #     gold_file_below_10 = open(gold_filename_below_10, "w")
        #     gold_filename_above_10 = os.path.join(args.output_dir, domain + '_' + data_split + '_above_10' + '.conllu')
        #     gold_file_above_10 = open(gold_filename_above_10, "w")

    if domain != args.src_domain:
        domain_results_dir = os.path.join(args.output_dir, 'tgt_domains', domain)
        if not os.path.isdir(domain_results_dir):
            os.makedirs(domain_results_dir)
    else:
        domain_results_dir = args.output_dir

    tmp_pred_filename = os.path.join(domain_results_dir, domain + '_pred_' + data_split + '.conllu')
    tmp_pred_file = open(tmp_pred_filename, "w")
    # if final_eval:
    #     tmp_pred_filename_below_10 = os.path.join(domain_results_dir, domain + '_pred_' + data_split + '_below_10.conllu')
    #     tmp_pred_file_below_10 = open(tmp_pred_filename_below_10, "w")
    #     tmp_pred_filename_above_10 = os.path.join(domain_results_dir, domain + '_pred_' + data_split + '_above_10.conllu')
    #     tmp_pred_file_above_10 = open(tmp_pred_filename_above_10, "w")
    # else:
    #     tmp_pred_filename = os.path.join(args.output_dir, domain + '_pred_' + data_split + '.conllu')
    #     tmp_pred_file = open(tmp_pred_filename, "w")
    #     if final_eval:
    #         tmp_pred_filename_below_10 = os.path.join(args.output_dir, domain + '_pred_' + data_split + '_below_10.conllu')
    #         tmp_pred_file_below_10 = open(tmp_pred_filename_below_10, "w")
    #         tmp_pred_filename_above_10 = os.path.join(args.output_dir, domain + '_pred_' + data_split + '_above_10.conllu')
    #         tmp_pred_file_above_10 = open(tmp_pred_filename_above_10, "w")

    model.eval()
    extra_labels_gold = {}
    extra_labels_predict = {}
    extra_evaluation = {}
    # if final_eval:
    #     extra_evaluation_below_10 = {}
    #     extra_evaluation_above_10 = {}
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = clean_batch(args, batch, mode='eval')
        with torch.no_grad():
            outputs = model(**batch)
            if hasattr(model, "module"):
                outputs = model.module.decode(outputs)
            else:
                outputs = model.decode(outputs)
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
            tmp_pred_file.writelines(predictions)
            # if final_eval:
            #     lengths = [len(x['words']) for x in batch['metadata']]
            #     tmp_pred_file_below_10.writelines([predictions[len_idx] for len_idx, length in enumerate(lengths) if length <= 10])
            #     tmp_pred_file_above_10.writelines([predictions[len_idx] for len_idx, length in enumerate(lengths) if length > 10])
            if not dataset_path_exists:
                gold_file.writelines(labels)
                # if final_eval:
                #     gold_file_below_10.writelines([labels[len_idx] for len_idx, length in enumerate(lengths) if length <= 10])
                #     gold_file_above_10.writelines([labels[len_idx] for len_idx, length in enumerate(lengths) if length > 10])

            ## gathering gold and predicted labels for the additional tasks
            for k in args.tasks:
                if k == 'deps':
                    continue
                if k not in extra_labels_gold:
                    extra_labels_gold[k] = []
                if k not in extra_labels_predict:
                    extra_labels_predict[k] = []
                if k not in extra_evaluation:
                    extra_evaluation[k] = {'loss': 0.0}

                sent_lengths = [len(x) for x in outputs['words']]
                extra_labels_gold[k] += [x[:sent_lengths[i]] for i, x in enumerate(outputs[k + '_tags'])]
                extra_labels_predict[k] += [x[:sent_lengths[i]] for i, x in enumerate(outputs['predicted_' + k])]
                extra_evaluation[k]['loss'] += outputs["loss_dict"][k]
                # if final_eval:
                #     extra_evaluation_below_10[k] = {'loss': extra_evaluation[k]['loss']}
                #     extra_evaluation_above_10[k] = {'loss': extra_evaluation[k]['loss']}

    tmp_pred_file.close()
    # if final_eval:
    #     tmp_pred_file_below_10.close()
    #     tmp_pred_file_above_10.close()
    if not dataset_path_exists:
        gold_file.close()
        # if final_eval:
        #     gold_file_below_10.close()
        #     gold_file_above_10.close()

    for k in extra_labels_gold.keys():
        extra_evaluation[k]['aligned_accuracy'] = accuracy_score(extra_labels_gold[k], extra_labels_predict[k])
        extra_evaluation[k]['precision'] = precision_score(extra_labels_gold[k], extra_labels_predict[k])
        extra_evaluation[k]['recall'] = recall_score(extra_labels_gold[k], extra_labels_predict[k])
        extra_evaluation[k]['f1'] = f1_score(extra_labels_gold[k], extra_labels_predict[k])
        cls_report = classification_report(extra_labels_gold[k], extra_labels_predict[k], digits=2)
        with open(os.path.join(domain_results_dir, domain + '_' + k + '_res_' + data_split + '.conllu'), "w") as text_file:
            text_file.write(cls_report)

        # if final_eval:
        #     extra_labels_predict_below_10 = [label for label in extra_labels_predict[k] if len(label) <= 10]
        #     extra_labels_predict_above_10 = [label for label in extra_labels_predict[k] if len(label) > 10]
        #     extra_labels_gold_below_10 = [label for label in extra_labels_gold[k] if len(label) <= 10]
        #     extra_labels_gold_above_10 = [label for label in extra_labels_gold[k] if len(label) > 10]
        #
        #     extra_evaluation_below_10[k]['aligned_accuracy'] = accuracy_score(extra_labels_gold_below_10, extra_labels_predict_below_10)
        #     extra_evaluation_below_10[k]['precision'] = precision_score(extra_labels_gold_below_10, extra_labels_predict_below_10)
        #     extra_evaluation_below_10[k]['recall'] = recall_score(extra_labels_gold_below_10, extra_labels_predict_below_10)
        #     extra_evaluation_below_10[k]['f1'] = f1_score(extra_labels_gold_below_10, extra_labels_predict_below_10)
        #     extra_evaluation_above_10[k]['aligned_accuracy'] = accuracy_score(extra_labels_gold_above_10, extra_labels_predict_above_10)
        #     extra_evaluation_above_10[k]['precision'] = precision_score(extra_labels_gold_above_10, extra_labels_predict_above_10)
        #     extra_evaluation_above_10[k]['recall'] = recall_score(extra_labels_gold_above_10, extra_labels_predict_above_10)
        #     extra_evaluation_above_10[k]['f1'] = f1_score(extra_labels_gold_above_10, extra_labels_predict_above_10)

    tmp_pred_file = load_conllu_file(tmp_pred_filename)
    gold_file = load_gold_file(args, gold_filename, data_split)
    evaluation = evaluate(gold_file, tmp_pred_file)
    evaluation.update(extra_evaluation)
    evaluation['global_step'] = global_step

    # if final_eval:
    #     tmp_pred_file_below_10 = load_conllu_file(tmp_pred_filename_below_10)
    #     tmp_pred_file_above_10 = load_conllu_file(tmp_pred_filename_above_10)
    #     gold_file_below_10 = load_gold_file(args, gold_filename_below_10, data_split)
    #     gold_file_above_10 = load_gold_file(args, gold_filename_above_10, data_split)
    #     evaluation_below_10 = evaluate(tmp_pred_file_below_10, gold_file_below_10)
    #     evaluation_below_10.update(extra_evaluation_below_10)
    #     evaluation_above_10 = evaluate(tmp_pred_file_above_10, gold_file_above_10)
    #     evaluation_above_10.update(extra_evaluation_above_10)
    #     res_filename_below_10 = os.path.join(domain_results_dir, 'eval_' + data_split + '_below_10' + '.csv')
    #     write_results_file(evaluation_below_10, res_filename_below_10)
    #     res_filename_above_10 = os.path.join(domain_results_dir, 'eval_' + data_split + '_above_10' + '.csv')
    #     write_results_file(evaluation_above_10, res_filename_above_10)

    if not final_eval:
        os.remove(tmp_pred_filename)
    if not dataset_path_exists:
        os.remove(gold_filename)
        # if final_eval:
        #     os.remove(gold_filename_below_10)
        #     os.remove(gold_filename_above_10)
    print_results(evaluation)
    return evaluation

def main_evaluation(args, dataloaders, model, optimizer, scheduler, amp, dev_eval_dict, patient, global_step, domain, final_eval=False):
    curr_dev_eval_dict = predict_and_evaluate(args, model, dataloaders['dev'], 'dev', global_step, domain, final_eval)
    if domain != args.src_domain:
        return
    # check if dict is empty
    if not bool(dev_eval_dict):
        is_best = True
    else:
        dev_eval_metrics = []
        curr_dev_eval_metrics = []
        for task in args.tasks:
            if task == 'deps':
                metric_name = 'LAS'
            else:
                metric_name = task
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

    # if perform_final_eval:
    #     # load best model
    #     model_path = os.path.join(args.output_dir, args.domain + '.tar.gz')
    #     if os.path.exists(model_path):
    #         print('Loading saved model from: %s' % model_path)
    #         checkpoint = torch.load(model_path, map_location=args.device)
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         if amp is not None:
    #             amp.load_state_dict(checkpoint['amp_state_dict'])
    #         global_step = checkpoint['dev_eval_dict']['global_step']
    #     else:
    #         print('We did not find a saved model. Evaluating with current model')
    #     # save in-domain checkpoint
    #     if args.training_set_size is not None:
    #         splits_to_write = dataloaders.keys()
    #     else:
    #         splits_to_write = ['dev', 'test']
    #     for split in splits_to_write:
    #         eval_dict = predict_and_evaluate(args, model, dataloaders[split], split, global_step)
    #         res_filename = os.path.join(args.output_dir, 'res_' + split + '.conllu')
    #         write_results_file(eval_dict, res_filename)
    print('\n')
    return dev_eval_dict, patient
