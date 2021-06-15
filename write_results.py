import csv
import os
from collections import OrderedDict

dataset = 'ontonotes'
domains = ['bc', 'bn', 'mz', 'nw', 'tc', 'wb']
pretrained_model = 'bert-base-cased_unfreeze'
data_split = 'test'
deps_models = ['train_percentage_' + train_percentage
             + '_deps_' + pretrained_model for train_percentage in ['0.05', '0.1']]
ner_models = ['train_percentage_' + train_percentage
             + '_ner_' + pretrained_model for train_percentage in ['0.05', '0.1']]
multitask_models = ['train_percentage_' + train_percentage
             + '_deps_ner_' + pretrained_model + '_4_layers' for train_percentage in ['0.05', '0.1']]
old_multitask_models = ['train_percentage_' + train_percentage
             + '_deps_ner_' + pretrained_model for train_percentage in ['0.05', '0.1']]
all_models = deps_models + ner_models + multitask_models + old_multitask_models

res_dict = OrderedDict()
for domain in domains:
    domain_dict = OrderedDict()
    for task in ['deps', 'ner']:
        for train_percentage in ['0.05', '0.1']:
            for model in all_models:
                model_name = dataset + '_' + domain + '_' + model
                if not (train_percentage in model_name and task in model_name):
                    continue
                res_file = os.path.join('saved_models', model_name, domain + '_res_' + data_split + '.conllu')
                uas_score = ""
                ner_score = ""
                if os.path.exists(res_file):
                    with open(res_file, 'r') as f:
                        for line in f.readlines():
                            line_ = line.split()
                            if line_[0] == 'UAS':
                                uas_score = line_[6]
                            if line_[0] == 'NER':
                                ner_score = line_[6]
                else:
                    print(res_file + " doesn't exist")
                if model_name not in domain_dict:
                    domain_dict[model_name] = {}
                domain_dict[model_name][task] = uas_score if task == 'deps' else ner_score
            res_dict[domain] = domain_dict

csv_file = 'results.csv'
with open(csv_file, 'w') as f:
    csv_writer = csv.writer(f)
    # csv_writer.writerow(['Domain'] + [subitem for item in [[model, model] for model in all_models]
    #                                   for subitem in item])
    for score_name in ['UAS', 'F1']:
        task = 'deps' if score_name == 'UAS' else 'ner'
        csv_writer.writerow(['Domain'] + [model for model in all_models if task in model])
        csv_writer.writerow([""] + [score_name] * len(all_models))
        models = []
        for domain, domain_dict in res_dict.items():
            row = [domain]
            for model in all_models:
                if task not in model:
                    continue
                model_name = dataset + '_' + domain + '_' + model
                row_score = domain_dict[model_name]
                if task in row_score:
                    models.append(model)
                    row += [row_score[task]]
            csv_writer.writerow(row)
        csv_writer.writerow(['\n'])