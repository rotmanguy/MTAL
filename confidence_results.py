from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)


# only single-task
# only multi-task
# single-task vs multi-task
# - ls vs + ls
# task 2 on task 1 vs task 1 on task 1
# Evaluation on Num tokens

def get_calibration_errors(file_name):
    calibration_errors = {}
    with open(file_name, 'r') as f:
        for line in f:
            key, value = line.strip().split(', ')
            calibration_errors[key] = value
    return calibration_errors

def get_r_square(file_name):
    with open(file_name, 'r') as f:
        for line in f:
            if 'R square' in line:
                _, r_square = line.strip().split(', ')
                break
    return r_square

dataset = 'ontonotes'
train_percentage = '0.02'
label_smoothing_list = [True, False]
plots_dir = 'plots/st_confidence_plots'

if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

domains = ['bc', 'bn', 'mz', 'nw', 'tc', 'wb']
pretrained_model = 'bert-base-cased_unfreeze'
tasks_list = [("deps",), ("ner", ), ("deps", "ner")]
tasks_list = [("deps",), ("ner", )]
# tasks_list = [("deps", "ner")]
task_for_scoring = ['ner', 'deps']
graph_types = ["Num_Tokens", "Num_Sentences"]
all_models = []
for tasks in tasks_list:
    if len(tasks) > 1:
        al_scoring_list = ['joint_entropy', 'entropy', 'dropout_agreement', 'dropout_joint_agreement']
        #al_scoring_list = ['joint_entropy', 'dropout_joint_agreement']
    else:
        al_scoring_list = ['entropy', 'dropout_agreement']
    for al_scoring in al_scoring_list:
        if (al_scoring == 'entropy' or al_scoring == 'dropout_agreement') and len(tasks) > 1:
            task_for_scoring_list = tasks
        else:
            task_for_scoring_list = (None,)
        for task_for_scoring in task_for_scoring_list:
            model = '_'.join(tasks) + '_al_scoring_' + al_scoring
            if task_for_scoring is not None:
                model += '_by_' + task_for_scoring
            all_models.append(model)

num_iters = 5

for graph_type in graph_types:
    res_dict = OrderedDict()
    for domain in domains:
        domain_dict = OrderedDict()
        for task in ['deps', 'ner']:
            domain_dict[task] = OrderedDict()
            for model in all_models:
                if task not in model or 'random' in model:
                    continue
                for label_smoothing in label_smoothing_list:
                    if label_smoothing:
                        model_name = dataset + '_' + domain + '_active_learning_train_percentage_' +\
                                 train_percentage + '_' + model + '_ls_0.2_' + pretrained_model
                    else:
                        model_name = dataset + '_' + domain + '_active_learning_train_percentage_' + \
                                     train_percentage + '_' + model + '_' + pretrained_model
                    x_axis = []
                    y_axis = []
                    metric = "LAS" if task == 'deps' else "NER"
                    for iter in range(num_iters):
                        x = -1
                        y = -1
                        train_res_file = os.path.join('saved_models', model_name, domain + '_res_train_iter_' + str(iter) + '.conllu')
                        if os.path.exists(train_res_file):
                            with open(train_res_file, 'r') as f:
                                for line in f.readlines():
                                    line_ = line.split()
                                    if graph_type in line_[0]:
                                        if line_[5] != '|':
                                            x = int(float(line_[5]))
                                        else:
                                            x = int(float(line_[6]))
                        al_scoring = model.split('al_scoring_')[1]
                        confidence_dir = os.path.join('saved_models', model_name, al_scoring.split('_by_')[0], str(iter))

                        calibration_error_file = [os.path.join(confidence_dir, filename) for filename in os.listdir(confidence_dir)
                                                  if filename.endswith("calibration_errors.csv")][0]
                        calibration_errors = get_calibration_errors(calibration_error_file)
                        y = float(calibration_errors['overconfident_expected_error'])

                        r_square_file = [os.path.join(confidence_dir, filename) for filename in os.listdir(confidence_dir)
                                                  if filename.endswith("rsqaure.csv")][0]
                        r_square = get_r_square(r_square_file)
                        if x == -1 or y == -1:
                            print(train_res_file + " or " + calibration_error_file + " doesn't exist")
                        else:
                            x_axis.append(x)
                            y_axis.append(y)
                    if 'random' in model_name:
                        model_name = 'random'
                    if len(x_axis) == num_iters and len(y_axis) == num_iters:
                        model_ = model + '_with_LS' if label_smoothing else model
                        domain_dict[task][model_] = (np.array(x_axis), np.array(y_axis))
                res_dict[domain] = domain_dict

    for domain in domains:
        for task in ['deps', 'ner']:
            plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
            print(domain + '_' + task)
            markers = ['D', 'o', 'x', '^', '+', 'H', 'p', '.', '1', '2', '3', '4', '8', 's', 'd', 'h']
            m_idx = 0
            c_idx = 0
            for model, model_dict in res_dict[domain][task].items():
                marker = markers[m_idx]
                color = 'C' + str(c_idx)
                m_idx += 1
                c_idx += 1
                plt.plot(model_dict[0], model_dict[1], c=color, marker=marker, label=model)
            plt.xlabel(graph_type)
            plt.ylabel("Overconfident expected error")
            plt.legend(loc="upper right")
            plt.title(domain + '_' + task)
            #plt.show()
            plt.savefig(os.path.join(plots_dir, graph_type + '_' + domain + '_' + task + '.png'))
            plt.close()


# decompose graphs according to research questions
# average graphs