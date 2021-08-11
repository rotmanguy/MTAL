from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)


# only single-task
# only multi-task
# single-task vs multi-task
# - ls vs + ls
# task 2 on task 1 vs task 1 on task 1
# Evaluation on Num tokens

dataset = 'ontonotes'
train_percentage = '0.02'
label_smoothing = True
plots_dir = 'plots/plots_entropy_ls'

if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

domains = ['bc', 'bn', 'mz', 'nw', 'tc', 'wb']
pretrained_model = 'bert-base-cased_unfreeze'
tasks_list = [("deps",), ("ner", ), ("deps", "ner")]
task_for_scoring = ['ner', 'deps']
task_to_name = {'ner': 'ner', 'deps': 'dp'}
graph_types = ["Num_Tokens", "Num_Sentences"]
graph_type_to_label = {"Num_Tokens": 'Number of Tokens', "Num_Sentences": 'Number of Sentences'}
all_models = []
for tasks in tasks_list:
    if len(tasks) > 1:
        #al_scoring_list = ['random', 'entropy', 'dropout_agreement', 'joint_entropy', 'dropout_joint_agreement']
        al_scoring_list = ['random', 'entropy', 'joint_entropy', 'joint_min_entropy']
    else:
        #al_scoring_list = ['random', 'entropy', 'dropout_agreement']
        al_scoring_list = ['random', 'entropy', 'dropout_agreement']
    for al_scoring in al_scoring_list:
        if (al_scoring == 'entropy' or al_scoring == 'dropout_agreement') and len(tasks) > 1:
            task_for_scoring_list = tasks
        else:
            task_for_scoring_list = (None, )
        for task_for_scoring in task_for_scoring_list:
            model = '_'.join(tasks) + '_al_scoring_' + al_scoring
            if task_for_scoring is not None:
                model += '_by_' + task_for_scoring
            all_models.append(model)

model_to_label = {
    'deps_al_scoring_random': 'ST-R',
    'deps_al_scoring_entropy': 'ST-EC-DP',
    'deps_al_scoring_dropout_agreement': 'ST-DA-DP',
    'ner_al_scoring_random': 'ST-R',
    'ner_al_scoring_entropy': 'ST-EC-NER',
    'ner_al_scoring_dropout_agreement': 'ST-DA-NER',
    'deps_ner_al_scoring_random': 'MT-R',
    'deps_ner_al_scoring_joint_entropy': 'MT-JEC',
    'deps_ner_al_scoring_entropy_by_deps': 'MT-EC-DP',
    'deps_ner_al_scoring_entropy_by_ner': 'MT-EC-NER',
    'deps_ner_al_scoring_joint_min_entropy': 'MT-JMIN'
}

num_iters = 5

for graph_type in graph_types:
    res_dict = OrderedDict()
    for domain in domains:
        domain_dict = OrderedDict()
        for task in ['deps', 'ner']:
            domain_dict[task] = OrderedDict()
            for model in all_models:
                if task not in model:
                    continue
                if model == 'deps_ner_al_scoring_entropy_by_ner' and task == 'deps':
                    continue
                if model == 'deps_ner_al_scoring_entropy_by_deps' and task == 'ner':
                    continue
                if model == 'deps_ner_al_scoring_joint_min_entropy' and task == 'deps':
                    continue
                if model == 'deps_ner_al_scoring_joint_entropy' and task == 'ner':
                    continue
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
                    test_res_file = os.path.join('saved_models', model_name, domain + '_res_test_iter_' + str(iter) + '.conllu')
                    if os.path.exists(test_res_file):
                        with open(test_res_file, 'r') as f:
                            for line in f.readlines():
                                line_ = line.split()
                                if line_[0] == metric:
                                    y = float(line_[6])

                    if x == -1 or y == -1:
                        print(train_res_file + " or " + test_res_file + " doesn't exist")
                    else:
                        x_axis.append(x)
                        y_axis.append(y)
                if 'random' in model_name:
                    model_name = 'random'
                if len(x_axis) == num_iters and len(y_axis) == num_iters:
                    domain_dict[task][model] = (np.array(x_axis), np.array(y_axis))
            res_dict[domain] = domain_dict

    for domain in domains:
        for task in ['deps', 'ner']:
            plt.figure(num=None, figsize=(7, 7), dpi=300, facecolor='w', edgecolor='k')
            print(domain + '_' + task)
            markers = ['D', 'o', 'x', '^', '+', 'H', 'p', '.', '1', '2']
            m_idx = 0
            c_idx = 0
            for model, model_dict in res_dict[domain][task].items():
                marker = markers[m_idx]
                color = 'C' + str(c_idx)
                m_idx += 1
                c_idx += 1
                plt.plot(model_dict[0], model_dict[1], c=color, marker=marker, label=model_to_label[model],
                         linewidth=3.0)
            plt.xlabel(graph_type_to_label[graph_type], fontsize=20, fontweight='bold')
            if task == 'deps':
                plt.ylabel("LAS", fontsize=18, fontweight='bold')
            else:
                plt.ylabel("F1", fontsize=18, fontweight='bold')
            plt.legend(loc="lower right", prop={'size': 17})
            plt.title(domain.upper() + ' - ' + task_to_name[task].upper(), fontsize=20, fontweight='bold')
            plt.show()
            plt.savefig(os.path.join(plots_dir, graph_type + '_' + domain + '_' + task + '.png'))
            plt.close()

# decompose graphs according to research questions
# average graphs
