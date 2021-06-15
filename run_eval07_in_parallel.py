import subprocess

train_percentage = '0.02'
label_smoothing_list = [False, True]

dataset = 'ontonotes'
domains = ['bc', 'bn', 'mz', 'nw', 'tc', 'wb']
pretrained_model = 'bert-base-cased_unfreeze'
tasks_list = [("deps",), ("deps", "ner")]
task_for_scoring = ['ner', 'deps']
all_models = []
for tasks in tasks_list:
    if len(tasks) > 1:
        al_scoring_list = ['joint_entropy', 'entropy', 'random', 'dropout_agreement', 'dropout_joint_agreement',
                           'joint_max_entropy', 'joint_min_entropy']
    else:
        al_scoring_list = ['entropy', 'random', 'dropout_agreement']
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

num_commands = 0
for label_smoothing in label_smoothing_list:
    for domain in domains:
        for task in ['deps']:
            for model in all_models:
                if task not in model:
                    continue
                if label_smoothing:
                    model_name = dataset + '_' + domain + '_active_learning_train_percentage_' +\
                             train_percentage + '_' + model + '_ls_0.2_' + pretrained_model
                else:
                    model_name = dataset + '_' + domain + '_active_learning_train_percentage_' + \
                                 train_percentage + '_' + model + '_' + pretrained_model
                command = "perl utils/eval07.pl -g data/ontonotes/" + domain + "/test.conllu " \
                          "-s saved_models/" + model_name + "/" + domain + "_pred_test.conllu " \
                          "-o saved_models/" + model_name + "/" + domain + "_res_test_eval07.conllu"
                print(command)
                p = subprocess.Popen(command, shell=True)
                p.communicate()
                num_commands += 1
print(num_commands)