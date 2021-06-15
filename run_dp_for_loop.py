import os
import subprocess

def call_func(src_domain,
              tgt_domain,
              tasks,
              task_levels,
              dataset='ud',
              unfreeze_mode=True,
              load_pretrained_model=None,
              do_mode='train',
              pretrained_model='bert-base-cased',
              vocab_path=None,
              bert_vocab_path=None,
              pretrained_tasks=None,
              training_set_percentage=None,
              training_set_size=None,
              pretrained_training_set_size=None,
              unlabeled_input_dir=None,
              train_unlabeled_data=False,
              train_labeled_and_unlabeled_data=False,
              load_model_path_bool=False,
              num_al_iterations=5,
              label_smoothing=None):
    shell_str = ['python run_dp.py',
                 '--dataset', dataset,
                 '--src_domain', src_domain,
                 '--tgt_domain', tgt_domain,
                 ' --max_seq_length 128 --per_gpu_train_batch_size 4 --gradient_accumulation_steps 8 --seed 1 --learning_rate 5e-5 --weight_decay 0.01  --dropout 0.33 --bert_dropout 0.15 --word_dropout 0.2 --bert_combine_layers all',
                 '--pretrained_model', pretrained_model,
                 '--vocab_path', vocab_path,
                 '--bert_vocab_path', bert_vocab_path,
                 '--num_al_iterations', str(num_al_iterations)
                 ]
    if dataset == 'ud':
        shell_str += ['--max_steps 10000 --warmup_steps 250 --eval_steps 250']
    else:
        shell_str += ['--max_steps 20000 --warmup_steps 500 --eval_steps 500']

    if label_smoothing is not None:
        shell_str += ['--label_smoothing', str(label_smoothing)]
    shell_str += ['--tasks', ' '.join(tasks)]
    shell_str += ['--task_levels', ' '.join([str(level) for level in task_levels])]
    if unfreeze_mode:
        shell_str += ['--unfreeze_bert']

    if load_pretrained_model is not None:
        shell_str += ['--load_pretrained_model', load_pretrained_model]

    if training_set_percentage is not None:
        shell_str += ['--training_set_percentage', str(training_set_percentage)]
    elif training_set_size is not None:
        shell_str += ['--training_set_size', str(training_set_size)]

    if pretrained_training_set_size is not None:
        shell_str += ['--pretrained_training_set_size', str(pretrained_training_set_size)]

    if unlabeled_input_dir is not None:
        shell_str += ['--unlabeled_input_dir', unlabeled_input_dir]

    if train_unlabeled_data:
        shell_str += ['--train_unlabeled_data']

    if train_labeled_and_unlabeled_data:
        shell_str += ['--train_labeled_and_unlabeled_data']

    if do_mode == 'eval':
        shell_str += ['--do_eval']

    if src_domain == tgt_domain:
        model_name = 'saved_models/' + dataset + '_' + src_domain
    else:
        model_name = 'saved_models/' + dataset + '_' + src_domain + '_' + tgt_domain

    if training_set_percentage is not None:
        model_name += '_train_percentage_' + str(training_set_percentage)
    elif training_set_size is not None:
        model_name += '_train_size_' + str(training_set_size)

    model_name += '_' + '_'.join(tasks)

    if label_smoothing is not None:
        model_name += '_ls'

    model_name += '_' + pretrained_model
    if unfreeze_mode:
        model_name += '_unfreeze'
    else:
        model_name += '_freeze'

    if train_unlabeled_data:
        model_name += '_unlabeled'

    if train_labeled_and_unlabeled_data:
        model_name += '_labeled_and_unlabeled'

    model_name += '_4_layers'

    shell_str += ['--output_dir', model_name]

    if not os.path.exists(os.path.join(model_name, tgt_domain + '.tar.gz')) or do_mode == 'eval':
        command = ' '.join(shell_str)
        print(command)
        return command
    else:
        print(' '.join(shell_str))
        print('skipping command')
        return None

# Pretraining on labeled data set
calls_list = []
unfreeze_list = [True]
dataset = "ontonotes"
if dataset == 'ud':
    domains = ['danish', 'english', 'italian', 'hebrew', 'indonesian', 'russian', 'turkish', 'vietnamese']
    vocab_path = "data/ud/vocab/multilingual/vocabulary"
    bert_vocab_path = "config/archive/bert-base-multilingual-cased/vocab.txt"
    pretrained_model = "bert-base-multilingual-cased"
else:
    domains = ['bc', 'bn', 'mz', 'nw', 'tc', 'wb']
    vocab_path = "data/ontonotes/vocab/all/vocabulary"
    bert_vocab_path = "config/archive/bert-large-cased/vocab.txt"
    pretrained_model = "bert-base-cased"

tasks_list = [("deps",), ("ner", ), ("deps", "ner")]
tasks_list = [("deps", "ner")]

task_levels_dict = {
    ("deps",): [-1],
    ("ner", ): [-1],
    ("deps", "ner"): [-1, -1]
}

num_al_iterations = 5
training_set_percentage_list = [0.05, 0.1]
label_smoothing = 0.1

for training_set_percentage in training_set_percentage_list:
    do_mode = 'train'
    for domain in domains:
        for tasks in tasks_list:
            for unfreeze_mode in unfreeze_list:
                task_levels = task_levels_dict[tasks]
                load_model_path = None
                pretrained_tasks = None
                command = call_func(dataset=dataset, vocab_path=vocab_path, bert_vocab_path=bert_vocab_path,
                                    src_domain=domain, tgt_domain=domain, tasks=tasks, task_levels=task_levels,
                                    unfreeze_mode=unfreeze_mode, load_pretrained_model=load_model_path,
                                    do_mode=do_mode, pretrained_model=pretrained_model, pretrained_tasks=pretrained_tasks,
                                    training_set_percentage=training_set_percentage, num_al_iterations=num_al_iterations,
                                    label_smoothing=label_smoothing)
                if command is not None:
                    # p = subprocess.Popen(command, shell=True)
                    # p.communicate()
                    pass