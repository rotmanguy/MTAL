import os
import subprocess

def call_func(src_domain,
              tasks,
              task_levels,
              dataset='ud',
              unfreeze_mode=True,
              load_pretrained_model=None,
              multitask_model_type='complex',
              do_mode='train',
              pretrained_model='bert-base-cased',
              vocab_path=None,
              bert_vocab_path=None,
              training_set_percentage=None,
              training_set_size=None,
              pretrained_training_set_size=None,
              unlabeled_input_dir=None,
              train_unlabeled_data=False,
              train_labeled_and_unlabeled_data=False,
              scale_temperature=None,
              al_selection_method=None,
              task_for_scoring=None,
              task_weights_for_loss=None,
              task_weights_for_selection=None,
              use_ner_embedding=None,
              use_pos_embedding=None,
              num_al_iterations=5,
              label_smoothing=None):

    shell_str = ['python run_mtal.py',
                 '--dataset', dataset,
                 '--src_domain', src_domain,
                 ' --max_seq_length 128 --per_gpu_train_batch_size 32 --seed 1 --learning_rate 5e-5 --weight_decay 0.01  --dropout 0.33 --bert_dropout 0.15 --mix_embedding',
                 '--pretrained_model', pretrained_model,
                 '--vocab_path', vocab_path,
                 '--bert_vocab_path', bert_vocab_path,
                 '--al_selection_method', al_selection_method,
                 '--num_al_iterations', str(num_al_iterations)
                 ]

    if len(tasks) > 1:
        shell_str += ['--multitask_model_type', multitask_model_type]

    if label_smoothing is not None:
        shell_str += ['--label_smoothing', str(label_smoothing)]

    if task_for_scoring is not None:
        shell_str += ['--task_for_scoring', task_for_scoring]

        if task_weights_for_loss is not None:
            shell_str += ['--task_weights_for_loss', ' '.join([str(weight) for weight in task_weights_for_loss])]

    if task_weights_for_selection is not None:
        shell_str += ['--task_weights_for_selection', ' '.join([str(weight) for weight in task_weights_for_selection])]

    if use_ner_embedding is not None:
        shell_str += ['--use_ner_embedding']

    if use_pos_embedding is not None:
        shell_str += ['--use_pos_embedding']

    # if dataset == 'ud':
    #     shell_str += ['--max_steps 10000 --warmup_steps 250 --eval_steps 250']
    # else:
    #     shell_str += ['--max_steps 20000 --warmup_steps 500 --eval_steps 500']

    shell_str += ['--max_steps 100 --warmup_steps 50 --eval_steps 99']

    # shell_str += ['--load_checkpoint_from_last_iter']

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

    if scale_temperature is not None:
        shell_str += ['--scale_temperature']

    if do_mode == 'eval':
        shell_str += ['--do_eval']

    model_name = 'saved_models/' + dataset + '_' + src_domain + '_active_learning'

    if len(tasks) > 1:
            model_name += '_' + multitask_model_type + '_model'

    if training_set_percentage is not None:
        model_name += '_train_percentage_' + str(training_set_percentage)
    elif training_set_size is not None:
        model_name += '_train_size_' + str(training_set_size)

    model_name += '_' + '_'.join(tasks)

    model_name += '_al_selection_method_' + al_selection_method
    if task_for_scoring is not None:
        model_name += '_by_' + task_for_scoring

    if task_weights_for_loss is not None:
        model_name += '_tw_for_l_' + '_'.join([str(weight) for weight in task_weights_for_loss])

    if task_weights_for_selection is not None:
        model_name += '_tw_for_s_' + '_'.join([str(weight) for weight in task_weights_for_selection])

    if label_smoothing is not None:
        model_name += '_ls_' + str(label_smoothing)

    if scale_temperature is not None:
        model_name += '_scale_tmp'

    model_name += '_' + pretrained_model

    if unfreeze_mode:
        model_name += '_unfreeze'
    else:
        model_name += '_freeze'

    if train_unlabeled_data:
        model_name += '_unlabeled'

    if train_labeled_and_unlabeled_data:
        model_name += '_labeled_and_unlabeled'

    shell_str += ['--output_dir', model_name]

    if do_mode == 'eval' and load_pretrained_model is None:
        shell_str += ['--load_pretrained_model', os.path.join(model_name, src_domain + '.tar.gz')]

    #if not os.path.exists(os.path.join(model_name, tgt_domain + '.tar.gz')) or do_mode == 'eval':
    if not os.path.exists(os.path.join(model_name, src_domain + '_res_test_iter_' + str(num_al_iterations - 1) + '.conllu')):
        command = ' '.join(shell_str)
        print(command)
        return command
    else:
        # print(' '.join(shell_str))
        # print('skipping command')
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
    # domains = ['bn', 'mz', 'bc', 'nw', 'tc', 'wb']
    domains = ['bc', 'bn', 'mz', 'nw', 'tc', 'wb']
    # domains = ['bn', 'mz', 'bc']
    # domains = ['nw', 'tc', 'wb']
    domains = ['mz']

    vocab_path = "data/ontonotes/vocab/all/vocabulary"
    bert_vocab_path = "config/archive/bert-large-cased/vocab.txt"
    pretrained_model = "bert-base-cased"

#tasks_list = [("deps", "ner"), ("deps",), ("ner", )]
# tasks_list = [("deps", "ner", "upos"), ("upos", )]
tasks_list = [("deps",), ("ner", ), ("deps", "ner")]

task_levels_dict = {
    ("deps",): [-1],
    ("ner", ): [-1],
    ("upos",): [-1],
    ("deps", "ner"): [-1, -1],
    ("deps", "ner", "upos"): [-1, -1, -1]
}

multitask_model_type_list = ['complex', 'simple']
do_mode = 'train_and_eval'
load_model_path = None

training_set_percentage_list = [0.02]
num_al_iterations = 2
scale_temperature_list = [None]
# label_smoothing = 0.2
# label_smoothing = None
# label_smoothing_list = [0.2, None]
label_smoothing_list = [0.2]
use_ner_embedding = None

num_calls = 0
for label_smoothing in label_smoothing_list:
    for training_set_percentage in training_set_percentage_list:
        for scale_temperature in scale_temperature_list:
            for domain in domains:
                for tasks in tasks_list:
                    task_levels = task_levels_dict[tasks]
                    if len(tasks) > 1:
                        al_selection_method_list = ['average_entropy_based_confidence', 'average_dropout_agreement',
                                                    'maximum_entropy_based_confidence', 'minimum_entropy_based_confidence',
                                                    'pareto_entropy_based_confidence', 'rrf_entropy_based_confidence',
                                                    'entropy_based_confidence',
                                                    'dropout_agreement', 'random']
                        multitask_model_type_list = ['complex', 'simple']
                        task_weights_for_selection_list = [None, [0.7, 0.3]]
                    else:
                        al_selection_method_list = ['entropy_based_confidence', 'dropout_agreement', 'random']
                        multitask_model_type_list = ['simple']
                        task_weights_for_selection_list = [None]

                    for multitask_model_type in multitask_model_type_list:
                        for task_weights_for_selection in task_weights_for_selection_list:
                            for unfreeze_mode in unfreeze_list:
                                for al_selection_method in al_selection_method_list:
                                    if al_selection_method in['entropy_based_confidence', 'dropout_agreement'] and len(tasks) > 1:
                                        task_for_scoring_list = tasks
                                    else:
                                        task_for_scoring_list = (None, )
                                    for task_for_scoring in task_for_scoring_list:
                                        command = call_func(dataset=dataset, vocab_path=vocab_path, bert_vocab_path=bert_vocab_path,
                                                            src_domain=domain, tasks=tasks, task_levels=task_levels, unfreeze_mode=unfreeze_mode,
                                                            do_mode=do_mode, pretrained_model=pretrained_model, multitask_model_type=multitask_model_type,
                                                            training_set_percentage=training_set_percentage, task_weights_for_selection=task_weights_for_selection,
                                                            al_selection_method=al_selection_method, task_for_scoring=task_for_scoring, num_al_iterations=num_al_iterations,
                                                            use_ner_embedding=use_ner_embedding, label_smoothing=label_smoothing, scale_temperature=scale_temperature)
                                        if command is not None:
                                            num_calls += 1
                                            # if num_calls % 2 == 0:
                                            #    continue
                                            # else:
                                            p = subprocess.Popen(command, shell=True)
                                            p.communicate()
                                            print()
print(num_calls)

# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=1

## temperature
# for domain in domains:
#     command = 'python run_dp.py --dataset ontonotes --src_domain ' + domain + '' \
#                 ' --tgt_domain ' + domain +  '' \
#                 ' --max_seq_length 128 --per_gpu_train_batch_size 4 --gradient_accumulation_steps 8' \
#                                             ' --seed 1 --learning_rate 5e-5 --weight_decay 0.01 --dropout 0.33 ' \
#                                             '--bert_dropout 0.15 --word_dropout 0.2 --bert_combine_layers all' \
#                                             ' --pretrained_model bert-base-cased --vocab_path data/ontonotes/vocab/all/vocabulary' \
#                                             ' --bert_vocab_path config/archive/bert-large-cased/vocab.txt --al_selection_method joint_entropy ' \
#                                             '--max_steps 20000 --warmup_steps 500 --eval_steps 500 --tasks deps ner --task_levels -1 -1 ' \
#                                             '--unfreeze_bert --training_set_percentage 0.02 --do_eval ' \
#                                             '--output_dir saved_models/check_' + domain + '_temperature ' \
#                                             '--load_pretrained_model saved_models/ontonotes_'+domain+'_active_learning_' \
#                                             'train_percentage_0.02_deps_ner_al_selection_method_random_bert-base-cased_unfreeze/'+domain+'.tar.gz'
#     print(command)
#     p = subprocess.Popen(command, shell=True)
#     p.communicate()

#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
