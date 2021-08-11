import os
import subprocess

def call_func(src_domain,
              tasks,
              task_levels,
              dataset='ud',
              unfreeze_mode=True,
              load_pretrained_model=None,
              do_mode='train',
              pretrained_model='bert-base-cased',
              vocab_path=None,
              load_sample_ids_dir=None,
              bert_vocab_path=None,
              training_set_percentage=None,
              training_set_size=None,
              pretrained_training_set_size=None,
              unlabeled_input_dir=None,
              train_unlabeled_data=False,
              train_labeled_and_unlabeled_data=False,
              al_scoring=None,
              al_scoring_to_load=None,
              task_for_scoring=None,
              num_al_iterations=5,
              label_smoothing=None):
    shell_str = ['python run_dp.py',
                 '--dataset', dataset,
                 '--src_domain', src_domain,
                 ' --max_seq_length 128 --per_gpu_train_batch_size 32 --seed 1 --learning_rate 5e-5 --weight_decay 0.01  --dropout 0.33 --bert_dropout 0.15 --word_dropout 0.2 --bert_combine_layers all',
                 '--pretrained_model', pretrained_model,
                 '--vocab_path', vocab_path,
                 '--bert_vocab_path', bert_vocab_path,
                 '--al_scoring', al_scoring,
                 '--num_al_iterations', str(num_al_iterations)
                 ]

    if src_domain == 'all':
        shell_str += ['--per_gpu_eval_batch_size', '4']

    if label_smoothing is not None:
        shell_str += ['--label_smoothing', str(label_smoothing)]

    if task_for_scoring is not None:
        shell_str += ['--task_for_scoring', task_for_scoring]

    if dataset == 'ud':
        shell_str += ['--max_steps 10000 --warmup_steps 250 --eval_steps 250']
    else:
        shell_str += ['--max_steps 20000 --warmup_steps 500 --eval_steps 500']

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

    if load_sample_ids_dir is not None:
        shell_str += ['--load_sample_ids_dir', load_sample_ids_dir]

    if do_mode == 'eval':
        shell_str += ['--do_eval']

    model_name = 'saved_models/' + dataset + '_' + src_domain + '_active_learning'

    if training_set_percentage is not None:
        model_name += '_train_percentage_' + str(training_set_percentage)
    elif training_set_size is not None:
        model_name += '_train_size_' + str(training_set_size)

    model_name += '_' + '_'.join(tasks)

    model_name += '_al_scoring_' + al_scoring
    model_name += '_' + al_scoring_to_load
    if task_for_scoring is not None:
        model_name += '_by_' + task_for_scoring

    if label_smoothing is not None:
        model_name += '_ls_' + str(label_smoothing)

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
    domains = ['bn', 'mz', 'bc', 'nw', 'tc', 'wb']
    # domains = ['bn', 'mz', 'bc']
    # domains = ['nw', 'tc', 'wb']

    vocab_path = "data/ontonotes/vocab/all/vocabulary"
    bert_vocab_path = "config/archive/bert-large-cased/vocab.txt"
    pretrained_model = "bert-base-cased"

#tasks_list = [("deps", "ner"), ("deps",), ("ner", )]
tasks_list = [("deps", ), ("ner", )]


task_levels_dict = {
    ("deps",): [-1],
    ("ner", ): [-1]
}

do_mode = 'train_and_eval'
al_scoring = 'from_file'
load_model_path = None

training_set_percentage_list = [0.02]
num_al_iterations = 5
# label_smoothing = 0.2
# label_smoothing = None
label_smoothing_list = [0.2, None]
num_calls = 0
for label_smoothing in label_smoothing_list:
    for training_set_percentage in training_set_percentage_list:
        for domain in domains:
            for tasks in tasks_list:
                task_levels = task_levels_dict[tasks]
                for unfreeze_mode in unfreeze_list:
                    if (al_scoring == 'entropy' or al_scoring == 'dropout_agreement') and len(tasks) > 1:
                        task_for_scoring_list = tasks
                    else:
                        task_for_scoring_list = (None, )


                    al_scoring_to_load_list = ['entropy', 'dropout_agreement']
                    for al_scoring_to_load in al_scoring_to_load_list:
                        load_sample_ids_dir = 'saved_models/' + dataset + '_' + domain + '_active_learning_train_percentage_'\
                                              + str(training_set_percentage) + '_' + '_'.join(tasks) + '_al_scoring_' + al_scoring_to_load
                        if label_smoothing is not None:
                            load_sample_ids_dir += '_ls_'+str(label_smoothing)
                        load_sample_ids_dir += '_' + pretrained_model
                        if unfreeze_mode:
                            load_sample_ids_dir += '_unfreeze'
                        else:
                            load_sample_ids_dir += '_freeze'

                        for task_for_scoring in task_for_scoring_list:
                            if tasks == ("deps",):
                                tasks_for_training = ("ner",)
                            else:
                                tasks_for_training = ("deps",)
                            command = call_func(dataset=dataset, vocab_path=vocab_path, bert_vocab_path=bert_vocab_path,
                                                src_domain=domain, tasks=tasks_for_training, load_sample_ids_dir=load_sample_ids_dir, task_levels=task_levels, unfreeze_mode=unfreeze_mode,
                                                do_mode=do_mode, pretrained_model=pretrained_model, training_set_percentage=training_set_percentage,
                                                al_scoring=al_scoring, al_scoring_to_load=al_scoring_to_load, task_for_scoring=task_for_scoring, num_al_iterations=num_al_iterations,
                                                label_smoothing=label_smoothing)
                            if command is not None:
                                num_calls += 1
                                # if num_calls % 2 == 1:
                                #    continue
                                # else:
                                # p = subprocess.Popen(command, shell=True)
                                # p.communicate()
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
#                                             ' --bert_vocab_path config/archive/bert-large-cased/vocab.txt --al_scoring joint_entropy ' \
#                                             '--max_steps 20000 --warmup_steps 500 --eval_steps 500 --tasks deps ner --task_levels -1 -1 ' \
#                                             '--unfreeze_bert --training_set_percentage 0.02 --do_eval ' \
#                                             '--output_dir saved_models/check_' + domain + '_temperature ' \
#                                             '--load_pretrained_model saved_models/ontonotes_'+domain+'_active_learning_' \
#                                             'train_percentage_0.02_deps_ner_al_scoring_random_bert-base-cased_unfreeze/'+domain+'.tar.gz'
#     print(command)
#     p = subprocess.Popen(command, shell=True)
#     p.communicate()

#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
