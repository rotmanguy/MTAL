Official code for the paper ["Multi-task Active Learning for Pre-trained Transformer-based Models
"](https://arxiv.org/abs/2208.05379).
Please cite our paper in case you are using this code.

# Requirements
To install the requirement simply run the following command:
```
pip install -r requirements.txt
```
# Data
The data should be in the .conllu format of ["Universal Dependencies"](https://universaldependencies.org/).
The only exception is that the NER column should be added at the end of each row as the 11th column.
An example of the data structure and the data files is available under the <em>data</em> directory.

# Running examples
We here supply examples to run our Single-task and Multi-task Active Learning scripts for pre-trained Transformer based models.

## Single-task Active Learning
To run a single-task dependency parsing model with a BERT encoder and the 'entropy_based_confidence' confidence score simply run the following script:
```
python run_mtal.py --dataset ontonotes --src_domain bc --max_seq_length 128 --per_gpu_train_batch_size 32 --seed 1 --learning_rate 5e-5 --weight_decay 0.01 --dropout 0.33 --bert_dropout 0.15 --mix_embedding --pretrained_model bert-base-cased --al_selection_method entropy_based_confidence --num_al_iterations 5 --label_smoothing 0.2 --max_steps 20000 --warmup_steps 500 --eval_steps 1000 --tasks deps --task_levels -1 --unfreeze_bert --training_set_percentage 0.02 --output_dir saved_models/ontonotes_bc_active_learning_simple_model_train_percentage_0.02_deps_al_selection_method_entropy_based_confidence_ls_0.2_bert-base-cased_unfreeze
```

- To train the model on a different domain simply modify the 'src_domain' argument and rename the 'output_dir' argument accordingly.
- To run the AL selection process with a different confidence score simply modify the 'al_selection_method' argument and rename the 'output_dir' argument accordingly. All available confidence scores can be found under the 'choices' of the 'al_selection_method' argument in `run_mtal.py`.
- To run a named entity recognition model simply modify the 'tasks' argument to 'ner' and rename the 'output_dir' argument accordingly.
- To run a RoBERTa encoder simply modify the 'pretrained_model' argument to 'roberta-base' and rename the 'output_dir' argument accordingly.

## Multi-task Active Learning
To run a multi-task model on dependency parsing and named entity recognition with a BERT encoder and the 'average_entropy_based_confidence' confidence score simply run the following script:
```
python run_mtal.py --dataset ontonotes --src_domain bc --max_seq_length 128 --per_gpu_train_batch_size 32 --seed 1 --learning_rate 5e-5 --weight_decay 0.01 --dropout 0.33 --bert_dropout 0.15 --mix_embedding --pretrained_model bert-base-cased --al_selection_method average_entropy_based_confidence --num_al_iterations 5 --multitask_model_type complex --label_smoothing 0.2 --task_weights_for_loss 0.5 0.5 --max_steps 20000 --warmup_steps 500 --eval_steps 1000 --tasks deps ner --task_levels -1 -1 --unfreeze_bert --training_set_percentage 0.02 --output_dir saved_models/ontonotes_bc_active_learning_complex_model_train_percentage_0.02_deps_ner_al_selection_method_average_entropy_based_confidence_tw_for_l_0.5_0.5_ls_0.2_bert-base-cased_unfreeze
```

- To train the model on a different domain simply modify the 'src_domain' argument and rename the 'output_dir' argument accordingly.
- To run the AL selection process with a different confidence score simply modify the 'al_selection_method' argument and rename the 'output_dir' argument accordingly. All available confidence scores can be found under the 'choices' of the 'al_selection_method' argument in `run_mtal.py`.
- To run a RoBERTa encoder simply modify the 'pretrained_model' argument to 'roberta-base' and rename the 'output_dir' argument accordingly.
- To run a simple multi-task model (please refer to SMT in our paper) that do not use non-shared components per task simply modify the 'multitask_model_type' argument to 'simple' and rename the 'output_dir' argument accordingly.
- To base the AL selection method on only one of the tasks please specify the 'task_for_scoring' argument and rename the 'output_dir' argument accordingly (currently only available for the following AL selection methods: ['entropy_based_confidence', 'dropout_agreement']).

## Weighted Multi-task Active Learning
In section 6 of the paper, we present an option to perform a weighted AL selection.
To use this option we simply add the 'task_weights_for_selection' argument to the previous command:
```
python run_mtal.py --dataset ontonotes --src_domain bc --max_seq_length 128 --per_gpu_train_batch_size 32 --seed 1 --learning_rate 5e-5 --weight_decay 0.01 --dropout 0.33 --bert_dropout 0.15 --mix_embedding --pretrained_model bert-base-cased --al_selection_method average_entropy_based_confidence --num_al_iterations 5 --multitask_model_type complex --label_smoothing 0.2 --task_weights_for_loss 0.5 0.5 --task_weights_for_selection 0.6 0.4 --max_steps 20000 --warmup_steps 500 --eval_steps 1000 --tasks deps ner --task_levels -1 -1 --unfreeze_bert --training_set_percentage 0.02 --output_dir saved_models/ontonotes_bc_active_learning_complex_model_train_percentage_0.02_deps_ner_al_selection_method_average_entropy_based_confidence_tw_for_l_0.5_0.5_tw_for_s_0.6_0.4_ls_0.2_bert-base-cased_unfreeze
```

This option is currently available for the following AL selection methods: ['average_entropy_based_confidence', 'average_dropout_agreement', 'maximum_entropy_based_confidence', 'minimum_entropy_based_confidence', 'pareto_entropy_based_confidence', 'rrf_entropy_based_confidence', 'independent_selection_entropy_based_confidence'].

## Binary Linear Programming for Constrained Multi-task Active Learning
In section 7 of the paper, we present an option to solve a Constrained Multi-task Active Learning using a Binary Linear Programming (BLP) formulation.
In `blp_optimzation_example.py` we supply a simple example of this option for a single AL iteration.
To run this option simply run the following script:
```
python blp_optimzation_example.py
```

# Citation
```
@article{rotman2022multi,
  title={Multi-task active learning for pre-trained transformer-based models},
  author={Rotman, Guy and Reichart, Roi},
  journal={Transactions of the Association for Computational Linguistics},
  volume={10},
  pages={1209--1228},
  year={2022},
  publisher={MIT Press}
}
```
