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
An example of the data files is available under the <em>data</em> directory.

# Running the large base model
We provide here examples where the large base model is BERT-base.

## Amazon Reviews
An example for training the base model on the Beauty domain from the Amazon Reviews dataset:
```
python run_mtal.py --dataset ontonotes --src_domain mz --max_seq_length 128 --per_gpu_train_batch_size 32 --seed 1 --learning_rate 5e-5
--weight_decay 0.01 --dropout 0.33 --bert_dropout 0.15 --mix_embedding --pretrained_model bert-base-cased --al_selection_method maximum_entropy_based_confidence --num_al_iterations 5 --multitask_model_type complex --label_smoothing 0.2 --task_weights_for_loss 0.6 0.4 --task_weights_for_selection 0.6 0.4 --max_steps 20000 --warmup_steps 500 --eval_steps 1000 --tasks deps ner --task_levels -1 -1 --unfreeze_bert --training_set_percentage 0.02 --output_dir saved_models/ontonotes_mz_active_learning_complex_model_train_percentage_0.02_deps_ner_al_selection_method_maximum_entropy_based_confidence_tw_for_l_0.6_0.4_tw_for_s_0.6_0.4_ls_0.2_bert-base-cased_unfreeze
```

To train the model on a different domain simply modify the src_domain and the output_dir accordingly.

## MultiNLI
An example for training the base model on the Fiction domain from the MultiNLI dataset:
```
python sentence_classification.py --dataset_name MNLI --src_domain fiction --tgt_domains captions fiction government slate telephone travel --bert_model bert-base-cased --task_name sentiment_cnn --cnn_window_size 9 --cnn_out_channels 16 --train_batch_size 32  --combine_layers mix --layer_dropout 0.1 --save_best_weights True --num_train_epochs 10 --warmup_proportion 0.0 --learning_rate 1e-4 --weight_decay 0.01 --bert_model_type default --output_dir saved_models/MNLI/fiction/original_model --model_name pytorch_model.bin --do_train --do_eval
```

To train the model on a different domain simply modify the src_domain and the output_dir accordingly.

# Running a compressed AMoC model
Once training the base model, we can now compress it to a smaller model by removing a subset of its layers

## Amazon Reviews
An example for training a compressed model on the Beauty domain from the Amazon Reviews dataset by removing all the odd layers:
```
python sentence_classification.py --dataset_name Amazon_Reviews --src_domain Beauty --tgt_domains Amazon_Instant_Video Beauty Digital_Music Musical_Instruments Sports_and_Outdoors Video_Games --bert_model bert-base-cased --task_name sentiment_cnn --cnn_window_size 9 --cnn_out_channels 16 --train_batch_size 32  --combine_layers mix --layer_dropout 0.1 --save_best_weights True --layers_to_prune 1 3 5 7 9 11 --num_train_epochs 1 --warmup_proportion 0.0 --learning_rate 1e-4 --bert_model_type layer_pruning --output_dir saved_models/Amazon_Reviews/Beauty/counterfactual_models/freeze_layers_1+3+5+7+9+11 --load_model_path saved_models/Amazon_Reviews/Beauty/original_model/pytorch_model.bin --model_name pytorch_model.bin --do_train --do_eval
```

To remove a different set of layers simply modify the layers_to_prune and the output_dir accordingly.

## MultiNLI
An example for training a compressed model on the Fiction domain from the MultiNLI dataset by removing all the odd layers:
```
python sentence_classification.py --dataset_name MNLI --src_domain fiction --tgt_domains captions fiction government slate telephone travel --bert_model bert-base-cased --task_name sentiment_cnn --cnn_window_size 9 --cnn_out_channels 16 --train_batch_size 32  --combine_layers mix --layer_dropout 0.1 --save_best_weights True --layers_to_prune 1 3 5 7 9 11 --num_train_epochs 1 --warmup_proportion 0.0 --learning_rate 1e-4 --bert_model_type layer_pruning --output_dir saved_models/MNLI/fiction/counterfactual_models/freeze_layers_1+3+5+7+9+11 --load_model_path saved_models/MNLI/fiction/original_model/pytorch_model.bin --model_name pytorch_model.bin --do_train --do_eval
```

To remove a different set of layers simply modify the layers_to_prune and the output_dir accordingly.

# Citation
```
@article{rotman2021model,
  title={Multi-task Active Learning for Pre-trained Transformer-based Models},
  author={Rotman, Guy and Reichart, Roi},
  journal={Transactions of the Association for Computational Linguistics},
  volume={10},
  year={2022},
  publisher={MIT Press}
}
```