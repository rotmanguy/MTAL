"""
The base UDify model for training and prediction
"""
import random
from copy import deepcopy
from typing import Optional, Any, Dict, List, Tuple
from overrides import overrides
import logging

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask

from torch.nn.modules.linear import Linear

from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer

from models.dependency_decoder import DependencyDecoder
from models.tag_decoder import TagDecoder
from modules.bert_pretrained import PretrainedBertIndexer
from modules.scalar_mix import ScalarMixWithDropout
from modules.text_field_embedder import HCDPTextFieldEmbedder

logger = logging.getLogger('mylog')


class Multitask_Model(Model):
    """
    The UDify model base class. Applies a sequence of shared encoders before decoding in a multi-task configuration.
    Uses TagDecoder and DependencyDecoder to decode each UD task.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 tasks: List[str],
                 task_levels: List[int],
                 task_weights: List[float] = None,
                 tag_representation_dim = 256,
                 arc_representation_dim = 768,
                 pos_embed_dim = None,
                 ner_embed_dim = None,
                 dropout: float = 0.0,
                 bert_dropout: float = 0.2,
                 word_dropout: float = 0.0,
                 layer_dropout: float = 0.0,
                 label_smoothing: float = 0.0,
                 mix_embedding: bool = False,
                 unfreeze_bert: bool = True,
                 scale_temperature: bool = False,
                 pretrained_model: str = "bert-base-cased",
                 bert_combine_layers: str = 'all',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Multitask_Model, self).__init__(vocab, regularizer)


        self.pretrained_model = pretrained_model
        assert len(tasks) == len(task_levels)
        self.task_to_layer = dict(zip(tasks, task_levels))
        # sorting task order
        # task_weights is converted to Dict[str, float] instead of List[float]
        self.tasks, self.task_weights = self.sort_tasks(tasks, task_weights)

        self.vocab = vocab
        if 'roberta' in pretrained_model:
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
            self.bert_vocab = self.tokenizer.encoder
        else:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            self.bert_vocab = self.tokenizer.vocab

        self.bert_combibe_layers = bert_combine_layers
        self.text_field_indexer = PretrainedBertIndexer(pretrained_model=pretrained_model,
                                                        use_starting_offsets=True,
                                                        do_lowercase=False)
        text_field_embedder_params_dict = {'allow_unmatched_keys': True,
                                            'dropout': dropout,
                                            'embedder_to_indexer_map': {'bert': ['bert', 'bert-offsets']},
                                            'token_embedders': {'bert': {'dropout': bert_dropout,
                                                                        'pretrained_model': pretrained_model,
                                                                        'requires_grad': unfreeze_bert,
                                                                        'type': 'bert-pretrained-embedder'}}}
        text_field_embedder_params = Params(text_field_embedder_params_dict)
        self.text_field_embedder = HCDPTextFieldEmbedder.from_params(vocab=vocab,
                                                                     params=text_field_embedder_params)
        if len(self.tasks) > 1:
            unshared_modules = {}
            shared_linear = {}
            unshared_linear = {}
            gating_modules = {}
            for task in tasks:
                unshared_modules[task] = torch.nn.ModuleList([deepcopy(self.text_field_embedder.token_embedder_bert.bert_model.encoder.layer[i])
                                                              for i in range(-4, 0, 1)])
                shared_linear[task] = Linear(768, 768)
                unshared_linear[task] = Linear(768, 768)
                gating_modules[task] = Gating(2, 768)
            self.unshared_modules = torch.nn.ModuleDict(unshared_modules)
            self.shared_linear = torch.nn.ModuleDict(shared_linear)
            self.unshared_linear = torch.nn.ModuleDict(unshared_linear)
            self.gating_modules = torch.nn.ModuleDict(gating_modules)

        # if 'deps' in tasks and 'ner' in tasks:
        #     unshared_modules = {}
        #     unshared_modules['deps'] = torch.nn.ModuleList([deepcopy(self.text_field_embedder.token_embedder_bert.bert_model.encoder.layer[i])
        #                                                     for i in range(-4, 0, 1)])
        #     unshared_modules['ner'] =torch.nn.ModuleList([deepcopy(self.text_field_embedder.token_embedder_bert.bert_model.encoder.layer[i])
        #                                                     for i in range(-4, 0, 1)])
        #     self.unshared_modules = torch.nn.ModuleDict(unshared_modules)
        #     shared_linear = {}
        #     shared_linear['deps'] = Linear(768, 768)
        #     shared_linear['ner'] = Linear(768, 768)
        #     self.shared_linear = torch.nn.ModuleDict(shared_linear)
        #     unshared_linear = {}
        #     unshared_linear['deps'] = Linear(768, 768)
        #     unshared_linear['ner'] = Linear(768, 768)
        #     self.unshared_linear = torch.nn.ModuleDict(unshared_linear)
        #     gating_modules = {}
        #     gating_modules['deps'] = Gating(2, 768)
        #     gating_modules['ner'] = Gating(2, 768)
        #     self.gating_modules = torch.nn.ModuleDict(gating_modules)

        self.pos_embed_dim = pos_embed_dim and 'upos' in tasks
        self.ner_embed_dim = ner_embed_dim and 'ner' in tasks
        self.word_dropout = word_dropout
        decoders = {}
        for task in tasks:
            if task == 'deps':
                decoders[task] = DependencyDecoder(vocab=vocab,
                                                   tag_representation_dim=tag_representation_dim,
                                                   arc_representation_dim=arc_representation_dim,
                                                   pos_embed_dim=self.pos_embed_dim,
                                                   ner_embed_dim=self.ner_embed_dim,
                                                   scale_temperature=scale_temperature,
                                                   label_smoothing=label_smoothing)

            else:
                decoders[task] = TagDecoder(vocab=vocab,
                                            task=task,
                                            output_dim=768,
                                            scale_temperature=scale_temperature,
                                            label_smoothing=label_smoothing)

        self.decoders = torch.nn.ModuleDict(decoders)

        self.layer_dropout = layer_dropout

        if mix_embedding:
            # the output of Bert is 1 + num_hidden_layers since the embedding layer is included in the output
            mixture_size = 1 + self.text_field_embedder.token_embedder_bert.bert_model.config.num_hidden_layers
            self.scalar_mix = torch.nn.ModuleDict({
                task: ScalarMixWithDropout(mixture_size=mixture_size,
                                           do_layer_norm=False,
                                           layer_dropout=self.layer_dropout)
                for task in self.decoders if task in ["ner", "deps"]
            })
        else:
            self.scalar_mix = None

        self.metrics = {}

        for task in self.tasks:
            if task not in self.decoders:
                raise ConfigurationError(f"Task {task} has no corresponding decoder. Make sure their names match.")

        initializer(self)
        self._count_params()

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]] = None,
                **kwargs: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:

        gold_tags = kwargs

        loss_dict = {}
        output_dict = {"loss_dict": loss_dict}
        loss = 0

        if "tokens" in self.tasks:
            # Model is predicting tokens, so add them to the gold tags
            gold_tags["tokens"] = tokens["tokens"]

        mask = get_text_field_mask(tokens)
        embedded_text_input = self.text_field_embedder(tokens)

        # Run through each of the tasks on the shared encoder and save predictions
        for task in self.tasks:
            if self.scalar_mix and task in self.scalar_mix:
                decoder_input = self.scalar_mix[task](embedded_text_input, mask)
            else:
                decoding_layer = self.task_to_layer[task] if task in self.task_to_layer else -1
                decoder_input = embedded_text_input[decoding_layer]

            if len(self.tasks) > 1:
                unshared_output = embedded_text_input[-5]
                for unshared_layer in self.unshared_modules[task]:
                    unshared_output = unshared_layer(unshared_output)[0]
                shared_output = self.shared_linear[task](decoder_input)
                unshared_output = self.unshared_linear[task](unshared_output)
                decoder_input = self.gating_modules[task](tuple([shared_output, unshared_output]))

            if task == "deps":
                # if "upos" in self.tasks:
                #     pos_input = output_dict["upos_logits"] if "upos_logits" in output_dict else None
                #     pos_type = "logits"
                # else:
                #     if self.pos_embed_dim is None:
                #         pos_input = None
                #         pos_type = None
                #     else:
                #         pos_input = gold_tags["upos"]
                #         pos_type = "tags"
                # if "ner" in self.tasks:
                #     # ner_input = output_dict["ner_preds"]
                #     ner_input = None
                # else:
                #     ner_input = None
                pos_input = None
                pos_type = None
                ner_input = None
                pred_output = self.decoders[task](encoded_text=decoder_input, mask=mask,
                                                  pos_input=pos_input, pos_type=pos_type,
                                                  ner_input=ner_input,
                                                  head_tags=gold_tags["head_tags"],
                                                  head_indices=gold_tags["head_indices"])

                loss_dict[task] = pred_output['loss']
                for key in ["predicted_heads", "predicted_head_tags", "head_logits", "tag_logits", "head_preds", "tag_preds", "mask"]:
                    output_dict[key] = pred_output[key]
            else:
                pred_output = self.decoders[task](decoder_input, mask, gold_tags)
                output_dict[task + '_preds'] = pred_output["preds"]
                output_dict[task + '_logits'] = pred_output["logits"]
                loss_dict[task] = pred_output['loss']

            if task in self.tasks or (task == "deps" and "head_tags" in gold_tags):
                # Keep track of the loss if we have the gold tags available
                if self.task_weights is None:
                    loss += pred_output["loss"]
                else:
                    loss += self.task_weights[task] * pred_output["loss"]

        if gold_tags:
            output_dict["loss"] = loss
            output_dict["loss_dict"] = loss_dict

        if metadata is not None:
            for x in metadata:
                for k, v in x.items():
                    if k not in output_dict:
                        output_dict[k] = []
                    output_dict[k].append(v)
        return output_dict

    def _apply_token_dropout(self, tokens):
        # Word dropout
        if "tokens" in tokens:
            oov_token = self.vocab.get_token_index(self.vocab._oov_token)
            ignore_tokens = [self.vocab.get_token_index(self.vocab._padding_token)]
            tokens["tokens"] = self.token_dropout(tokens["tokens"],
                                                  oov_token=oov_token,
                                                  padding_tokens=ignore_tokens,
                                                  p=self.word_dropout,
                                                  training=self.training)

        if "bert" in tokens:
            # BERT token dropout
            if 'roberta' in self.pretrained_model.lower():
                oov_token = self.bert_vocab["<unk>"]
                ignore_tokens = [self.bert_vocab["<pad>"], self.bert_vocab["<s>"], self.bert_vocab["</s>"]]
                tokens["bert"] = self.token_dropout(tokens["bert"],
                                                    oov_token=oov_token,
                                                    padding_tokens=ignore_tokens,
                                                    p=self.word_dropout,
                                                    training=self.training)
            else:
                oov_token = self.bert_vocab["[MASK]"]
                ignore_tokens = [self.bert_vocab["[PAD]"], self.bert_vocab["[CLS]"], self.bert_vocab["[SEP]"]]
                tokens["bert"] = self.token_dropout(tokens["bert"],
                                                    oov_token=oov_token,
                                                    padding_tokens=ignore_tokens,
                                                    p=self.word_dropout,
                                                    training=self.training)

    @staticmethod
    def token_dropout(tokens: torch.LongTensor,
                      oov_token: int,
                      padding_tokens: List[int],
                      p: float = 0.2,
                      training: float = True) -> torch.LongTensor:
        """
        During training, randomly replaces some of the non-padding tokens to a mask token with probability ``p``

        :param tokens: The current batch of padded sentences with word ids
        :param oov_token: The mask token
        :param padding_tokens: The tokens for padding the input batch
        :param p: The probability a word gets mapped to the unknown token
        :param training: Applies the dropout if set to ``True``
        :return: A copy of the input batch with token dropout applied
        """
        if training and p > 0:
            # Ensure that the tensors run on the same device
            device = tokens.device

            # This creates a mask that only considers unpadded tokens for mapping to oov
            padding_mask = torch.ones(tokens.size(), dtype=torch.bool).to(device)
            for pad in padding_tokens:
                padding_mask &= (tokens != pad).to(dtype=torch.bool)

            # Create a uniformly random mask selecting either the original words or OOV tokens
            dropout_mask = (torch.empty(tokens.size()).uniform_() < p).to(device, dtype=torch.bool)
            oov_mask = dropout_mask & padding_mask

            oov_fill = torch.empty(tokens.size(), dtype=torch.long).fill_(oov_token).to(device)

            result = torch.where(oov_mask, oov_fill, tokens)

            return result
        else:
            return tokens

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for task in self.tasks:
            self.decoders[task].decode(output_dict)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {name: task_metric
                   for task in self.tasks
                   for name, task_metric in self.decoders[task].get_metrics(reset).items()}

        # The "sum" metric summing all tracked metrics keeps a good measure of patience for early stopping and saving
        metrics_to_track = {"upos", "xpos", "feats", "lemmas", "LAS", "UAS"}
        metrics[".run/.sum"] = sum(metric
                                   for name, metric in metrics.items()
                                   if not name.startswith("_") and set(name.split("/")).intersection(metrics_to_track))

        return metrics

    def _count_params(self):
        self.total_params = sum(p.numel() for p in self.parameters())
        self.total_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Total number of parameters: {self.total_params}")
        logger.info(f"Total number of trainable parameters: {self.total_train_params}")

    def sort_tasks(self, tasks, task_weights):
        if task_weights is not None:
            if len(task_weights) != len(tasks):
                raise ValueError('task_weights does not have the same length as tasks')
            task_to_weight = dict(zip(tasks, task_weights))
        else:
            task_to_weight = None
        sorted_tasks = [k for k in sorted(self.task_to_layer, key=self.task_to_layer.get) if k in tasks]
        for task in tasks:
            if task not in sorted_tasks:
                sorted_tasks.append(task)
        # Making sure 'deps' is always the last task
        if 'deps' in tasks:
            sorted_tasks.remove('deps')
            sorted_tasks.append('deps')
        if task_weights is not None:
            sorted_task_weights = dict([(task, task_to_weight[task]) for task in sorted_tasks])
        else:
            sorted_task_weights = None
        return sorted_tasks, sorted_task_weights

class Gating(torch.nn.Module):
    # Implementation of:
    # Sato, Motoki, et al. "Adversarial training for cross-domain universal dependency parsing."
    # Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal
    #  Dependencies. 2017.‏
    def __init__(self, num_gates, input_dim):
        super(Gating, self).__init__()
        self.num_gates = num_gates
        self.input_dim = input_dim
        if self.num_gates == 2:
            self.linear = torch.nn.Linear(self.num_gates * self.input_dim, self.input_dim)
        elif self.num_gates > 2:
            self.linear = torch.nn.Linear(self.num_gates * self.input_dim, self.num_gates * self.input_dim)
            self.softmax = torch.nn.Softmax(-1)
        else:
            raise ValueError('num_gates should be greater or equal to 2')

    def forward(self, tuple_of_inputs):
        # output size should be equal to the input sizes
        if self.num_gates == 2:
            alpha = torch.sigmoid(self.linear(torch.cat(tuple_of_inputs, dim=-1)))
            output = torch.mul(alpha, tuple_of_inputs[0]) + torch.mul(1 - alpha, tuple_of_inputs[1])
        else:  # elif self.num_gates > 2:
            # extend the gating mechanism to more than 2 encoders
            batch_size, len_size, dim_size = tuple_of_inputs[0].size()
            alpha = torch.sigmoid(self.linear(torch.cat(tuple_of_inputs, dim=-1)))
            alpha = self.softmax(alpha.view(batch_size, len_size, dim_size, self.num_gates))
            output = torch.sum(torch.mul(alpha, torch.stack(tuple_of_inputs, dim=-1)), dim=-1)
        return output