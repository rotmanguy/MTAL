from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask

from copy import deepcopy
import logging
from overrides import overrides
import torch

from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer

from typing import Optional, Any, Dict, List

from models.dependency_decoder import DependencyDecoder
from models.tag_decoder import TagDecoder
from modules.bert_pretrained import PretrainedBertIndexer
from modules.scalar_mix import ScalarMixWithDropout
from modules.text_field_embedder import HCDPTextFieldEmbedder

logger = logging.getLogger('mylog')


class MultitaskModel(Model):
    """
    The Multitask model base class. Applies a sequence of shared encoders before decoding in a multi-task configuration.
    Uses TagDecoder and DependencyDecoder to decode each task.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 tasks: List[str],
                 task_levels: List[int],
                 task_weights_for_loss: List[float] = None,
                 tag_representation_dim = 256,
                 arc_representation_dim = 768,
                 dropout: float = 0.0,
                 bert_dropout: float = 0.2,
                 layer_dropout: float = 0.0,
                 label_smoothing: float = 0.0,
                 mix_embedding: bool = False,
                 multitask_model_type: str = 'complex', # set 'simple' or 'complex'
                 unfreeze_bert: bool = True,
                 pretrained_model: str = "bert-base-cased",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(MultitaskModel, self).__init__(vocab, regularizer)

        self.pretrained_model = pretrained_model
        self.multitask_model_type = multitask_model_type

        # Set tasks and task weights
        self.tasks = tasks
        self.task_weights_for_loss =  task_weights_for_loss

        # Set output layers for each task
        if task_levels is None:
            task_levels = [-1 for _ in tasks]
        self.task_to_layer = dict(zip(tasks, task_levels))

        # Set vocab
        self.vocab = vocab

        # Set layer dropout for ScalarMixWithDropout
        self.layer_dropout = layer_dropout

        # Set pretrained model
        if 'roberta' in pretrained_model:
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
            self.bert_vocab = self.tokenizer.encoder
        else: # 'bert'
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            self.bert_vocab = self.tokenizer.vocab

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

        # Set the shared and unshared layers + the gating mechanisms
        if len(self.tasks) > 1 and self.multitask_model_type == 'complex':
            unshared_modules = {}
            shared_linear = {}
            unshared_linear = {}
            gating_modules = {}
            for task in self.tasks:
                # copying last 4 layers for the unshared layers
                # the shared layers will be up to the final 4 layers
                unshared_modules[task] = torch.nn.ModuleList([deepcopy(self.text_field_embedder.token_embedder_bert.bert_model.encoder.layer[i])
                                                              for i in range(-4, 0, 1)])
                # setting additional linear layers for shared and unshared outputs
                shared_linear[task] = torch.nn.modules.linear.Linear(768, 768)
                unshared_linear[task] = torch.nn.modules.linear.Linear(768, 768)
                # setting the final gating mechanism
                gating_modules[task] = Gating(2, 768)
            self.unshared_modules = torch.nn.ModuleDict(unshared_modules)
            self.shared_linear = torch.nn.ModuleDict(shared_linear)
            self.unshared_linear = torch.nn.ModuleDict(unshared_linear)
            self.gating_modules = torch.nn.ModuleDict(gating_modules)

        # Set the decoders
        decoders = {}
        for task in self.tasks:
            if task == 'deps':
                decoders[task] = DependencyDecoder(vocab=vocab,
                                                   tag_representation_dim=tag_representation_dim,
                                                   arc_representation_dim=arc_representation_dim,
                                                   label_smoothing=label_smoothing)

            else:
                decoders[task] = TagDecoder(vocab=vocab,
                                            task=task,
                                            output_dim=768,
                                            label_smoothing=label_smoothing)
        self.decoders = torch.nn.ModuleDict(decoders)

        # Set mix embeddings
        # In this case we do not use self.task_to_layer, and instead, we compute a parameterized weighted average over all layers
        if mix_embedding:
            # the output of a Transformer is 1 + num_hidden_layers since the embedding layer is included in the output
            mixture_size = 1 + self.text_field_embedder.token_embedder_bert.bert_model.config.num_hidden_layers
            self.scalar_mix = torch.nn.ModuleDict({
                task: ScalarMixWithDropout(mixture_size=mixture_size,
                                           do_layer_norm=False,
                                           layer_dropout=self.layer_dropout)
                for task in self.decoders if task in ["ner", "deps"]
            })
        else:
            self.scalar_mix = None

        # Initialize metrics
        self.metrics = {}

        # Initialize parameters
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

        # Embed tokens using the pretrained model
        mask = get_text_field_mask(tokens)
        embedded_text_input = self.text_field_embedder(tokens)

        # Decode each of the tasks
        for task_idx, task in enumerate(self.tasks):
            # Get decoding input (either by scalar_mix or by task_to_layer)
            if self.scalar_mix and task in self.scalar_mix:
                decoder_input = self.scalar_mix[task](embedded_text_input, mask)
            else:
                decoding_layer = self.task_to_layer[task] if task in self.task_to_layer else -1
                decoder_input = embedded_text_input[decoding_layer]

            # Merge shared and unshared outputs
            if len(self.tasks) > 1 and self.multitask_model_type == 'complex':
                unshared_output = embedded_text_input[-5]
                for unshared_layer in self.unshared_modules[task]:
                    unshared_output = unshared_layer(unshared_output)[0]
                shared_output = self.shared_linear[task](decoder_input)
                unshared_output = self.unshared_linear[task](unshared_output)
                decoder_input = self.gating_modules[task](tuple([shared_output, unshared_output]))

            if task == "deps":
                pred_output = self.decoders[task](encoded_text=decoder_input,
                                                  mask=mask,
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
                if self.task_weights_for_loss is None:
                    loss += pred_output["loss"]
                else:
                    loss += self.task_weights_for_loss[task_idx] * pred_output["loss"]

        # Add loss to output_dict
        if gold_tags:
            output_dict["loss"] = loss
            output_dict["loss_dict"] = loss_dict

        # Add metadata to output_dict
        if metadata is not None:
            for x in metadata:
                for k, v in x.items():
                    if k not in output_dict:
                        output_dict[k] = []
                    output_dict[k].append(v)
        return output_dict

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

class Gating(torch.nn.Module):
    # Implementation of:
    # Sato, Motoki, et al. "Adversarial training for cross-domain universal dependency parsing."
    # Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal
    # Dependencies. 2017.‏;
    # and of:
    # Rotman, Guy, and Roi Reichart. "Deep contextualized self-training for low resource dependency parsing."
    # Transactions of the Association for Computational Linguistics 7 (2019): 695-713.

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