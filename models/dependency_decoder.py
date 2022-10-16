"""
Decodes dependency trees given a list of contextualized word embeddings
"""
import sys
from typing import Dict, Optional, Tuple
import logging
import copy

from overrides import overrides
import torch
import torch.nn.functional as F
import numpy

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from allennlp.nn.util import get_range_vector, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_device_of, masked_softmax, get_lengths_from_binary_sequence_mask
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.training.metrics import AttachmentScores

logger = logging.getLogger('mylog')

class DependencyDecoder(Model):
    """
    Modifies BiaffineDependencyParser, removing the input TextFieldEmbedder dependency to allow the model to
    essentially act as a decoder when given intermediate word embeddings instead of as a standalone model.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 tag_representation_dim: int,
                 arc_representation_dim: int,
                 use_mst_decoding_for_validation: bool = True,
                 dropout: float = 0.0,
                 label_smoothing: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DependencyDecoder, self).__init__(vocab, regularizer)

        encoder_output_dim = arc_representation_dim

        self.dropout = torch.nn.Dropout(p=dropout)

        self.label_smoothing = label_smoothing

        self.head_arc_feedforward = FeedForward(encoder_output_dim, 1,
                                                arc_representation_dim,
                                                Activation.by_name("elu")())
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(arc_representation_dim,
                                                     arc_representation_dim,
                                                     use_input_biases=True)

        num_labels = self.vocab.get_vocab_size("head_tags")

        self.head_tag_feedforward = FeedForward(encoder_output_dim, 1,
                                                tag_representation_dim,
                                                Activation.by_name("elu")())
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = torch.nn.modules.Bilinear(tag_representation_dim,
                                                      tag_representation_dim,
                                                      num_labels)

        self._dropout = InputVariationalDropout(dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, encoder_output_dim]))

        check_dimensions_match(tag_representation_dim, self.head_tag_feedforward.get_output_dim(),
                               "tag representation dim", "tag feedforward output dim")
        check_dimensions_match(arc_representation_dim, self.head_arc_feedforward.get_output_dim(),
                               "arc representation dim", "arc feedforward output dim")

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        self._attachment_scores = AttachmentScores()
        if initializer is not None:
            initializer(self)

    def forward(self,
                encoded_text: torch.FloatTensor,
                mask: torch.LongTensor,
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        batch_size, _, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)
        float_mask = mask.float()
        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(head_arc_representation,
                                           child_arc_representation)

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads, predicted_head_tags = self._greedy_decode(head_tag_representation,
                                                                       child_tag_representation,
                                                                       attended_arcs,
                                                                       mask)
            energy = None
        else:
            (predicted_heads, predicted_head_tags), energy = self._mst_decode(head_tag_representation,
                                                                              child_tag_representation,
                                                                              attended_arcs,
                                                                              mask)
        predicted_heads = predicted_heads.to(encoded_text.device)
        predicted_head_tags = predicted_head_tags.to(encoded_text.device)
        # if head_indices is not None and head_tags is not None:
        if self.training:
            head_logits, tag_logits, head_preds, tag_preds, arc_nll, tag_nll = \
                self._construct_loss(head_tag_representation=head_tag_representation,
                                     child_tag_representation=child_tag_representation,
                                     attended_arcs=attended_arcs,
                                     head_indices=head_indices,
                                     head_tags=head_tags,
                                     mask=mask)
            loss = arc_nll + tag_nll

            # We calculate attachment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._attachment_scores(predicted_heads[:, 1:],
                                    predicted_head_tags[:, 1:],
                                    head_indices[:, 1:],
                                    head_tags[:, 1:],
                                    mask[:, 1:])
        else:
            head_logits, tag_logits, head_preds, tag_preds, arc_nll, tag_nll = \
                self._construct_loss(head_tag_representation=head_tag_representation,
                                     child_tag_representation=child_tag_representation,
                                     attended_arcs=attended_arcs,
                                     head_indices=predicted_heads.long(),
                                     head_tags=predicted_head_tags.long(),
                                     mask=mask)
            loss = arc_nll + tag_nll

        output_dict = {
            "predicted_heads": predicted_heads,
            "predicted_head_tags": predicted_head_tags,
            "head_logits": head_logits,
            "tag_logits": tag_logits,
            "head_preds": head_preds,
            "tag_preds": tag_preds,
            "energy": energy,
            "loss": loss,
            "mask": mask,
        }
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        head_tags = output_dict["predicted_head_tags"].cpu().detach().numpy()
        heads = output_dict["predicted_heads"].cpu().detach().numpy()
        mask = output_dict["mask"]
        lengths = get_lengths_from_binary_sequence_mask(mask)
        head_tag_labels = []
        head_indices = []
        for instance_heads, instance_tags, length in zip(heads, head_tags, lengths):
            instance_heads = list(instance_heads[1:length])
            instance_tags = instance_tags[1:length]
            labels = [self.vocab.get_token_from_index(label, "head_tags")
                      for label in instance_tags]
            head_tag_labels.append(labels)
            head_indices.append(instance_heads)

        output_dict["predicted_head_tags"] = head_tag_labels
        output_dict["predicted_heads"] = head_indices
        return output_dict

    def _construct_loss(self,
                        head_tag_representation: torch.Tensor,
                        child_tag_representation: torch.Tensor,
                        attended_arcs: torch.Tensor,
                        head_indices: torch.Tensor,
                        head_tags: torch.Tensor,
                        mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.
        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.
        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs)).unsqueeze(1)

        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = self.masked_log_softmax(attended_arcs,
                                                        mask) * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(attended_arcs.new(mask.size(1)).fill_(-sys.maxsize - 1))
        head_preds = masked_softmax(attended_arcs,  mask)

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation, child_tag_representation, head_indices)

        normalised_head_tag_logits = self.masked_log_softmax(head_tag_logits,
                                                             mask.unsqueeze(-1)) * float_mask.unsqueeze(-1)
        tag_preds = masked_softmax(head_tag_logits, mask.unsqueeze(-1))

        if self.label_smoothing is not None and self.label_smoothing > 0.0:
            mask_copy = copy.deepcopy(mask)
            mask_copy[:, 0] = 0.0
            arc_nll = sequence_cross_entropy_with_logits(normalised_arc_logits,
                                                         head_indices,
                                                         mask_copy,
                                                         label_smoothing=self.label_smoothing)
            tag_nll = sequence_cross_entropy_with_logits(normalised_head_tag_logits,
                                                         head_tags,
                                                         mask_copy,
                                                         label_smoothing=self.label_smoothing)
        else:
            # index matrix with shape (batch, sequence_length)
            timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
            child_index = timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
            # shape (batch_size, sequence_length)
            arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
            tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
            # We don't care about predictions for the symbolic ROOT token's head,
            # so we remove it from the loss.
            arc_loss = arc_loss[:, 1:]
            tag_loss = tag_loss[:, 1:]

            # The number of valid positions is equal to the number of unmasked elements minus
            # 1 per sequence in the batch, to account for the symbolic HEAD token.
            valid_positions = mask.sum() - batch_size

            arc_nll = -arc_loss.sum() / valid_positions.float()
            tag_nll = -tag_loss.sum() / valid_positions.float()

        return attended_arcs, head_tag_logits, head_preds, tag_preds, arc_nll, tag_nll

    def _greedy_decode(self,
                       head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                       attended_arcs: torch.Tensor,
                       mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).
        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).bool().unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag_representation,
                                              child_tag_representation,
                                              heads)
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def _mst_decode(self,
                    head_tag_representation: torch.Tensor,
                    child_tag_representation: torch.Tensor,
                    attended_arcs: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.
        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels, sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(0, 3, 1, 2)

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = (1 - mask.float()) * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits)
        return (self._run_mst_decoding(batch_energy, lengths), batch_energy)

    @staticmethod
    def _run_mst_decoding(batch_energy: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necesarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return torch.from_numpy(numpy.stack(heads)), torch.from_numpy(numpy.stack(head_tags))

    def _get_head_tags(self,
                       head_tag_representation: torch.Tensor,
                       child_tag_representation: torch.Tensor,
                       head_indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.
        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.
        Returns
        -------
        head_tag_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(batch_size, get_device_of(head_tag_representation)).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(selected_head_tag_representations,
                                            child_tag_representation)
        return head_tag_logits

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {f".run/deps/{metric_name}": metric
                for metric_name, metric in self._attachment_scores.get_metric(reset).items()}

    def masked_log_softmax(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
        masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
        ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
        ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
        broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
        unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
        do it yourself before passing the mask into this function.
        In the case that the input vector is completely masked, the return value of this function is
        arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
        of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
        that we deal with this case relies on having single-precision floats; mixing half-precision
        floats with fully-masked vectors will likely give you ``nans``.
        If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
        lower), the way we handle masking here could mess you up.  But if you've got logit values that
        extreme, you've got bigger problems than this.
        """
        if mask is not None:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
            # results in nans when the whole vector is masked.  We need a very small value instead of a
            # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
            # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
            # becomes 0 - this is just the smallest value we can actually use.
            vector = vector + (mask + 1e-32).log()
        return torch.nn.functional.log_softmax(vector, dim=dim)
