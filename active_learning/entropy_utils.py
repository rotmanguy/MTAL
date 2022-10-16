import torch

def entropy(x):
    """
    Compute sentence-level entropy for a sequence of probabilities.
    :param x: tensor.
    :return: tensor's entropy.
    """
    entropy = - 1.0 * torch.sum((x + 1e-32) * torch.log((x + 1e-32)), axis=(-2, -1))
    return entropy

def compute_dependency_entropy(head_preds, tag_preds, lengths):
    """
    Computing the normalized sentence-level entropy for the outputs of dependency parsing.
    We currently compute the entropy only for head_preds and keep in comment the entropy of tag_preds.
    :param head_preds: head probabilities.
    :param tag_preds: tag probabilities.
    :param lengths: lengths of each sentence in the batch.
    :return: the normalized sentence-level entropy for the outputs of dependency parsing.
    """
    # Compute sentence-level entropy
    heads_entropy = entropy(head_preds[:, 1:, :])
    # The following comments consider the tag entropies as well for future use:
    # tag_entropy = entropy(tag_preds[:, 1:, :])
    # num_tags = torch.tensor(tag_preds.size(-1) - 1, dtype=tag_entropy.dtype, device=tag_entropy.device)
    # class_logs_tags = torch.log(num_tags)
    # class_logs_tags[class_logs_tags == 0] = 1.
    # tag_entropy = tag_entropy / (lengths * class_logs_tags)

    # Normalize the entropy
    class_logs = torch.log((lengths).type(head_preds.dtype))
    class_logs[class_logs == 0] = 1.
    heads_entropy = heads_entropy / (lengths * class_logs)

    dependency_entropy = heads_entropy
    # dependency_entropy = (heads_entropy + tag_entropy) / 2
    return dependency_entropy

def compute_sequence_label_entropy(sl_preds, lengths):
    """
    Computing the normalized sentence-level entropy for the outputs of a sequence labeling task.
    :param sl_preds: sequence label probabilities.
    :param lengths: lengths of each sentence in the batch.
    :return: the normalized sentence-level entropy for the outputs of the sequence labeling task.
    """
    # Compute sentence-level entropy
    sl_entropy = entropy(sl_preds)

    # Normalize the entropy
    num_tags = torch.tensor(sl_preds.size(-1), dtype=sl_entropy.dtype, device=sl_entropy.device)
    class_logs = torch.log(num_tags)
    class_logs[class_logs == 0] = 1.
    sl_entropy = sl_entropy / (lengths * class_logs)
    return sl_entropy




