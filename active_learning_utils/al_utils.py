import numpy as np
import torch

def entropy(x):
    entropy = - 1.0 * torch.sum((x + 1e-32) * torch.log((x + 1e-32)), axis=(-2, -1))
    return entropy

def median_entropy(x):
    token_entropy = - 1.0 * torch.sum((x + 1e-32) * torch.log((x + 1e-32)), axis=-1)
    entropies = []
    for token_row in token_entropy:
        vals = []
        for val in token_row:
            if val > 0.0:
                vals.append(val)
        entropies.append(torch.stack(vals))
    entropy = torch.stack([torch.median(entropy_) for entropy_ in entropies])
    return entropy

def compute_dependency_entropy(head_preds, tag_preds, lengths):
    heads_entropy = entropy(head_preds[:, 1:, :])
    class_logs = torch.log((lengths).type(head_preds.dtype))
    class_logs[class_logs == 0] = 1.
    heads_entropy = heads_entropy / ((lengths) * class_logs)
    # tag_entropy = entropy(tag_preds[:, 1:, :])
    # num_tags = torch.tensor(tag_preds.size(-1) - 1, dtype=tag_entropy.dtype, device=tag_entropy.device)
    # class_logs_tags = torch.log(num_tags)
    # class_logs_tags[class_logs_tags == 0] = 1.
    # tag_entropy = tag_entropy / (lengths * class_logs_tags)
    # dependency_entropy = (heads_entropy + tag_entropy) / 2
    dependency_entropy = heads_entropy
    return dependency_entropy

def compute_sequence_label_entropy(sl_preds, lengths):
    sl_entropy = entropy(sl_preds)
    num_tags = torch.tensor(sl_preds.size(-1), dtype=sl_entropy.dtype, device=sl_entropy.device)
    class_logs = torch.log(num_tags)
    class_logs[class_logs == 0] = 1.
    sl_entropy = sl_entropy / (lengths * class_logs)
    return sl_entropy




