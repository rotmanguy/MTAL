from io_.io_utils import clean_batch
import logging
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm, trange
from utils.utils import apply_temperature, set_seed

logger = logging.getLogger('mylog')

def train_temperature(args, model, dev_dataloader):
    """
    Tune the temperature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    nll_loss_total = torch.tensor(0.0)
    epoch_iterator = tqdm(dev_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        batch = clean_batch(args, batch, mode='train')
        outputs = model(**batch)
        nll_loss = outputs['loss']
        if args.n_gpu > 1:
            nll_loss = nll_loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            nll_loss = nll_loss / args.gradient_accumulation_steps
        nll_loss_total += nll_loss
    return nll_loss_total

def train_temperature_adam(args, model, dev_dataloader, optimizer, amp):
    """
    Tune the temperature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    model.zero_grad()
    train_iterator = trange(int(10), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(dev_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = clean_batch(args, batch, mode='train')
            outputs = model(**batch)
            loss = outputs['loss']
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
    return model

def eval_ece(args, model, dev_dataloader, ece_criterion):
    # # Calculate NLL and ECE after temperature scaling
    head_tag_logits_list = []
    head_tags_list = []
    ner_logits_list = []
    ner_tags_list = []
    tr_loss = 0.0
    epoch_iterator = tqdm(dev_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            batch = clean_batch(args, batch, mode='train')
            outputs = model(**batch)
            nll_loss = outputs['loss']
            if args.n_gpu > 1:
                nll_loss = nll_loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                nll_loss = nll_loss / args.gradient_accumulation_steps

            if 'deps' in args.tasks:
                tag_logits = outputs['tag_logits'][:, 1:, :]  # remove root logits
                head_tag_logits_list.append(tag_logits.reshape(tag_logits.size(0) * tag_logits.size(1), -1))
                head_tags_list.append(batch['head_tags'].reshape(-1))
            if 'ner' in args.tasks:
                ner_logits_list.append(
                    outputs['ner_logits'].reshape(outputs['ner_logits'].size(0) * outputs['ner_logits'].size(1),
                                                  -1))
                ner_tags_list.append(batch['ner'].reshape(-1))
            tr_loss += nll_loss.item()

    head_tag_logits_list = torch.cat(head_tag_logits_list) if head_tag_logits_list else None
    head_tags_list = torch.cat(head_tags_list) if head_tags_list else None
    ner_logits_list = torch.cat(ner_logits_list) if ner_logits_list else None
    ner_tags_list = torch.cat(ner_tags_list) if ner_tags_list else None
    # Calculate NLL and ECE after temperature scaling
    logger.info('temperature - NLL: %.3f' % (tr_loss))
    if head_tag_logits_list is not None and head_tags_list is not None:
        deps_temperature_ece = ece_criterion(head_tag_logits_list, head_tags_list).item()
        logger.info('temperature - DEPS ECE: %.3f' % (deps_temperature_ece))
    if ner_logits_list is not None and ner_tags_list is not None:
        ner_temperature_ece = ece_criterion(ner_logits_list, ner_tags_list).item()
        logger.info('temperature - NER ECE: %.3f' % (ner_temperature_ece))

def set_temperature(args, vocab, model_, dev_dataloader):
    # def closure():
    #     loss = train_temperature(args, model, dev_dataloader)
    #     loss.backward()
    #     return loss

    ece_criterion = _ECELoss().to(args.device)
    model, optimizer, amp = apply_temperature(args, vocab, model_)
    eval_ece(args, model, dev_dataloader, ece_criterion)
    #optimizer.step(closure)
    model = train_temperature_adam(args, model, dev_dataloader, optimizer, amp)
    eval_ece(args, model, dev_dataloader, ece_criterion)
    return model

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece