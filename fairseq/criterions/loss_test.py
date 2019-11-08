# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion

import torch
import pdb


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    non_pad_mask = target.ne(ignore_index)
    nll_loss = nll_loss[non_pad_mask]

    mask = torch.ones(lprobs.shape).long().cuda()
    helper_ix = torch.arange(target.shape[0]).unsqueeze(0).T.expand(target.shape)
    mask[helper_ix, target] = 0
    mask = mask.bool()
    smooth_loss = -lprobs[mask]

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)

    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('loss_test')
class LossTest(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True, print_recall=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, recall = self.compute_loss(model, net_output, sample, reduce=reduce, print_recall=print_recall)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        if recall is not None:
            logging_output['recall'] = utils.item(recall.data) if reduce else recall.data
            print(recall.data / sample['target'].size(0))

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, print_recall=False):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        # target = model.get_targets(sample, net_output).view(-1, 1)
        target = model.get_targets(sample, net_output)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        recall = None
        if print_recall:
            recall = self.get_recall(lprobs, target, top_k=500)
        return loss, nll_loss, recall

    def get_recall(self, lprobs, target, top_k=1000):
        top = lprobs.sort(descending=True)[1][:, :top_k]
        tgt_vocab = target

        totals = torch.zeros(len(lprobs)).cuda()
        corrects = torch.zeros(len(lprobs)).cuda()
        for idx, tgt in enumerate(tgt_vocab):
            for token in tgt:
                if token <= 3:
                    continue

                totals[idx] += 1
                if token in top[idx]:
                    corrects[idx] += 1

        recalls = corrects / totals

        return recalls.sum()


    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        return {
            'recall': sum(log.get('recall', 0) for log in logging_outputs) / nsentences if sample_size > 0 else 0.,
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
