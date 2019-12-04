# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion

import pdb

@register_criterion('binary_cross_entropy')
class BinaryCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True, print_recall=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output, expand_steps=False).float()

        if hasattr(model, 'get_target_weights'):
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()
        else:
            weights = 1.

        loss = F.binary_cross_entropy_with_logits(logits, target, reduce=False)
        loss = loss * weights

        if reduce:
            loss = loss.sum()

        sample_size = target.numel()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample_size,
            'nsentences': logits.size(0),
            'sample_size': sample_size,
        }

        if print_recall:
            recall = self.get_recall(logits, target, top_k=3000)
            logging_output['recall'] = utils.item(recall.data) if reduce else recall.data
            print(recall.data.float() / logits.size(0))

        return loss, sample_size, logging_output


    def get_recall(self, lprobs, target, top_k=1000):
        lprobs = lprobs.sort(descending=True)[1][:, :top_k]
        # tgt_vocab = target.nonzero()

        pdb.set_trace()

        totals = torch.zeros(len(lprobs)).cuda()
        corrects = torch.zeros(len(lprobs)).cuda()
        for idx, tgt in enumerate(target):
            for token in tgt:
                totals[idx] += 1
                if token.int() in lprobs[idx]:
                    corrects[idx] += 1

        k = lprobs.shape[1]

        recall = (corrects / totals).mean().item() * 100
        perfect_recall = (corrects == totals).float().mean().item() * 100

        print("k: {}, recall: {:.3f}, perfect recall: {:.3f}".format(k, recall, perfect_recall))

        return (corrects / totals).sum()


    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        recall_sum = sum(log.get('recall', 0) for log in logging_outputs)
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'recall': recall_sum / nsentences,
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
