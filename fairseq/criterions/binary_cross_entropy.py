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

        # size_ns = 10000
        # mask = torch.zeros((len(target), size_ns)).long().cuda()
        # mask = mask.random_(0, size_ns)
        # for idx, tgt in enumerate(target):
        #     positives = tgt.nonzero().flatten()
        #     num_positives = len(positives)
        #     mask[idx, :num_positives] = positives
        #
        # logits = logits.gather(dim=-1, index=mask)
        # weights = weights.gather(dim=-1, index=mask)
        # target = target.gather(dim=-1, index=mask)

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
            recall = self.get_recall(logits, target, top_k=500)
            logging_output['recall'] = utils.item(recall.data) if reduce else recall.data
            print(recall.data.float() / logits.size(0))

        return loss, sample_size, logging_output


    def get_recall(self, lprobs, target, top_k=1000):
        top = lprobs.sort(descending=True)[1][:, :top_k]
        tgt_vocab = target.nonzero()

        totals = torch.zeros(len(lprobs)).cuda()
        corrects = torch.zeros(len(lprobs)).cuda()
        for tgt in tgt_vocab:
            idx = tgt[0]
            token = tgt[1]

            if token <= 3:
                continue

            totals[idx] += 1


            # top_tokens = torch.arange(1000).cuda()
            # ix = top_tokens.view(1, -1).eq(top[idx].view(-1, 1)).sum(0) == 0
            # extra_tokens = top_tokens[ix]
            # vocab_tokens = torch.cat((top[idx], extra_tokens))[:1000]

            # if token < top_k:
            if token in top[idx]:
            # if token in top[idx] or token < 500:
            # if token in vocab_tokens:
                corrects[idx] += 1

        recalls = corrects / totals
        # recalls = corrects == totals

        return recalls.sum()


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
