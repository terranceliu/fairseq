# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import data_utils, FairseqDataset, LanguagePairDataset
from .language_pair_dataset import collate as collate_orig

import pdb


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            print("| alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        # if input_feeding:
        #     # we create a shifted version of targets for feeding the
        #     # previous output token(s) into the next decoder step
        #     prev_output_tokens = merge(
        #         'target',
        #         left_pad=left_pad_target,
        #         move_eos_to_beginning=True,
        #     )
        #     prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    target_vocab = None
    if samples[0].get('target_vocab', None) is not None:
        target_vocab = merge('target_vocab', left_pad=None)
        target_vocab = target_vocab.index_select(0, sort_order)

    target_mapped = None
    if samples[0].get('target_mapped', None) is not None:
        target_mapped = merge('target_mapped', left_pad=left_pad_target)
        target_mapped = target_mapped.index_select(0, sort_order)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target_mapped',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    if target_vocab is not None:
        batch['net_input']['target_vocab'] = target_vocab
    if target_mapped is not None:
        batch['target_mapped'] = target_mapped

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch

# def collate(samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
#                input_feeding=True,):
#     batch = collate_orig(samples, pad_idx, eos_idx, left_pad_source, left_pad_target, input_feeding)
#     # pdb.set_trace()
#     #
#     # print(samples)
#     # print(samples[0].keys())
#
#     batch['target_vocab'] = samples[0].get('target_vocab')
#     batch['target_mapped'] = samples[0].get('target_mapped')
#
#     # target_vocab = samples[0].get('target_vocab')
#
#     return batch

class LanguagePairDatasetED(LanguagePairDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None, tgt_vocab_size=None,
    ):
        super().__init__(src, src_sizes, src_dict,
            tgt=tgt, tgt_sizes=tgt_sizes, tgt_dict=tgt_dict,
            left_pad_source=left_pad_source, left_pad_target=left_pad_target,
            max_source_positions=max_source_positions, max_target_positions=max_target_positions,
            shuffle=shuffle, input_feeding=input_feeding,
            remove_eos_from_source=remove_eos_from_source, append_eos_to_target=append_eos_to_target,
            align_dataset=align_dataset)
        self.tgt_vocab_size = tgt_vocab_size
        self.tgt_vocab = None
        self.tgt_mapped = None

        if tgt_vocab_size is not None:
            self.generate_tgt_vocab()
            self.generate_tgt_mapped()

    def __getitem__(self, index):
        example = super().__getitem__(index)
        if self.tgt_vocab_size is not None:
            example['target_vocab'] = self.tgt_vocab[index]
            example['target_mapped'] = self.tgt_mapped[index]
        return example

    def generate_tgt_vocab(self):
        top_tokens = torch.arange(self.tgt_vocab_size)

        self.tgt_vocab = torch.ones((len(self.tgt), self.tgt_vocab_size)).long()
        for i in range(len(self.tgt)):
            tgt_tokens = self.tgt[i]
            tgt_tokens = torch.unique(tgt_tokens)
            assert (tgt_tokens.shape[0] <= self.tgt_vocab_size)

            ix = top_tokens.view(1, -1).eq(tgt_tokens.view(-1, 1)).sum(0) == 0
            extra_tokens = top_tokens[ix]
            vocab_tokens = torch.cat((tgt_tokens, extra_tokens))[:self.tgt_vocab_size]
            vocab_tokens, _ = torch.sort(vocab_tokens)

            self.tgt_vocab[i] = vocab_tokens

    def generate_tgt_mapped(self):
        self.tgt_mapped = []

        for i in range(len(self.tgt)):
            target = self.tgt[i]
            tgt_vocab = self.tgt_vocab[i]

            # Just added for now. Probably don't need
            if self.append_eos_to_target:
                eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
                if self.tgt and target[-1] != eos:
                    target = torch.cat([target, torch.LongTensor([eos])])

            index = np.argsort(tgt_vocab)
            sorted_x = tgt_vocab[index]
            sorted_index = np.searchsorted(sorted_x, target)

            yindex = np.take(index, sorted_index, mode="clip")
            mask = tgt_vocab[yindex] != target

            result = np.ma.array(yindex, mask=mask).data
            self.tgt_mapped.append(torch.tensor(result))

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )



