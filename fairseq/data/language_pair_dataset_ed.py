# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch

from . import data_utils, FairseqDataset, LanguagePairDataset

from tqdm import tqdm
from datetime import timedelta
import time

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

    if samples[0].get('source_roberta', None) is not None:
        src_tokens = merge('source_roberta', left_pad=left_pad_source)
    else:
        src_tokens = merge('source', left_pad=left_pad_source)

    id = torch.LongTensor([s['id'] for s in samples])
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

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    target_vocab = None
    if samples[0].get('target_vocab', None) is not None:
        target_vocab = merge('target_vocab', left_pad=None)
        target_vocab = target_vocab.index_select(0, sort_order)

    target_old = None
    if samples[0].get('target_old', None) is not None:
        target_old = merge('target_old', left_pad=left_pad_target)
        target_old = target_old.index_select(0, sort_order)

    target_mapped = None
    if samples[0].get('target_mapped', None) is not None:
        target_mapped = merge('target_mapped', left_pad=left_pad_target)
        target_mapped = target_mapped.index_select(0, sort_order)
        ntokens = sum(len(s['target_mapped']) for s in samples)

    target_vocab_bow = None
    if samples[0].get('target_vocab_bow', None) is not None:
        target_vocab_bow = merge('target_vocab_bow', left_pad=None)
        target_vocab_bow = target_vocab_bow.index_select(0, sort_order)

    target_vocab_nopad = None
    if samples[0].get('target_vocab_nopad', None) is not None:
        target_vocab_nopad = merge('target_vocab_nopad', left_pad=None)
        target_vocab_nopad = target_vocab_nopad.index_select(0, sort_order)

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
    if target_old is not None:
        batch['target_old'] = target_old
    if target_mapped is not None:
        batch['target_mapped'] = target_mapped
    if target_vocab_bow is not None:
        batch['target_vocab_bow'] = target_vocab_bow
    if target_vocab_nopad is not None:
        batch['target_vocab_nopad'] = target_vocab_nopad

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
        self, split, data_path,
        src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        tgt_vocab_size=None, oracle=False, num_extra_bpe=0,
    ):
        super().__init__(src, src_sizes, src_dict,
            tgt=tgt, tgt_sizes=tgt_sizes, tgt_dict=tgt_dict,
            left_pad_source=left_pad_source, left_pad_target=left_pad_target,
            max_source_positions=max_source_positions, max_target_positions=max_target_positions,
            shuffle=shuffle, input_feeding=input_feeding,
            remove_eos_from_source=remove_eos_from_source, append_eos_to_target=append_eos_to_target,
            align_dataset=align_dataset)

        self.split = split
        self.data_path = data_path
        self.preprocessed_path = os.path.join(data_path, "ed_preprocessed")
        if not os.path.isdir(self.preprocessed_path):
            os.mkdir(self.preprocessed_path)

        self.tgt_vocab_size = tgt_vocab_size
        self.oracle = oracle
        self.num_extra_bpe = num_extra_bpe

        self.tgt_old = self.tgt

        self.tgt_vocab = None
        self.tgt_mapped = None
        self.top_logits = None
        self.extra_bpe_tokens = None
        self.bpe2idx = None

        self.mandatory_tokens = torch.tensor(
            [self.tgt_dict.bos(), self.tgt_dict.pad(), self.tgt_dict.eos(), self.tgt_dict.unk()])

        if self.tgt_vocab_size is not None:
            if self.oracle:
                self.generate_tgt_vocab()
                self.generate_tgt_mapped()
            else:
                if self.num_extra_bpe > 0:
                    self.generate_extra_bpe()
                    assert(len(self.bpe2idx.keys()) <= self.tgt_vocab_size)

                path = 'checkpoints/iswlt14_de-en/vp_lstm_onevsall/checkpoint_best.pt'
                # path = 'checkpoints/wmt16.en-de.joined-dict.newstest2014_ott2018/vp_lstm_onevsall/checkpoint_best.pt'
                # path = 'checkpoints/wmt16.en-de.joined-dict.newstest2014_ott2018/vp_mlp_binary_bndo/checkpoint_best.pt'
                # path = 'checkpoints/wmt14_en_de/vp_lstm/checkpoint_best.pt'

                self.generate_top_logits(path)
                self.generate_tgt_vocab_vp()
                self.generate_tgt_mapped_vp()

                # self.get_recall(self.tgt_vocab, self.tgt)
                # pdb.set_trace()

        # DEBUGGING
        # self.test_get_bpe_target()
        # self.verify_tgt_mapped_vp()
        # pdb.set_trace()


    def __getitem__(self, index):
        example = super().__getitem__(index)
        example['target_old'] = self.tgt_old[index]
        if self.tgt_vocab is not None:
            example['target_vocab'] = self.tgt_vocab[index]
            example['target_mapped'] = self.tgt_mapped[index]

        # if self.split == 'train' and np.random.rand() < 0.5:
        #     prob = np.random.rand()
        #     target = example['target']
        #     tgt_vocab = example['target_vocab']
        #     mask = np.random.choice(a=[True, False], size=len(target), p=[prob, 1 - prob])
        #
        #     new_target = self.get_bpe_target(target, mask)
        #     new_target_mapped = self.get_mapped_target(new_target, tgt_vocab)
        #
        #     example['target'] = new_target
        #     example['target_mapped'] = new_target_mapped

        return example


    def generate_tgt_vocab(self):
        filepath = os.path.join(self.preprocessed_path, "{}.tgt_vocab{}.pt".format(
            self.split, self.tgt_vocab_size))

        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("Found preprocessed file for tgt_vocab!")
            self.tgt_vocab = torch.load(filepath)
        else:
            print("Generating tgt_vocab...")
            self.tgt_vocab = torch.ones((len(self.tgt), self.tgt_vocab_size)).long()
            self.top_tokens = torch.arange(self.tgt_vocab_size)

            for i in range(len(self.tgt)):
                tgt_tokens = self.tgt[i]
                tgt_tokens = torch.cat((tgt_tokens, self.mandatory_tokens))
                tgt_tokens = torch.unique(tgt_tokens)
                assert (tgt_tokens.shape[0] <= self.tgt_vocab_size)

                # if self.oracle:
                #     self.tgt_vocab[i, :tgt_tokens.shape[0]] = tgt_tokens
                # else:

                ix = self.top_tokens.view(1, -1).eq(tgt_tokens.view(-1, 1)).sum(0) == 0
                remaining_tokens = self.top_tokens[ix]
                vocab_tokens = torch.cat((tgt_tokens, remaining_tokens))[:self.tgt_vocab_size]
                vocab_tokens, _ = torch.sort(vocab_tokens)

                self.tgt_vocab[i] = vocab_tokens

            torch.save(self.tgt_vocab, filepath)


    def generate_tgt_mapped(self):
        filepath = os.path.join(self.preprocessed_path, "{}.tgt_mapped{}.pt".format(
            self.split, self.tgt_vocab_size))

        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("Found preprocessed file for tgt_mapped!")
            self.tgt_mapped = torch.load(filepath)
        else:
            print("Generating tgt_mapped...")
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

            torch.save(self.tgt_mapped, filepath)


    def generate_extra_bpe(self):
        self.extra_bpe_tokens = []
        self.bpe2idx = {}
        for i in range(len(self.tgt_dict)):
            token = self.tgt_dict[i]
            if '&' not in token and i not in self.mandatory_tokens:
                x = token.replace('@', '')
                if len(x) > self.num_extra_bpe:
                    continue
            self.extra_bpe_tokens.append(i)
            self.bpe2idx[token] = i
        self.extra_bpe_tokens = torch.tensor(self.extra_bpe_tokens)


    def generate_top_logits(self, path):
        filepath = os.path.join(self.preprocessed_path, "{}.vp_logits.pt".format(self.split))
        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("Found preprocessed file for top_logits (vp)!")
            self.top_logits = torch.load(filepath)
        else:
            print("Generating top_logits...")

            state = torch.load(path)
            args = state['args']
            model = self.vocab_task.build_model(args).cuda()
            model.load_state_dict(state['model'], strict=True)

            self.top_logits = []
            for src_tokens in tqdm(self.src):
                # net_output = model(src_tokens.unsqueeze(0), None, None)
                src_lengths = torch.tensor(len(src_tokens)).unsqueeze(0).cuda()
                src_tokens = src_tokens.unsqueeze(0).cuda()
                net_output = model(src_tokens, src_lengths, None)
                self.top_logits.append(model.get_logits(net_output).float().detach())
            self.top_logits = torch.cat(self.top_logits).detach().cpu()

            torch.save(self.top_logits, filepath)
            del model


    def generate_tgt_vocab_vp(self):
        filepath = os.path.join(self.preprocessed_path, "{}.tgt_vocab_vp_{}_extra{}.pt".format(
            self.split, self.tgt_vocab_size, self.num_extra_bpe))

        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("Found preprocessed file for tgt_vocab (vp)!")
            self.tgt_vocab = torch.load(filepath)
        else:
            print("Generating tgt_vocab...")

            self.tgt_vocab = self.top_logits.sort(descending=True)[1][:, :self.tgt_vocab_size]

            if self.num_extra_bpe > 0:
                for i in tqdm(range(len(self.tgt))):
                    ix = self.tgt_vocab[i].view(1, -1).eq(self.extra_bpe_tokens.view(-1, 1)).sum(0) == 0
                    remaining_tokens = self.tgt_vocab[i][ix]
                    vocab_tokens = torch.cat((self.extra_bpe_tokens, remaining_tokens))[:self.tgt_vocab_size]
                    self.tgt_vocab[i] = vocab_tokens

            elif True: #self.split in ['train']:
                for i in tqdm(range(len(self.tgt))):
                    tgt_vocab_true = torch.unique(self.tgt[i])
                    ix = self.tgt_vocab[i].view(1, -1).eq(tgt_vocab_true.view(-1, 1)).sum(0) == 0
                    remaining_tokens = self.tgt_vocab[i][ix]
                    vocab_tokens = torch.cat((tgt_vocab_true, remaining_tokens))[:self.tgt_vocab_size]
                    self.tgt_vocab[i] = vocab_tokens

            self.tgt_vocab, _ = self.tgt_vocab.sort(dim=1)
            torch.save(self.tgt_vocab, filepath)


    def generate_tgt_mapped_vp(self):
        filepath = os.path.join(self.preprocessed_path, "{}.tgt_mapped_vp_{}_extra{}.pt".format(
            self.split, self.tgt_vocab_size, self.num_extra_bpe))

        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("Found preprocessed file for tgt_mapped (vp)!")
            self.tgt_mapped, self.tgt, self.tgt_sizes = torch.load(filepath)
        else:
            print("Generating tgt_mapped...")

            tgt_mapped = []
            new_tgt = []
            new_tgt_sizes = []

            for i in tqdm(range(len(self.tgt))):
                target = self.tgt[i]
                tgt_vocab = self.tgt_vocab[i]

                # Just added for now. Probably don't need
                if self.append_eos_to_target:
                    eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
                    if self.tgt and target[-1] != eos:
                        target = torch.cat([target, torch.LongTensor([eos])])

                if self.num_extra_bpe > 0:
                    # reconstruct missing tokens. First try looking up 2-char bpe, then 1-char
                    mask = []
                    for i in target:
                        mask.append(i not in tgt_vocab)
                    mask = np.array(mask)

                    target = self.get_bpe_target(target, mask)

                target_mapped = self.get_mapped_target(target, tgt_vocab)

                new_tgt.append(target)
                new_tgt_sizes.append(len(target))
                tgt_mapped.append(target_mapped)

            self.tgt_mapped = tgt_mapped
            self.tgt = new_tgt
            self.tgt_sizes = np.array(new_tgt_sizes)

            torch.save((self.tgt_mapped, self.tgt, self.tgt_sizes), filepath)


    def generate_tgt_vocab_padded(self):
        filepath_pad = os.path.join(self.preprocessed_path, "{}.tgt_vocab_padded.pt".format(self.split))
        filepath_lengths = os.path.join(self.preprocessed_path, "{}.tgt_vocab_lengths.pt".format(self.split))

        if os.path.exists(filepath_pad) and os.path.isfile(filepath_pad):
            print("Found preprocessed file for tgt_vocab_padded!")
            self.tgt_vocab_padded = torch.load(filepath_pad)
            self.tgt_vocab_lengths = torch.load(filepath_lengths)
        else:
            print("Generating tgt_vocab_padded..")
            tgt_vocab_nopad = []
            for idx in tqdm(range(len(self.tgt))):
                tgt_tokens = self.tgt[idx]
                tgt_tokens = torch.cat((tgt_tokens, self.mandatory_tokens))
                tgt_tokens = torch.unique(tgt_tokens)
                tgt_vocab_nopad.append(tgt_tokens)

            self.tgt_vocab_lengths = torch.tensor([len(x) for x in tgt_vocab_nopad])
            max_length = self.tgt_vocab_lengths.max()

            self.tgt_vocab_padded = torch.ones((len(tgt_vocab_nopad), max_length)).long()
            for idx, tgt_vocab in enumerate(tqdm(tgt_vocab_nopad)):
                self.tgt_vocab_padded[idx, :len(tgt_vocab)] = tgt_vocab

            torch.save(self.tgt_vocab_padded, filepath_pad)
            torch.save(self.tgt_vocab_lengths, filepath_lengths)


    def get_bpe_target(self, target, mask):
        new_target = target.tolist()
        replace_indices = np.nonzero(mask)[0]
        replace_tokens = target[mask]

        for ix, token in enumerate(replace_tokens):
            token = self.tgt_dict[token]
            if '<' in token:
                continue
            if '&' in token:
                continue
            if len(token.replace('@', '')) <= 1:
                continue

            if token[-1] == '@':
                token_pt1 = token[:-3]
                token_pt2 = token[-3:]
            else:
                token_pt1 = token[:-1]
                token_pt2 = token[-1:]

            replacement_tokens = []
            for c in token_pt1:
                replacement_tokens.append(self.bpe2idx[c + '@@'])
            replacement_tokens.append(self.bpe2idx[token_pt2])

            new_target[replace_indices[ix]] = replacement_tokens

        temp = new_target
        new_target = []
        for x in temp:
            if isinstance(x, list):
                new_target += x
            else:
                new_target.append(x)

        return torch.tensor(new_target)


    def get_mapped_target(self, new_target, tgt_vocab):
        index = np.argsort(tgt_vocab)
        sorted_x = tgt_vocab[index]
        sorted_index = np.searchsorted(sorted_x, new_target)

        yindex = np.take(index, sorted_index, mode="clip")
        mask = tgt_vocab[yindex] != new_target

        result = np.ma.array(yindex, mask=mask).data

        return torch.tensor(result)


    def get_recall(self, lprobs, target):
        totals = torch.zeros(len(lprobs)).cuda()
        corrects = torch.zeros(len(lprobs)).cuda()
        for idx, tgt in enumerate(target):
            for token in tgt:
                totals[idx] += 1
                if token in lprobs[idx]:
                    corrects[idx] += 1

        k = lprobs.shape[1]

        recall = (corrects / totals).mean().item() * 100
        perfect_recall = (corrects == totals).float().mean().item() * 100

        print("k: {}, recall: {:.3f}, perfect recall: {:.3f}".format(k, recall, perfect_recall))

        return recall, perfect_recall


    def verify_tgt_mapped_vp(self):
        print("Verifying tgt_mapped_vp")
        for i in tqdm(range(len(self.tgt))):
            example = self.__getitem__(i)

            original = example['target_old']
            original_words = []
            for i in original:
                original_words.append(self.tgt_dict[i])
            original_words = ' '.join(original_words)
            original_words = original_words.replace('@@ ', '')

            target = example['target']
            target_words = []
            for i in target:
                target_words.append(self.tgt_dict[i])
            target_words = ' '.join(target_words)
            target_words = target_words.replace('@@ ', '')

            tgt_vocab = example['target_vocab']
            target_mapped = example['target_mapped']
            target_mapped_words = []
            for i in target_mapped:
                target_mapped_words.append(self.tgt_dict[tgt_vocab[i]])
            target_mapped_words = ' '.join(target_mapped_words)
            target_mapped_words = target_mapped_words.replace('@@ ', '')

            if target_words != original_words or target_mapped_words != original_words:
                pdb.set_trace()

            assert(target_words == original_words)
            assert(target_mapped_words == original_words)

    def test_get_bpe_target(self):
        print("Testing get_bpe_target tgt_mapped_vp")
        for _ in tqdm(range(1000)):
            i = np.random.randint(len(self.tgt))
            example = self.__getitem__(i)

            original = example['target_old']
            original_words = []
            for i in original:
                original_words.append(self.tgt_dict[i])
            original_words = ' '.join(original_words)
            original_words = original_words.replace('@@ ', '')

            target = example['target']
            tgt_vocab = example['target_vocab']
            prob = 0.8
            mask = np.random.choice(a=[False, True], size=len(target), p=[prob, 1 - prob])
            new_target = self.get_bpe_target(target, mask)
            new_target_mapped = self.get_mapped_target(new_target, tgt_vocab)

            new_target_words = []
            for i in new_target:
                new_target_words.append(self.tgt_dict[i])
            new_target_words = ' '.join(new_target_words)
            new_target_words = new_target_words.replace('@@ ', '')

            new_target_mapped_words = []
            for i in new_target_mapped:
                new_target_mapped_words.append(self.tgt_dict[tgt_vocab[i]])
            new_target_mapped_words = ' '.join(new_target_mapped_words)
            new_target_mapped_words = new_target_mapped_words.replace('@@ ', '')

            if new_target_words != original_words or new_target_mapped_words != original_words:
                y = [(x in tgt_vocab) for x in new_target]
                pdb.set_trace()

            assert (new_target_words == original_words)
            assert (new_target_mapped_words == original_words)


    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )



