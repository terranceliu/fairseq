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
        align_dataset=None, tgt_vocab_size=None, perfect_oracle=False,
        tgt_bow=False, use_roberta=False, ft_roberta=False,
        vocab_task=None,
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

        self.tgt_vocab_size = tgt_vocab_size
        self.perfect_oracle = perfect_oracle
        self.tgt_bow = tgt_bow

        self.tgt_old = self.tgt

        self.tgt_vocab = None
        self.tgt_mapped = None
        self.tgt_vocab_bow = None

        self.mandatory_tokens = torch.tensor(
            [self.tgt_dict.bos(), self.tgt_dict.pad(), self.tgt_dict.eos(), self.tgt_dict.unk()])

        if self.tgt_vocab_size is not None:
            self.top_tokens = torch.arange(self.tgt_vocab_size)
            if self.tgt_vocab_size == len(tgt_dict):
                self.tgt_vocab = torch.arange(self.tgt_vocab_size).repeat(1,len(tgt)).view(len(tgt),-1).long()
                self.tgt_mapped = self.tgt
                return

            self.generate_tgt_vocab()
            self.generate_tgt_mapped()

        self.tgt_vocab_padded = None
        self.tgt_vocab_lengths = None
        self.generate_tgt_vocab_padded()

        self.use_roberta = use_roberta
        self.ft_roberta = ft_roberta
        self.src_roberta_tokens = None
        self.src_roberta_feats = None

        if self.use_roberta:
            from fairseq.models.roberta import RobertaModel
            self.roberta = RobertaModel.from_pretrained('checkpoints/pretrained/roberta.base', checkpoint_file='model.pt')
            self.roberta = self.roberta.cuda()
            self.roberta.eval()

            # self.generate_roberta_tokens()
            # self.generate_roberta_features()

            if self.ft_roberta:
                self.generate_roberta_tokens()
            else:
                self.generate_roberta_features()
            del self.roberta


        # self.num_extra_bpe = 1
        # self.extra_tokens = []
        # self.token2idx = {}
        # if self.num_extra_bpe > 0:
        #     for i in range(len(self.tgt_dict)):
        #         token = self.tgt_dict[i]
        #         token_trunc = token.replace('@', '')
        #         if len(token_trunc) <= self.num_extra_bpe:
        #             self.extra_tokens.append(i)
        #             self.token2idx[token] = i
        #     self.extra_tokens = torch.tensor(self.extra_tokens)

        # # for WMT16
        # if self.split == 'train':
        #     return

        # self.vocab_task = vocab_task
        # self.top_logits = None
        #
        # self.generate_top_logits()
        # self.generate_tgt_vocab_vp()
        # self.generate_tgt_mapped_vp()
        #
        # if ((self.tgt_vocab < 4).sum(dim=1) != 4).sum() != 0:
        #     print("tgt_vocab missing mandatory tokens")
        #     # pdb.set_trace()


    def __getitem__(self, index):
        example = super().__getitem__(index)
        example['target_old'] = self.tgt_old[index]
        if self.tgt_vocab_size is not None:
            example['target_vocab'] = self.tgt_vocab[index]
            example['target_mapped'] = self.tgt_mapped[index]
        if self.tgt_bow:
            tgt_vocab_len = self.tgt_vocab_lengths[index]
            example['target_vocab_nopad'] = self.tgt_vocab_padded[index, :tgt_vocab_len]
        if self.use_roberta:
            if self.src_roberta_tokens is not None:
                example['source_roberta'] = self.src_roberta_tokens[index]
            else:
                example['source_roberta'] = self.src_roberta_feats[index]
        return example

    def generate_roberta_tokens(self):
        filepath = os.path.join(self.preprocessed_path, "{}.roberta_tokens.pt".format(self.split))
        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("Found preprocessed filed for src_roberta_tokens!")
            self.src_roberta_tokens = torch.load(filepath)
        else:
            print("Generating src_roberta_tokens...")
            excluded_tokens = [self.src_dict.bos(), self.src_dict.eos(), self.src_dict.pad()]
            self.src_roberta_tokens = []
            for idx, src in enumerate(tqdm(self.src)):
                sent = []
                for token in src:
                    if token in excluded_tokens:
                        continue
                    sent.append(self.src_dict[token])
                sent = " ".join(sent)
                tokens = self.roberta.encode(sent)
                self.src_roberta_tokens.append(tokens)

            torch.save(self.src_roberta_tokens, filepath)

    def generate_roberta_features(self):
        filepath = os.path.join(self.preprocessed_path, "{}.roberta_feats.pt".format(self.split))
        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("Found preprocessed filed for src_roberta_feats!")
            self.src_roberta_feats = torch.load(filepath)
        else:
            print("Generating src_roberta_feats...")
            if self.src_roberta_tokens is None:
                self.generate_roberta_tokens()

            num_ex = len(self.src)
            self.src_roberta_feats = torch.zeros(num_ex, 768)
            token_lengths = torch.tensor([len(x) for x in self.src_roberta_tokens])

            max_tokens = 5000
            start = 0
            time_start = time.time()
            while start < num_ex:
                end = start + 1
                while end + 1 <= num_ex and (end + 1 - start) * token_lengths[start:end + 1].max() < max_tokens:
                    end += 1

                tokens = self.src_roberta_tokens[start:end]
                lengths = token_lengths[start:end]
                max_length = lengths.max()

                x = torch.ones((len(tokens), max_length)).long()
                for idx, token in enumerate(tokens):
                    x[idx, :len(token)] = token
                x = x[:, :512] # roberta doesn't allow sentence lengths > 512
                mask = (x == 1) # removing padding

                x = x.cuda()
                x = self.roberta.extract_features(x).detach()
                x = x.cpu().numpy()
                x[mask] = np.nan
                x = np.nanmean(x, axis=1)

                # make sure we don't have nans for some reason
                if np.abs(x.mean()) < 0:
                    print("error: nan in features")
                    pdb.set_trace()

                self.src_roberta_feats[start:end] = torch.tensor(x)

                percent_progress = end / num_ex
                time_elapsed = time.time() - time_start
                time_elapsed = timedelta(seconds=time_elapsed).total_seconds()
                time_remaining = time_elapsed / percent_progress - time_elapsed
                print("Progress: {:.2f}%, batch_size: {}, elapsed {:.2f}s, remaining {:.2f}s".format(
                     percent_progress * 100, end - start, time_elapsed, time_remaining))

                start = end

            self.src_roberta_tokens = None

            torch.save(self.src_roberta_feats, filepath)


    def generate_top_logits(self):
        filepath = os.path.join(self.preprocessed_path, "{}.vp_logits.pt".format(self.split))
        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("Found preprocessed file for top_logits (vp)!")
            self.top_logits = torch.load(filepath)
        else:
            print("Generating top_logits...")

            # path = 'checkpoints/iswlt14_de-en/vp_mlp_binary/checkpoint_best.pt'
            path = 'checkpoints/wmt16.en-de.joined-dict.transformer_ott2018/vp_mlp_losstest/checkpoint_best.pt'
            state = torch.load(path)
            args = state['args']
            model = self.vocab_task.build_model(args).cuda()
            model.load_state_dict(state['model'], strict=True)

            self.top_logits = []
            for src_tokens in tqdm(self.src):
                net_output = model(src_tokens.unsqueeze(0), None, None)
                self.top_logits.append(model.get_logits(net_output).float().detach())
            self.top_logits = torch.cat(self.top_logits)

            torch.save(self.top_logits, filepath)
            del model


    def generate_tgt_vocab_vp(self):
        filepath = os.path.join(self.preprocessed_path, "{}.tgt_vocab_vp_{}_extra{}.pt".format(
            self.split, self.tgt_vocab_size, self.num_extra_bpe))

        # if self.split != 'train':
        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("Found preprocessed file for tgt_vocab (vp)!")
            self.tgt_vocab = torch.load(filepath)
        else:
            print("Generating tgt_vocab...")
            self.tgt_vocab = self.top_logits.sort(descending=True)[1][:, :self.tgt_vocab_size]

            if len(self.extra_tokens) > 0:
                for i in tqdm(range(len(self.tgt))):
                    ix = self.tgt_vocab[i].view(1, -1).eq(self.extra_tokens.view(-1, 1)).sum(0) == 0
                    extra_tokens = self.tgt_vocab[i][ix]
                    vocab_tokens = torch.cat((self.extra_tokens, extra_tokens))[:self.tgt_vocab_size]

                    self.tgt_vocab[i] = vocab_tokens

            self.tgt_vocab, _ = self.tgt_vocab.sort(dim=1)

            torch.save(self.tgt_vocab, filepath)


    def generate_tgt_mapped_vp(self):
        filepath = os.path.join(self.preprocessed_path, "{}.tgt_mapped_vp_{}_extra{}.pt".format(
            self.split, self.tgt_vocab_size, self.num_extra_bpe))

        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("Found preprocessed file for tgt_mapped (vp)!")
            # self.tgt_mapped, self.tgt, self.tgt_sizes = torch.load(filepath)

            # Debugging
            old_target = self.tgt
            old_target_sizes = self.tgt_sizes
            self.tgt_mapped, self.tgt, self.tgt_sizes = torch.load(filepath)

            new_target = self.tgt
            new_target_sizes = self.tgt_sizes

            # count_problem = 0
            # for i in range(len(old_target)):
            #     old_sentence = ''
            #     for char in old_target[i]:
            #         old_sentence = old_sentence + self.tgt_dict[char] + ' '
            #     old_sentence = old_sentence.replace('@@ ', '')
            #
            #
            #     new_sentence = ''
            #     for char in new_target[i]:
            #         new_sentence = new_sentence + self.tgt_dict[char] + ' '
            #     new_sentence = new_sentence.replace('@@ ', '')
            #
            #     if old_sentence != new_sentence:
            #         count_problem += 1
            #
            # print(count_problem)
            # pdb.set_trace()
            #
            # num_correct = 0.0
            # num_total = 0.0
            # for i in range(len(self.tgt_mapped)):
            #     tgt_vocab = self.tgt_vocab[i]
            #     tgt = self.tgt[i]
            #
            #     for token in tgt:
            #         num_total += 1
            #
            #         if token in tgt_vocab:
            #             num_correct += 1
            #
            # print(num_correct / num_total)
            # pdb.set_trace()

            # pdb.set_trace()
            # print((new_target_sizes != old_target_sizes).sum())
            # print((new_target_sizes - old_target_sizes).mean())

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
                    missing_tokens = []
                    for i in target:
                        if i not in tgt_vocab:
                            missing_tokens.append(i.item())
                    missing_tokens = torch.tensor(missing_tokens).unique().tolist()

                    missing2replacement = {}
                    for i in missing_tokens:
                        replacement = []
                        token = self.tgt_dict[i].replace('@@', '')
                        for idx, char in enumerate(token):
                            if idx == len(token) - 1 and self.tgt_dict[i][-1] != '@':
                                lookup = char
                            else:
                                lookup = char + '@@'
                            if lookup not in self.token2idx.keys():
                                print(lookup)
                                continue
                            replacement.append(self.token2idx[lookup])
                        missing2replacement[i] = replacement

                    missing2replacement = {}
                    for i in missing_tokens:
                        replacement = []
                        token = self.tgt_dict[i].replace('@@', '')

                        idx = 0
                        while idx < len(token):
                            try:
                                char = token[idx:idx + 2]
                                if char not in self.token2idx.keys() or char + '@@' not in self.token2idx.keys():
                                    char = token[idx]
                                    idx += 1
                                else:
                                    idx += 2
                            except:
                                char = token[idx]
                                idx += 1

                            if idx >= len(token) and self.tgt_dict[i][-1] != '@':
                                lookup = char
                            else:
                                lookup = char + '@@'
                            if lookup not in self.token2idx.keys():
                                print(lookup)
                                continue
                            replacement.append(self.token2idx[lookup])
                        missing2replacement[i] = replacement

                    for i in missing2replacement.keys():
                        replace_idx = np.where(target == i)[0]
                        for multiplier, idx in enumerate(replace_idx):
                            idx = idx + (len(missing2replacement[i]) - 1) * multiplier
                            target = np.insert(target, idx + 1, missing2replacement[i])
                            target = np.delete(target, idx)

                new_tgt.append(target)
                new_tgt_sizes.append(len(target))

                # map from old to new
                index = np.argsort(tgt_vocab)
                sorted_x = tgt_vocab[index]
                sorted_index = np.searchsorted(sorted_x, target)

                yindex = np.take(index, sorted_index, mode="clip")
                mask = tgt_vocab[yindex] != target

                result = np.ma.array(yindex, mask=mask).data

                tgt_mapped.append(torch.tensor(result))

            self.tgt_mapped = tgt_mapped
            self.tgt = new_tgt
            self.tgt_sizes = np.array(new_tgt_sizes)

            torch.save((self.tgt_mapped, self.tgt, self.tgt_sizes), filepath)


    def generate_tgt_vocab(self):
        filepath = os.path.join(self.preprocessed_path, "{}.tgt_vocab_{}_oracle{}.pt".format(
            self.split, self.tgt_vocab_size, self.perfect_oracle))

        if os.path.exists(filepath) and os.path.isfile(filepath):
            print("Found preprocessed file for tgt_vocab!")
            self.tgt_vocab = torch.load(filepath)
        else:
            print("Generating tgt_vocab...")
            self.tgt_vocab = torch.ones((len(self.tgt), self.tgt_vocab_size)).long()
            for i in range(len(self.tgt)):
                tgt_tokens = self.tgt[i]
                tgt_tokens = torch.cat((tgt_tokens, self.mandatory_tokens))
                tgt_tokens = torch.unique(tgt_tokens)
                assert (tgt_tokens.shape[0] <= self.tgt_vocab_size)

                if self.perfect_oracle:
                    self.tgt_vocab[i, :tgt_tokens.shape[0]] = tgt_tokens
                else:
                    ix = self.top_tokens.view(1, -1).eq(tgt_tokens.view(-1, 1)).sum(0) == 0
                    extra_tokens = self.top_tokens[ix]
                    vocab_tokens = torch.cat((tgt_tokens, extra_tokens))[:self.tgt_vocab_size]
                    vocab_tokens, _ = torch.sort(vocab_tokens)

                    self.tgt_vocab[i] = vocab_tokens

            torch.save(self.tgt_vocab, filepath)


    def generate_tgt_mapped(self):
        filepath = os.path.join(self.preprocessed_path, "{}.tgt_mapped_{}_oracle{}.pt".format(
            self.split, self.tgt_vocab_size, self.perfect_oracle))

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


    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )



