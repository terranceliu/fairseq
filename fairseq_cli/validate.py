#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import random

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter

import pdb

def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        # arg_overrides=eval(args.model_overrides),
        task=task,
    )
    model = models[0].cuda()
    model.eval()

    valid_subsets = args.valid_subset.split(',')

    # valid_logits, valid_targets, src_lengths = validate(args, task, model, valid_subsets, top_k=50000)
    valid_logits, valid_targets, src_lengths = validate(args, task, model, valid_subsets, top_k=None)

    top_k_dict = get_min_k_precall(valid_logits, valid_targets, src_lengths)

    pdb.set_trace()

    # for k in [1000, 2000, 3000, 4000, 5000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]:
    # for k in [100, 1000, 3000, 5000, 10000, 20000, 30000, 40000, 50000, 75000]:
    for k in [200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]:
        logits = valid_logits[:, :k]
        get_recall(logits, valid_targets)

    # for k in [100, 1000, 3000, 5000, 10000, 20000, 30000]:
    #     valid_logits, valid_targets = validate(args, task, model, epoch_itr, valid_subsets, criterion, top_k=k)
    #     get_recall(valid_logits, valid_targets)

def get_recall(lprobs, target):
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

    return recall, perfect_recall

def get_min_k_precall(valid_logits, valid_targets, src_lengths):
    min_top_k = []
    for i in range(len(valid_logits)):
        logit = valid_logits[i]
        target = valid_targets[i]
        top_k = []
        for token in target:
            index = (logit == token).nonzero()
            top_k.append(index)
        top_k = torch.tensor(top_k)
        min_top_k.append(top_k.max() + 1)
    min_top_k = torch.tensor(min_top_k)

    top_k_dict = {}
    for i in range(src_lengths.max()):
        top_k_dict[i + 1] = []
    for i in range(len(src_lengths)):
        length = src_lengths[i].item()
        k = min_top_k[i].item()
        top_k_dict[length].append(k)

    for k in top_k_dict.keys():
        x = np.array(top_k_dict[k])
        if len(x) == 0:
            continue
        mean = x.mean()
        median = np.median(x)
        max = x.max()
        min = x.min()
        std = x.std()
        print("k: {}, mean: {:.2f}, median: {}, max: {}, min: {}, std: {:.2f}".format(
            k, mean, median, max, min, std))

    return top_k_dict

def validate(args, task, model, subsets, top_k=None):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                model.max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

        progress = progress_bar.build_progress_bar(
            args, itr, 0,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        num_examples = len(task.dataset(subset))
        if top_k is None:
            top_k = len(task.dataset(subset).tgt_dict)
        valid_logits = torch.zeros((num_examples, top_k), dtype=torch.int32)
        valid_targets = []
        src_lengths = []
        i = 0


        # pdb.set_trace()

        for idx, sample in enumerate(progress):
            batch_size = len(sample['target'])

            if torch.cuda.is_available():
                sample = utils.move_to_cuda(sample)
            net_output = model(**sample['net_input'])
            logits = model.get_logits(net_output).float()
            logits = logits.sort(descending=True)[1]
            logits = logits[:, :top_k]

            valid_logits[i:i + batch_size] = logits.detach().cpu()
            src_lengths.append(sample['net_input']['src_lengths'].cpu())
            valid_targets.append(sample['target_vocab_nopad'].cpu().int())

            i += batch_size

        src_lengths = torch.cat(src_lengths)

        temp = []
        for x in valid_targets:
            for target_vocab in x:
                temp.append(torch.unique(target_vocab))
        valid_targets = temp

    return valid_logits, valid_targets, src_lengths


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    options.add_common_eval_args(parser)
    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)

if __name__ == '__main__':
    cli_main()
