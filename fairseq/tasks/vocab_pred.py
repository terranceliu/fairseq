# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os

import torch
from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDatasetED,
    PrependTokenDataset,
)

from . import register_task
from .translation_ed import TranslationEDTask

import pdb

@register_task('vocab_pred')
class VocabPredTask(TranslationEDTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    def valid_step(self, sample, model, criterion, print_recall=False, get_logits=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, print_recall=print_recall)
        return loss, sample_size, logging_output

