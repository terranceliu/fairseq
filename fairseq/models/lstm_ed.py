# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax
from .lstm import LSTMModel, LSTMEncoder, LSTMDecoder, Embedding

import numpy as np
import pdb

@register_model('lstm_ed')
class LSTMEDModel(LSTMModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-freeze-embed', action='store_true',
                            help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-freeze-embed', action='store_true',
                            help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )
        decoder = LSTMEDDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )
        return cls(encoder, decoder)

class LSTMEDDecoder(LSTMDecoder):
    """LSTM decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
    ):
        super().__init__(dictionary, embed_dim=embed_dim, hidden_size=hidden_size, out_embed_dim=out_embed_dim,
                         num_layers=num_layers, dropout_in=dropout_in, dropout_out=dropout_out, attention=attention,
                         encoder_output_units=encoder_output_units, pretrained_embed=pretrained_embed,
                         share_input_output_embed=share_input_output_embed, adaptive_softmax_cutoff=adaptive_softmax_cutoff)

        self.efficient_decoding = True
        self.oracle = False
        self.tgt_vocab_size = 1000
        self.use_dot = False

        self.num_embeddings = len(dictionary)
        self.embed_dim = embed_dim
        self.top_tokens = torch.arange(self.tgt_vocab_size).cuda()

        # if self.efficient_decoding and self.use_dot:
        #     self.fc_out = Linear(out_embed_dim, embed_dim, dropout=dropout_out)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, target=None):
        x, attn_scores = super().forward(prev_output_tokens, encoder_out, incremental_state=incremental_state, target=None)

        if self.efficient_decoding:
            tgt_vocab = []
            for i in range(target.shape[0]):
                tgt_tokens = torch.unique(target[i])
                if self.oracle:
                    vocab_tokens = torch.ones(self.tgt_vocab_size, dtype=torch.long).cuda()
                    vocab_tokens[:tgt_tokens.shape[0]] = tgt_tokens
                elif tgt_tokens.shape[0] >= self.tgt_vocab_size:
                    vocab_tokens = tgt_tokens[:self.tgt_vocab_size]
                else:
                    # get tokens from top_tokens not in tgt_tokens
                    ix = self.top_tokens .view(1, -1).eq(tgt_tokens.view(-1, 1)).sum(0) == 0
                    extra_tokens = self.top_tokens[ix]
                    vocab_tokens = torch.cat((tgt_tokens, extra_tokens))[:self.tgt_vocab_size]

                tgt_vocab.append(vocab_tokens)
                if len(vocab_tokens) != self.tgt_vocab_size:
                    pdb.set_trace()

            tgt_vocab = torch.stack(tgt_vocab)

            if self.use_dot:
                tgt_vocab_embeddings = self.embed_tokens(tgt_vocab)
                tgt_vocab_embeddings = tgt_vocab_embeddings.view(len(tgt_vocab_embeddings), self.embed_dim, self.tgt_vocab_size)

                x = torch.bmm(x, tgt_vocab_embeddings)
                x_final = torch.zeros((x.shape[0], x.shape[1], self.num_embeddings)).cuda()
                x_final[:] = -float('inf')

                helper_ix = np.array([np.arange(tgt_vocab.shape[0]), ] * tgt_vocab.shape[1]).transpose()
                helper_ix = torch.from_numpy(helper_ix).cuda()

                x_final[helper_ix, :, tgt_vocab] = x.view(x.shape[0], x.shape[2], x.shape[1])
            else:
                x_final = torch.zeros(x.shape).cuda()
                x_final[:] = -float('inf')

                helper_ix = torch.arange(tgt_vocab.shape[0]).unsqueeze(0).T.expand(tgt_vocab.shape)
                # helper_ix = np.array([np.arange(tgt_vocab.shape[0]), ] * tgt_vocab.shape[1]).transpose()
                # helper_ix = torch.from_numpy(helper_ix).cuda()

                x_final[helper_ix, :, tgt_vocab] = x[helper_ix, :, tgt_vocab]

            x = x_final

        return x, attn_scores

@register_model_architecture('lstm_ed', 'lstm_ed')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')


@register_model_architecture('lstm_ed', 'lstm_ed_wiseman_iwslt_de_en')
def lstm_wiseman_iwslt_de_en(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    base_architecture(args)


@register_model_architecture('lstm_ed', 'lstm_ed_luong_wmt_en_de')
def lstm_luong_wmt_en_de(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1000)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1000)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 1000)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0)
    base_architecture(args)
