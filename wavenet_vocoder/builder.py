# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

from .util import is_scalar_input
#from .. import hparams


def wavenet( hparams ):
    from wavenet_vocoder import WaveNet

    model = WaveNet(out_channels=hparams.out_channels,
                    layers=hparams.layers,
                    stacks=hparams.stacks,
                    residual_channels=hparams.residual_channels,
                    gate_channels=hparams.gate_channels,
                    skip_out_channels=hparams.skip_out_channels,
                    cin_channels=hparams.cin_channels,
                    gin_channels=hparams.gin_channels,
                    weight_normalization=hparams.weight_normalization,
                    n_speakers=hparams.n_speakers,
                    dropout=hparams.dropout,
                    kernel_size=hparams.kernel_size,
                    upsample_conditional_features=hparams.upsample_conditional_features,
                    upsample_scales=hparams.upsample_scales,
                    freq_axis_kernel_size=hparams.freq_axis_kernel_size,
                    scalar_input=is_scalar_input(hparams.input_type)
                    )

    return model


def wavernn( hparams ):
    from wavenet_vocoder import WaveRNN
    model = WaveRNN(out_channels=hparams.out_channels,
                    gru_hidden_size=hparams.gru_hidden_size,
                    cin_channels=hparams.cin_channels,
                    gin_channels=hparams.gin_channels,
                    weight_normalization=hparams.weight_normalization,
                    n_speakers=hparams.n_speakers,
                    dropout=hparams.dropout,
                    upsample_conditional_features=hparams.upsample_conditional_features,
                    upsample_scales=hparams.upsample_scales,
                    freq_axis_kernel_size=hparams.freq_axis_kernel_size,
                    scalar_input=is_scalar_input(hparams.input_type)
                    )

    return model


def fftnet( hparams ):
    from wavenet_vocoder import FFTNet
    model = FFTNet(in_channels=hparams.quantize_channels , out_channels=hparams.out_channels,
                   layers=hparams.layers,
                   cin_channels=hparams.cin_channels,
                   gin_channels=hparams.gin_channels,
                   n_speakers=hparams.n_speakers,
                   dropout=hparams.dropout,
                   upsample_conditional_features=hparams.upsample_conditional_features,
                   scalar_input=is_scalar_input(hparams.input_type), hop_size=hparams.hop_size)

    return model

