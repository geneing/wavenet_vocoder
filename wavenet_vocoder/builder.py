# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

from .util import is_scalar_input
#from .. import hparams

def wavenet(out_channels=256,
            layers=20,
            stacks=2,
            residual_channels=512,
            gate_channels=512,
            skip_out_channels=512,
            cin_channels=-1,
            gin_channels=-1,
            weight_normalization=True,
            dropout=1 - 0.95,
            kernel_size=3,
            n_speakers=None,
            upsample_conditional_features=False,
            upsample_scales=[16, 16],
            freq_axis_kernel_size=3,
            scalar_input=False,
            use_speaker_embedding=True,
            legacy=True,
            ):
    from wavenet_vocoder import WaveNet

    model = WaveNet(out_channels=out_channels, layers=layers, stacks=stacks,
                    residual_channels=residual_channels,
                    gate_channels=gate_channels,
                    skip_out_channels=skip_out_channels,
                    kernel_size=kernel_size, dropout=dropout,
                    weight_normalization=weight_normalization,
                    cin_channels=cin_channels, gin_channels=gin_channels,
                    n_speakers=n_speakers,
                    upsample_conditional_features=upsample_conditional_features,
                    upsample_scales=upsample_scales,
                    freq_axis_kernel_size=freq_axis_kernel_size,
                    scalar_input=scalar_input,
                    use_speaker_embedding=use_speaker_embedding,
                    legacy=legacy,
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

