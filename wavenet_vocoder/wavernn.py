# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from .modules import Embedding
from .mixture import sample_from_discretized_mix_logistic
import audio

def _expand_global_features(B, T, g, bct=True):
    """Expand global conditioning features to all time steps

    Args:
        B (int): Batch size.
        T (int): Time length.
        g (Variable): Global features, (B x C) or (B x C x 1).
        bct (bool) : returns (B x C x T) if True, otherwise (B x T x C)

    Returns:
        Variable: B x C x T or B x T x C or None
    """
    if g is None:
        return None
    g = g.unsqueeze(-1) if g.dim() == 2 else g
    if bct:
        g_bct = g.expand(B, -1, T)
        return g_bct.contiguous()
    else:
        g_btc = g.expand(B, -1, T).transpose(1, 2)
        return g_btc.contiguous()


class WaveRNN(nn.Module):
    """The WaveRNN model that supports local and global conditioning.

    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized
          one-hot vecror. this must equal to the quantize channels. Other wise
          num_mixtures x 3 (pi, mu, log_scale).
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        n_speakers (int): Number of speakers. Used only if global conditioning
          is enabled.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise
          quantized one-hot vector is expected.
    """

    def __init__(self, in_channels=256, out_channels=256,
                 gru_hidden_size=896,
                 cin_channels=-1, gin_channels=-1, n_speakers=None,
                 dropout=False, upsample_conditional_features=False,
                 scalar_input=False, hop_size=1
                 ):
        super(WaveRNN, self).__init__()
        self.scalar_input = scalar_input
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.hidden_size = gru_hidden_size
        self.hop_size = hop_size
        self.receptive_field = 0 #doesn't make sense for wavernn
        self.upsample_conditional_features = upsample_conditional_features

        assert (not scalar_input)

        cond_channels = cin_channels if cin_channels > 0 else 0
        cond_channels += gin_channels if gin_channels > 0 else 0

        self.layers = nn.ModuleList()
        rnn = nn.GRU(input_size=in_channels+cond_channels, hidden_size=gru_hidden_size, bias=True, batch_first=False)
        self.layers.append(rnn)

        linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.layers.append(nn.ReLU(linear1))

        linear2 = nn.Linear(self.hidden_size, self.out_channels)
        self.layers.append(linear2)

        if gin_channels > 0:
            assert n_speakers is not None
            self.embed_speakers = Embedding(
                n_speakers, gin_channels, padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None


    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.cin_channels > 0

    def forward(self, x, c=None, g=None, softmax=False):
        """Forward step

        Args:
            x (Variable): One-hot encoded audio signal, shape (B x C x T)
            c (Variable): Local conditioning features,
              shape (B x cin_channels x T)
            g (Variable): Global conditioning features,
              shape (B x gin_channels x 1) or speaker Ids of shape (B x 1).
              Note that ``self.use_speaker_embedding`` must be False when you
              want to disable embedding layer and use external features
              directly (e.g., one-hot vector).
              Also type of input tensor must be FloatTensor, not LongTensor
              in case of ``self.use_speaker_embedding`` equals False.
            softmax (bool): Whether applies softmax or not.

        Returns:
            Variable: output, shape B x out_channels x T
        """
        B, n_outchannels, T = x.size()
        output = torch.zeros_like(x)
        #hidden = torch.zeros((1, B, self.hidden_size), dtype=x.dtype, device=x.device)

        if g is not None:
            if self.embed_speakers is not None:
                # (B x 1) -> (B x 1 x gin_channels)
                g = self.embed_speakers(g.view(B, -1))
                # (B x gin_channels x 1)
                g = g.transpose(1, 2)
                assert g.dim() == 3

                # Expand global conditioning features to all time steps
                g_bct = _expand_global_features(B, T, g, bct=True)
                x = torch.cat((x,g_bct),1)

        # Local Conditioning
        if c is not None:
            if c.size(-1) != x.size(-1):
                # B x C x T
                c_bct = torch.zeros((c.size(0), c.size(1), x.size(2)), dtype=x.dtype, requires_grad=False, device=x.device)

                for i in range(T):
                    # upsampling through linear interpolation
                    try:
                        c_bct[:,:,i] = torch.lerp(c[:,:,i//self.hop_size-1], c[:,:,i//self.hop_size], 1.-((i/self.hop_size) % 1))
                    except IndexError:
                        c_bct[:,:,i] = c[:, :, i//self.hop_size]*((i/self.hop_size) % 1)

                assert c_bct.size(-1) == x.size(-1)
                x = torch.cat((x, c_bct), 1)
            else:
                c_bct = c

        # Feed data to network
        #for t in range(T):
        out, _ = self.layers[0](x.permute(2,0,1))
        lin1 = self.layers[1](out)
        lin2 = self.layers[2](lin1)
        output = F.softmax(lin2, dim=2) if softmax else lin2

        return output.permute(1,2,0)

    def incremental_forward(self, initial_input=None, c=None, g=None,
                            T=100, test_inputs=None,
                            tqdm=lambda x: x, softmax=True, quantize=True,
                            log_scale_min=-7.0):
        """Incremental forward step

        Due to linearized convolutions, inputs of shape (B x C x T) are reshaped
        to (B x T x C) internally and fed to the network for each time step.
        Input of each time step will be of shape (B x 1 x C).

        Args:
            initial_input (Variable): Initial decoder input, (B x C x 1)
            c (Variable): Local conditioning features, shape (B x C' x T)
            g (Variable): Global conditioning features, shape (B x C'' or B x C''x 1)
            T (int): Number of time steps to generate.
            test_inputs (Variable): Teacher forcing inputs (for debugging)
            tqdm (lamda) : tqdm
            softmax (bool) : Whether applies softmax or not
            quantize (bool): Whether quantize softmax output before feeding the
              network output to input for the next time step. TODO: rename
            log_scale_min (float):  Log scale minimum value.

        Returns:
            Variable: Generated one-hot encoded samples. B x C x T　
              or scalar vector B x 1 x T
        """
        self.clear_buffer()
        B = 1
        hidden = torch.zeros((B,self.hidden_size), dtype=initial_input.dtype, device=initial_input.device)

        # shape (B x C x T)
        if test_inputs is not None:
            B = test_inputs.size(0)
            if T is None:
                T = test_inputs.size(2)
            else:
                T = max(T, test_inputs.size(2))
        # cast to int in case of numpy.int64...
        T = int(T)

        # Global conditioning
        if g is not None:
            if self.embed_speakers is not None:
                # (B x 1) -> (B x 1 x gin_channels)
                g = self.embed_speakers(g.view(B, -1))
                # (B x gin_channels x 1)
                g = g.transpose(1, 2)
                assert g.dim() == 3

                # Expand global conditioning features to all time steps
                g_bct = _expand_global_features(B, T, g, bct=True)

        # Local Conditioning
        if c is not None:
            # B x C x T
            c_bct = torch.zeros((c.size(0), c.size(1), T), dtype=c.dtype, requires_grad=False, device=c.device)

            for i in range(T):
                # upsampling through linear interpolation
                try:
                    c_bct[:,:,i] = torch.lerp(c[:,:,i//self.hop_size-1], c[:,:,i//self.hop_size], 1.-((i/self.hop_size) % 1))
                except IndexError:
                    c_bct[:,:,i] = c[:, :, i//self.hop_size]*((i/self.hop_size) % 1)

        outputs = []
        if initial_input is None:
            if self.scalar_input:
                initial_input = Variable(torch.zeros(B, 1, 1))
            else:
                initial_input = Variable(torch.zeros(B, self.out_channels, 1))
                initial_input = audio.dummy_silence().squeeze().unsqueeze(0)  # TODO: is this ok?
            # https://github.com/pytorch/pytorch/issues/584#issuecomment-275169567
            if next(self.parameters()).is_cuda:
                initial_input = initial_input.cuda()

        current_input = initial_input[:, :, 0]
        for t in tqdm(range(T)):
            if test_inputs is not None and t < test_inputs.size(2):
                current_input = test_inputs[:, :, t]
            else:
                if t > 0:
                    current_input = outputs[-1]

            # Conditioning features for single time step
            ct = None if c is None else c_bct[:, :, t]
            gt = None if g is None else g_bct[:, :, t]

            if gt is not None:
                current_input=torch.cat((current_input,gt),1)

            if ct is not None:
                ct = ct.to(current_input.device)
                current_input=torch.cat((current_input, ct), 1)

            hidden = self.layers[0](current_input, hidden)
            lin1 = self.layers[1](hidden)
            x = self.layers[2](lin1)

            # Generate next input by sampling
            if self.scalar_input:
                x = sample_from_discretized_mix_logistic(
                    x.view(B, -1, 1), log_scale_min=log_scale_min)
            else:
                x = F.softmax(x.view(B, -1), dim=1) if softmax else x.view(B, -1)
                if quantize:
                    sample = np.random.choice(
                        np.arange(self.out_channels), p=x.view(-1).data.cpu().numpy())
                    x.zero_()
                    x[:, sample] = 1.0

            outputs += [x]

        # T x B x C
        outputs = torch.stack(outputs)
        # B x C x T
        outputs = outputs.transpose(0, 1).transpose(1, 2).contiguous()

        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        for f in self.layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def make_generation_fast_(self):
        pass