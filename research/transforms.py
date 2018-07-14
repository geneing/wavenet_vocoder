#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
from __future__ import division
import numpy as np
import scipy as sp
import scipy

import scipy.io.wavfile as wav
import nltk
import pyworld as pw
from pylab import *

import simpleaudio as sa


#%%
from scipy.signal import stft, istft
import lws

(rate,sig) = wav.read("/home/eugening/Neural/MachineLearning/Speech/TrainingData/LJSpeech-1.0/wavs/LJ001-0001.wav")
noverlap = 768
fftsize = 1024
shift_len = nperseg-noverlap
n=(sig.shape[0]//shift_len)*shift_len
sig=sig[:n]

(f,t,Zxx)=stft(sig, fs=rate, nperseg=fftsize, noverlap=noverlap)
(t1,x1)=istft(Zxx, fs=rate, nperseg=fftsize, noverlap=noverlap)

lws_proc=lws.lws(fftsize, shift_len, mode='speech')

a = lws_proc.stft(sig)
a0 = np.abs(a)
stft0 = lws_proc.run_lws(a0)
sig0 = lws_proc.istft(stft0)


