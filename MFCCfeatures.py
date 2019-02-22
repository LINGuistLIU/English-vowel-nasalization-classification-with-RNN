from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os
import numpy as np
import random
from collections import defaultdict


def fileNamelist(dirname):
    """
    NEW: Get the list of file names to be used for experiments
    """
    filenamelist = []
    for item in os.listdir(dirname):
        if '.wav' in item:
            filenamelist.append(dirname+item)
    return filenamelist

def file2category(filelist, nasal_namelist):
    """
    make a dictionary={'NASAL':list-of-nasalized-vowels, 'ORAL':list-of-non-nasalized-vowels}
    """
    cfdict = defaultdict(list)
    nasalnames = set(nasal_namelist)
    for n in filelist:
        if n in nasalnames:
            cfdict['NASAL'].append(n)
        else:
            cfdict['ORAL'].append(n)
    return cfdict

def MFCCfeatures(filename):
    (rate, sig) = wav.read(filename)
    mfcc_feat = mfcc(sig, rate, nfft=512*3)
    d_mfcc_feat = delta(mfcc_feat, 2) #delta mfcc
    fbank_feat = logfbank(sig, rate, nfft=512*3)
    #return fbank_feat
    return mfcc_feat
    #return d_mfcc_feat

def ChangeDim(fbank_feat):
    d3_fbank_feat = fbank_feat[:, np.newaxis, :]
    return d3_fbank_feat

def getF1score(precision, recall):
    return (2*precision*recall)/(precision+recall)

def majorityVote(predlist):
    predcounts = defaultdict(int)
    for item in predlist:
        predcounts[item] += 1
    predcounts_sorted = sorted(list(predcounts.items()), key=lambda x:x[1], reverse=True)
    return predcounts_sorted[0][0]

#######################################

import torch
import torch.autograd as autograd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def preparedata(d3_fbank_feat):
    tensor = torch.FloatTensor(d3_fbank_feat)
    return tensor.to(device)

def sound2tensor(soundfn):
    fbank_feat = MFCCfeatures(soundfn)
    d3_fbank_feat = ChangeDim(fbank_feat)
    soundtensor = preparedata(d3_fbank_feat)
    return soundtensor

