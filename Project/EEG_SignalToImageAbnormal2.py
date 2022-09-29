
##import mne
import numpy as np
from numpy import newaxis
import pyedflib
##import torch
##import torch.nn as nn
##import torch.nn.functional as F
##from torch.autograd import Variable
##
##from IPython.display import Image
##from IPython.core.display import HTML
##import matplotlib.pyplot as plt
import os
##from sklearn.decomposition import PCA
##import pandas as pd
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
##import librosa, librosa.display
##
##
##import matplotlib.pyplot as plt
##from mpl_toolkits.axes_grid1 import ImageGrid
##from pyts.image import GramianAngularField
##from pyts.datasets import load_gunpoint
##from pyts.image import RecurrencePlot

import matlab.engine
print('Matlab engine starting')
eng = matlab.engine.start_matlab()
print('Matlab engine started')
# # Entire Directory

# In[2]:


def signalToImage(sigbufs,i):
    
    #spectograms
    pname = 'spectograms'

    fname = dname+'/'+pname+'/'+stype+'/'+pname+str(i)+'.png'

    plot.figure(figsize=(4, 4))
    plot.specgram(sigbufs,Fs=256)
    plot.tick_params(axis='both', labelsize=0, length = 0)
    plt.tight_layout()
    plot.savefig(fname)
    plot.close('all')

def signalToKurto(sigbufs,i):

    #Kurtograms
    pname = 'kurtograms'
    print('Starting kurtogram')
    sigbuf = sigbufs.flatten('F')
    #print(str(k)+' channel')
    fname = dname+'/'+pname+'/'+stype
    hb = pname+str(i)
    x = matlab.double(list(sigbuf))
    c = eng.Fast_kurtogram(x,nlevel,Fs,fname,hb,nargout = 1)
        
##i, j = 0, 0
##edfs, tses = [], []
nlevel = 6
Fs = 256
##stype = 'eval/1'
##dname = './edf'
##
##directory = './edf/eval/normal/'
##
##print('Starting conversion of eval normal')
##for root, dirs, files in os.walk(directory, topdown=False):
##    i+=1
##    
##    for name in files:
##        if name[-3:]=='edf':
##            j+=1
##            edfs.append(os.path.join(root, name))
##            print(j,os.path.join(root, name))
##            filename = os.path.join(root, name)
##            f = pyedflib.EdfReader(filename)
##            n = f.signals_in_file
##            signal_labels = f.getSignalLabels()
##            dict(zip(np.arange(0, len(signal_labels)), signal_labels))
##            sigbufs = np.zeros((n-3, f.getNSamples()[0]))
##            for k in np.arange(n-3):
##              sigbufs[k, :] = f.readSignal(k)
##            signalToKurto(sigbufs,j)
##            
##print("end")


# In[ ]:



# Change the following parameters
# 1 directory
# 2 np.ones and np.zeros
# 3 Path of the numpy files
##i, j = 0, 0
##edfs, tses = [], []
##
##stype = 'eval/0'
##dname = './edf'
##
##directory = './edf/eval/abnormal/'
##
##print('Starting conversion of eval abnormal')
##for root, dirs, files in os.walk(directory, topdown=False):
##    i+=1
##    for name in files:
##        if name[-3:]=='edf':
##            j+=1
##            edfs.append(os.path.join(root, name))
##            print(j,os.path.join(root, name))
##            filename = os.path.join(root, name)
##            f = pyedflib.EdfReader(filename)
##            n = f.signals_in_file
##            signal_labels = f.getSignalLabels()
##            dict(zip(np.arange(0, len(signal_labels)), signal_labels))
##            sigbufs = np.zeros((n-3, f.getNSamples()[0]))
##            for k in np.arange(n-3):
##              sigbufs[k, :] = f.readSignal(k)
##            signalToKurto(sigbufs,j)
##            
##print("final")


# In[ ]:



# Change the following parameters
# 1 directory
# 2 np.ones and np.zeros
# 3 Path of the numpy files
##i, j = 0, 0
##edfs, tses = [], []
##
##stype = 'train/1'
##dname = './edf'
##
##directory = './edf/train/normal/'
##
##print('Starting conversion of train normal')
##for root, dirs, files in os.walk(directory, topdown=False):
##    i+=1
##    for name in files:
##        if name[-3:]=='edf':
##            j+=1
##            edfs.append(os.path.join(root, name))
##            print(j,os.path.join(root, name))
##            if j>747:
##                filename = os.path.join(root, name)
##                f = pyedflib.EdfReader(filename)
##                n = f.signals_in_file
##                signal_labels = f.getSignalLabels()
##                dict(zip(np.arange(0, len(signal_labels)), signal_labels))
##                sigbufs = np.zeros((n-3, f.getNSamples()[0]))
##                for k in np.arange(n-3):
##                  sigbufs[k, :] = f.readSignal(k)
##                signalToKurto(sigbufs,j)
##            
##print("final")

i, j = 0, 0
edfs, tses = [], []

stype = 'train/0'
dname = './edf'

directory = './edf/train/abnormal/'

print('Starting conversion of train abnormal')
for root, dirs, files in os.walk(directory, topdown=False):
    i+=1
    for name in files:
        if name[-3:]=='edf':
            j+=1
            try:
                edfs.append(os.path.join(root, name))
                print(j,os.path.join(root, name))
                filename = os.path.join(root, name)
                f = pyedflib.EdfReader(filename)
                n = f.signals_in_file
                signal_labels = f.getSignalLabels()
                dict(zip(np.arange(0, len(signal_labels)), signal_labels))
                sigbufs = np.zeros((n-3, f.getNSamples()[0]))
                for k in np.arange(n-3):
                  sigbufs[k, :] = f.readSignal(k)
                if j<=9:
                    signalToKurto(sigbufs,j)
                else :
                    break
            except:
                print(j,os.path.join(root, name))
            
print("final")

