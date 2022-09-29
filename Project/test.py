import os
import numpy as np
import pyedflib
from scipy.io import savemat

def saveM(sigbufs,j):
    sigbuf = sigbufs.flatten('F')
    dic = {"a":sigbuf}
    print('data'+str(j)+'.mat')
    filename = dname+'/'+stype+'/'+'data'+str(j)+'.mat'
    savemat(filename,dic)



print("Starting")

# I took only the following signal labels for my sigbufs and this increased my accuracy from 50 to 60
main_sig = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']

i, j = 0, 0
edfs, tses = [], []
nlevel = 6
Fs = 256
stype = 'eval/1'
dname = './edf2'

directory = './edf/eval/normal/'

print('Starting conversion of eval normal')
for root, dirs, files in os.walk(directory, topdown=False):
    i+=1
    
    for name in files:
        if name[-3:]=='edf':
            j+=1
            edfs.append(os.path.join(root, name))
            print(j,os.path.join(root, name))
            filename = os.path.join(root, name)
            f = pyedflib.EdfReader(filename)
            n = f.signals_in_file
            signal_labels = f.getSignalLabels()
            dict(zip(np.arange(0, len(signal_labels)), signal_labels))
            # There are only 21 signals in main_sig
            sigbufs = np.zeros((21, f.getNSamples()[0]))

            #In this part I took only those signals that are present in main_sig
            cnt = 0
            for k in np.arange(n):
                if signal_labels[k] in main_sig:
                    sigbufs[cnt, :] = f.readSignal(k)
                    cnt+=1

            # nt is the column size that I am going to split
            nt = sigbufs.shape[1]

            #In this part, using numpy array slicing, I took all the rows but first 0 to nt//2 columns and converted them
            saveM(sigbufs[:,:nt//2],j)

            #In this part, using numpy array slicing, I took all the rows but nt//2 to rest of the columns and converted them
            j+=1
            saveM(sigbufs[:,nt//2:],j)
            
print("end")



#### For the rest of the code, it is same.




###In[ ]:
##
##
##
## Change the following parameters
## 1 directory
## 2 np.ones and np.zeros
## 3 Path of the numpy files
i, j = 0, 0
edfs, tses = [], []

stype = 'eval/0'
dname = './edf2'

directory = './edf/eval/abnormal/'

print('Starting conversion of eval abnormal')
for root, dirs, files in os.walk(directory, topdown=False):
    i+=1
    for name in files:
        if name[-3:]=='edf':
            j+=1
            edfs.append(os.path.join(root, name))
            print(j,os.path.join(root, name))
            filename = os.path.join(root, name)
            f = pyedflib.EdfReader(filename)
            n = f.signals_in_file
            signal_labels = f.getSignalLabels()
            dict(zip(np.arange(0, len(signal_labels)), signal_labels))
            sigbufs = np.zeros((21, f.getNSamples()[0]))
            cnt = 0
            for k in np.arange(n):
                if signal_labels[k] in main_sig:
                    sigbufs[cnt, :] = f.readSignal(k)
                    cnt+=1
            nt = sigbufs.shape[1]
            saveM(sigbufs[:,:nt//2],j)
            j+=1
            saveM(sigbufs[:,nt//2:],j)
            
print("final")


###In[ ]:
##
##
##
## Change the following parameters
## 1 directory
## 2 np.ones and np.zeros
## 3 Path of the numpy files
i, j = 0, 0
edfs, tses = [], []

stype = 'train/1'
dname = './edf2'

directory = './edf/train/normal/'

print('Starting conversion of train normal')
for root, dirs, files in os.walk(directory, topdown=False):
    i+=1
    for name in files:
        if name[-3:]=='edf':
            j+=1
            edfs.append(os.path.join(root, name))
            print(j,os.path.join(root, name))
            filename = os.path.join(root, name)
            f = pyedflib.EdfReader(filename)
            n = f.signals_in_file
            signal_labels = f.getSignalLabels()
            dict(zip(np.arange(0, len(signal_labels)), signal_labels))
            sigbufs = np.zeros((21, f.getNSamples()[0]))
            cnt = 0
            for k in np.arange(n):
                if signal_labels[k] in main_sig:
                    sigbufs[cnt, :] = f.readSignal(k)
                    cnt+=1
            nt = sigbufs.shape[1]
            saveM(sigbufs[:,:nt//2],j)
            j+=1
            saveM(sigbufs[:,nt//2:],j)
            
print("final")

i, j = 0, 0
edfs, tses = [], []

stype = 'train/0'
dname = './edf2'

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
                sigbufs = np.zeros((21, f.getNSamples()[0]))
                cnt = 0
                for k in np.arange(n):
                    if signal_labels[k] in main_sig:
                        sigbufs[cnt, :] = f.readSignal(k)
                        cnt+=1
                nt = sigbufs.shape[1]
                saveM(sigbufs[:,:nt//2],j)
                j+=1
                saveM(sigbufs[:,nt//2:],j)
            except:
                print(j,os.path.join(root, name))
            
print("final")



