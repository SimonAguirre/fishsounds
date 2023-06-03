import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

def plot_signals(signals):
    rows, cols = 1, 4
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=False, 
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i=0
    for x in range(rows):
        for y in range(cols):
            axes[y].set_title(list(signals.keys())[i])
            axes[y].plot(list(signals.values())[i])
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1
            
def plot_fft(fft):
    rows, cols = 1, 4
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=False, 
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i=0
    for x in range(rows):
        for y in range(cols):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[y].set_title(list(fft.keys())[i])
            axes[y].plot(freq, Y)
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1
            
def plot_fbank(fbank):
    rows, cols = 1, 4
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=False, 
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i=0
    for x in range(rows):
        for y in range(cols):
            axes[y].set_title(list(fbank.keys())[i])
            axes[y].imshow(list(fbank.values())[i],
                             cmap='hot', interpolation='nearest')
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    rows, cols = 1, 4
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=False, 
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i=0
    for x in range(rows):
        for y in range(cols):
            axes[y].set_title(list(mfccs.keys())[i])
            axes[y].imshow(list(mfccs.values())[i],
                             cmap='hot', interpolation='nearest')
            axes[y].get_xaxis().set_visible(False)
            axes[y].get_yaxis().set_visible(False)
            i += 1

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask
    
def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)


THRESHOLD = 0.0001

df = pd.read_csv(r'C:/Users/Simon/fishsounds/metadata/metadata.csv')
classes_select = ['knocks', 'croak', 'drums', 'grunt']
df = df.loc[df['Class'].isin(classes_select)]
df.set_index('file_names', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('C:/Users/Simon/fishsounds/denoised-segmented/'+f)
    df.at[f,'length'] = signal.shape[0]/rate

df['Class'] = df['Class'].apply(lambda x: x.strip())
classes = list(np.unique(df.Class))
class_dist = df.groupby(['Class'])['length'].mean()


fig, ax = plt.subplots()
ax.set_title("Class Distribution", y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()

df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    wav_file = df[df.Class == c].iloc[0,0]
    signal, rate = librosa.load("C:/Users/Simon/fishsounds/denoised-segmented/"+wav_file, sr=22050)
    mask = envelope(signal, rate, threshold=THRESHOLD)
    signal = signal[mask]
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)
    bank = logfbank(signal[:int(rate/2)], rate, nfilt=15, nfft=551).T # nfft = rate/40  40 is tthe window size
    fbank[c] = bank
    mel = mfcc(signal[:int(rate/2)], rate, numcep=13, nfilt=15, nfft=551).T
    mfccs[c] = mel

plot_signals(signals)
plt.show()
plot_fft(fft)
plt.show()
plot_fbank(fbank)
plt.show()
plot_mfccs(mfccs)
plt.show()


if (len(os.listdir('C:/Users/Simon/fishsounds/clean'))==0):
    for f in tqdm(df.file_names):
        signal, rate = librosa.load("C:/Users/Simon/fishsounds/denoised-segmented/"+f, sr = 8000)
        mask = envelope(signal, rate, THRESHOLD)
        print("Writing "+str(len(signal)))
        wavfile.write(filename = 'C:/Users/Simon/fishsounds/clean/'+f, rate=rate, data=signal)

df.to_csv("C:/Users/Simon/fishsounds/metadata/metadata_clean.csv", sep=",",index=False)


