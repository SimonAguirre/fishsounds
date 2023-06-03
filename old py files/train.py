# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:43:59 2023

@author: Simon
"""
import os
import pickle
import tqdm
import numpy as np
import fishNet


def check_data(config):
    if os.path.isfile(config.p_path):
        print("Loading existing data for {} model".format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None
    
def build_train_feat(n_samples, class_dist, prob_dist):
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.Class == rand_class].index)
        rate, wav = wavfile.read("clean/"+file)
        label = df.at[file, 'Class']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                        numcep=config.nfeat,
                        nfilt=config.nfilt,
                        nfft=config.nfft,
                        highfreq=config.hfreq).T
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(label))
        
    config.min = _min
    config.max = _max
    
    X, y = np.array(X), np.array(y)
    X = (X -_min)/(_max-_min)

    if (config.mode =='conv'):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=NUM_CLASS)
    
    config.data = (X, y)
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)
    return X, y
    

if __name__ == "__main__":
    X, y = build_train_feat()