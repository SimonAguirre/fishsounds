# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:12:33 2023

@author: Simon
"""

import os
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.utils import to_categorical
from tqdm import tqdm
from python_speech_features import mfcc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.models import load_model
import librosa
from cfg import Config
import datetime as t
import pandas as pd

class Model():  
    def __init__(self, config):
        self.EPOCHS = config.epochs
        self.HFREQ = config.hfreq
        self.metadata = config.metadata
        self.MODE = config.mode
        self.MODEL_PATH = config.model_path
        self.N_FEATURES = config.nfeat
        self.N_FFT = config.nfft
        self.N_FILTERS = config.nfilt
        self.CONFIG_PATH = config.p_path
        self.RATE = config.rate
        self.STEP = config.step
        self.transferlearning = config.transferlearning
        
        self.INPUT_SHAPE_CNN = (13,9,1)
        self.INPUT_SHAPE_RNN = (13,9)
        
        self.NUM_CLASS = len(np.unique(config.metadata.Class))
        self.CLASSES = list(np.unique(config.metadata.Class))
        
        self.model = ...
        self.history = ...
        self.min = ...
        self.max = ...
        self.data = ...
        
        
    def create_model(self):
        if self.MODE == 'conv':
            self.model = keras.Sequential([
                layers.Conv2D(16, (3,3), activation='relu', strides=(1,1),
                                 padding='same', input_shape=self.INPUT_SHAPE_CNN),
                layers.Conv2D(32, (3,3), activation='relu', strides=(1,1),
                                  padding='same'),
                layers.Conv2D(64, (3,3), activation='relu', strides=(1,1),
                              padding='same'),
                layers.Conv2D(128, (3,3), activation='relu', strides=(1,1),
                                  padding='same'),
                layers.MaxPool2D((2,2)),
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(254, activation='relu'),
                layers.Dense(128, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(self.NUM_CLASS, activation='softmax')
                ])
        elif self.MODE == 'recur':
            self.model = keras.Sequential([
                layers.LSTM(128, return_sequences=True, input_shape=self.INPUT_SHAPE_RNN),
                layers.LSTM(128, return_sequences=True),
                layers.Dropout(0.5),
                layers.TimeDistributed(layers.Dense(64, activation='relu')),
                layers.TimeDistributed(layers.Dense(32, activation='relu')),
                layers.TimeDistributed(layers.Dense(16, activation='relu')),
                layers.TimeDistributed(layers.Dense(8, activation='relu')),
                layers.Flatten(),
                layers.Dense(self.NUM_CLASS, activation = 'softmax'),
                ])
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy',
                      optimizer= optimizers.Adam(learning_rate=0.0001),
                      metrics=['acc'])
        
    def plot_evaluation(self):
        plt.figure(figsize=(12, 4))
        history_path = f"{self.MODEL_PATH[:-5]}.csv"
        history_df = pd.DataFrame(columns = ['acc','val_acc','loss','val_loss'])
        # Accuracy Curves
        plt.subplot(1, 2, 1)
        
        plt.plot(self.history.history['acc'], label='Training Accuracy')
        history_df['acc'] = self.history.history['acc']
        plt.plot(self.history.history['val_acc'], label='Validation Accuracy')
        history_df['val_acc'] = self.history.history['val_acc']
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss Curves
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        history_df['loss'] = self.history.history['loss']
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        history_df['val_loss'] = self.history.history['val_loss']
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        os.makedirs(r"models/checkpoint/metrics", exist_ok=True)
        history_df.to_csv(f"models/checkpoint/metrics/{self.MODEL_PATH[:-5]}.csv", index=False)
        with open(r'models/checkpoint/metrics/modelsummary.txt', 'w') as summary_txt:
            self.model.summary(print_fn=lambda x: summary_txt.write(x + '\n'))
        
    def check_data(self):
        try:
            if self.transferlearning or os.path.isfile(self.CONFIG_PATH):
                print("Loading existing data for {} model".format(self.MODE))
                with open(self.CONFIG_PATH, 'rb') as handle:
                    tmp = pickle.load(handle)
                    return tmp
            else:
                return None
        except:
            return None
        
    # def build_feat(self):
    #     tmp = self.check_data()
    #     if tmp:
    #         return tmp.data[0], tmp.data[1]
    #     X = []
    #     y = []
    #     _min, _max = float('inf'), -float('inf')
    #     # n_samples = 2 * int(self.metadata['length'].sum()/(self.STEP/self.RATE))
    #     class_dist = self.metadata.groupby(['Class'])['length'].mean()
    #     prob_dist = class_dist / class_dist.sum()
        
    #     for _ in tqdm(self.metadata['file_names']):
    #         rand_class = np.random.choice(class_dist.index, p=prob_dist)
    #         file = np.random.choice(self.metadata[self.metadata.Class == rand_class].index)
    #         wav, rate = librosa.load("clean/"+file, sr=self.RATE)
    #         label = self.metadata.at[file, 'Class']
    #         print(wav.shape[0]-self.STEP)
    #         if self.STEP >= wav.shape[0]:
    #             continue
    #         rand_index = np.random.randint(0, wav.shape[0]-self.STEP)
    #         sample = wav[rand_index:rand_index+self.STEP]
    #         X_sample = mfcc(sample, rate,
    #                         numcep=self.N_FEATURES,
    #                         nfilt=self.N_FILTERS,
    #                         nfft=self.N_FFT,
    #                         highfreq=self.HFREQ).T
    #         _min = min(np.amin(X_sample), _min)
    #         _max = max(np.amax(X_sample), _max)
    #         X.append(X_sample)
    #         y.append(self.CLASSES.index(label))
            
    #     self.min = _min
    #     self.max = _max
        
    #     X, y = np.array(X), np.array(y)
    #     X = (X -_min)/(_max-_min)
        
    #     if (self.MODE =='conv'):
    #         X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    #     elif self.MODE == 'recur':
    #         X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    #     y = to_categorical(y, num_classes=self.NUM_CLASS)
        
    #     self.data = (X, y)
    #     return X, y
    
    def build_rand_feat(self):
        tmp = self.check_data()
        if tmp:
            return tmp.data[0], tmp.data[1]
        X = []
        y = []
        _min, _max = float('inf'), -float('inf')
        n_samples = 2 * int(self.metadata['length'].sum()/(self.STEP/self.RATE))
        class_dist = self.metadata.groupby(['Class'])['length'].mean()
        prob_dist = class_dist / class_dist.sum()
        
        for _ in tqdm(range(n_samples)):
            rand_class = np.random.choice(class_dist.index, p=prob_dist)
            file = np.random.choice(self.metadata[self.metadata.Class == rand_class].index)
            wav, rate = librosa.load("clean/"+file, sr=self.RATE)
            label = self.metadata.at[file, 'Class']
            print(wav.shape[0]-self.STEP)
            if self.STEP >= wav.shape[0]:
                continue
            rand_index = np.random.randint(0, wav.shape[0]-self.STEP)
            sample = wav[rand_index:rand_index+self.STEP]
            X_sample = mfcc(sample, rate,
                            numcep=self.N_FEATURES,
                            nfilt=self.N_FILTERS,
                            nfft=self.N_FFT,
                            highfreq=self.HFREQ).T
            _min = min(np.amin(X_sample), _min)
            _max = max(np.amax(X_sample), _max)
            X.append(X_sample)
            y.append(self.CLASSES.index(label))
            
        self.min = _min
        self.max = _max
        
        X, y = np.array(X), np.array(y)
        X = (X -_min)/(_max-_min)
        
        if (self.MODE =='conv'):
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        elif self.MODE == 'recur':
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        y = to_categorical(y, num_classes=self.NUM_CLASS)
        
        self.data = (X, y)
        return X, y
    
    def train_model(self):
        X, y = self.build_rand_feat()
        y_flat = np.argmax(y, axis=1)
        # self.num_class = len(np.unique(y_flat))
        if self.MODE == 'conv':
            self.INPUT_SHAPE_CNN = (X.shape[1], X.shape[2], 1)
        elif self.MODE == 'recur':
            self.INPUT_SHAPE_RNN = (X.shape[1], X.shape[2])
            
        self.create_model()
        class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_flat), y=y_flat)
        class_weight = dict(enumerate(class_weight))
        
        datetime = t.datetime.strptime(str(t.datetime.now())[:16], '%Y-%m-%d %H:%M')
        datetime = str(datetime).replace(' ', '_').replace(':', '-')

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        checkpoint_path = f"models/checkpoint/fishNet-{self.MODE}-model_{datetime}_*epoch*.hdf5"
        checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:03d}")
        
        checkpoint = ModelCheckpoint(checkpoint_path,
                                     monitor='val_acc',
                                     verbose=1, mode='max', save_best_only=True,
                                     save_weight_only=False, save_freq="epoch")
                
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=69)
        
        self.history = self.model.fit(self.X_train, self.y_train, epochs=self.EPOCHS, batch_size=len(self.X_train)//100, 
                            shuffle=True, class_weight=class_weight, 
                            validation_data=(self.X_test, self.y_test),
                            callbacks=[checkpoint])
        
        self.CONFIG_PATH = f"models/pickles/fishNet-{self.MODE}-model_{datetime}.p"
        self.MODEL_PATH =  sorted(os.listdir("models/checkpoint"))[-1]  # get the latest model path
        self.model.save(self.MODEL_PATH)
        
        os.makedirs(r"models/pickles", exist_ok=True)
        
        with open(self.CONFIG_PATH, 'wb') as handle:
            new_config = Config(mode = self.MODE,
                            epochs = self.EPOCHS,
                            nfilt = self.N_FILTERS,
                            nfeat = self.N_FEATURES,
                            nfft = self.N_FFT,
                            rate = self.RATE,
                            hfreq = self.HFREQ,
                            transferlearning = self.transferlearning,
                            
                            model_path = self.MODEL_PATH,
                            p_path = self.CONFIG_PATH,
                            metadata = self.metadata,
                            data_min = self.min,
                            data_max = self.max,
                            data = self.data
                            )
            pickle.dump(new_config, handle, protocol=2)
            
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)
        self.plot_evaluation()

class Evaluate():    
    def __init__(self, config_path, **kwargs):
        with open(config_path, 'rb') as handle:
            self.config = pickle.load(handle)
        self.model = load_model(self.config.model_path)

        try:
            self.test_dataframe = kwargs['labels']
        except:
            self.test_dataframe = self.config.metadata
        
        self.classes = list(np.unique(self.test_dataframe.Class))
        self.fn2class = dict(zip(self.test_dataframe.file_names, self.test_dataframe.Class))
        
        
    def evaluate_testset(self, **kwargs):
        try:
            audio_dir = kwargs['audio_path']
            
        except:
            audio_dir = 'clean'
        y_true, y_pred, fn_prob = self.build_predictions(self.config, audio_dir)
        
        if "Class" in list(self.test_dataframe):
            acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)
        else:
            acc_score = "[error] no provided ground truth"
            
        y_probs = []
        for i, row in self.test_dataframe.iterrows():
            y_prob = fn_prob[row.file_names]
            y_probs.append(y_prob)
            for c, p in zip(self.classes, y_prob):
                self.test_dataframe.at[i, c] = p
            
        y_pred = [self.classes[np.argmax(y)] for y in y_probs]
        self.test_dataframe['Prediction'] = y_pred
        print(f"Prediction Accuracy: {acc_score}")
        print("Saving Predictions to CSV")
        self.test_dataframe.to_csv(f"predictions_{self.config.mode}.csv", index=False)
        
    def build_predictions(self, config, audio_dir):
        y_true = []
        y_pred = []
        fn_prob = {}
        
        print('Extracting features from audio')
        
        for fn in tqdm(os.listdir(audio_dir)):
            wav, rate = librosa.load(os.path.join(audio_dir, fn), sr=config.rate)
            label = self.fn2class[fn]
            c = self.classes.index(label)
            y_prob = []
            # print(f"Extracting features from {fn}")
            for i in range(0, wav.shape[0]-config.step, config.step):
                sample = wav[i:i+config.step]
                x = mfcc(sample, rate,
                          numcep=config.nfeat,
                          nfilt=config.nfilt,
                          nfft=config.nfft,
                          highfreq=config.hfreq).T
                # print(f"A:{x}")
                x = (x-config.min)/(config.max - config.min)
                # print(f"B:{x}")
                if config.mode == 'conv':
                    x = x.reshape(1, x.shape[0], x.shape[1], 1)
                elif config.mode == 'recur':
                    x = np.expand_dims(x, axis=0)
                y_hat = self.model.predict(x, verbose=0)
                # print(f"YHAT: {y_hat}")
                y_prob.append(y_hat)
                y_pred.append(np.argmax(y_hat))
                y_true.append(c)
            fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
            # print(f"MEAN: {fn_prob[fn]}")
            
        return y_true, y_pred, fn_prob   
    
class Predict():    
    def __init__(self, config_path, **kwargs):
        with open(config_path, 'rb') as handle:
            config = pickle.load(handle)
            
        self.EPOCHS = config.epochs
        self.HFREQ = config.hfreq
        self.metadata = config.metadata
        self.MODE = config.mode
        self.MODEL_PATH = config.model_path
        self.N_FEATURES = config.nfeat
        self.N_FFT = config.nfft
        self.N_FILTERS = config.nfilt
        self.CONFIG_PATH = config.p_path
        self.RATE = config.rate
        self.STEP = config.step
        self.transferlearning = config.transferlearning
        
        self.INPUT_SHAPE_CNN = (13,9,1)
        self.INPUT_SHAPE_RNN = (13,9)
        
        self.NUM_CLASS = len(np.unique(config.metadata.Class))
        self.CLASSES = list(np.unique(config.metadata.Class))
        
        self.model = load_model(config.model_path)
        self.min = config.min
        self.max = config.max
   
    def predict(self, **kwargs):
        try:
            self.df = kwargs['test_set']
        except:
            return None
        
        # self.fn2class = dict(zip(self.test_dataframe.file_names, self.test_dataframe.Class))
        
        for i in self.df.index:
            self.df.at[i,'path'] = self.df.at[i,'directory'] + r'/' + self.df.at[i, 'file_name']
            
        self.build_predictions(self.df['path'])
        
        return self.df.drop(columns=['path'])
        
    def build_predictions(self, audio_list):
        y_pred = []
       
        print('Extracting features from audio')
       
        for curr_i in tqdm(self.df.index): # for every audio in df
            fn = self.df.at[curr_i, 'path']
            if self.df.at[curr_i, 'events'] == 0 and self.df.at[curr_i+1, 'events'] == 1:
                continue
            self.start = int(self.df.at[curr_i, 'event_start'])
            self.end = int(self.df.at[curr_i, 'event_end'])
            
            wav, rate = librosa.load(fn, sr=self.RATE) 
            wav = wav[self.start:self.end] # get the signal too be classified
            
            # label = self.fn2class[fn]     # what is the label of this file
            # c = self.CLASSES.index(label) # index of this class in list of classes
            
            y_prob = []
            # print(f"Extracting features from {fn}")
            # for i in each window from 0 to last signal point
            
            for i in range(0, wav.shape[0]-self.STEP, self.STEP): 
                # get each window
                sample = wav[i:i+self.STEP]
                # get mfcc of this window (13,9)
                x = mfcc(sample, rate,
                          numcep=self.N_FEATURES,
                          nfilt=self.N_FILTERS,
                          nfft=self.N_FFT,
                          highfreq=self.HFREQ).T
                
                # normalize the x
                x = (x-self.min)/(self.max - self.min)
                # reshape the x to match the model fitting
                if self.MODE == 'conv':
                    x = x.reshape(1, x.shape[0], x.shape[1], 1)
                elif self.MODE == 'recur':
                    x = np.expand_dims(x, axis=0)
                # store the prediction (4,), four classes
                y_hat = self.model.predict(x, verbose=0)
                
                # y_prob stores (y_hat, y_hat, y_hat)
                y_prob.append(y_hat)
                
                # y_pred stores the index that contains the highest value 
                y_pred.append(np.argmax(y_hat))
            
            # get the mean of predictions from each window
            y_prob = np.mean(y_prob, axis=0).flatten()
            # get the actual prediction and score for this event
            pred = self.CLASSES[np.argmax(y_prob)]
            pred_score = y_prob[np.argmax(y_prob)]
            
            if pred_score > 0.5:
                self.df.at[curr_i, 'Prediction'] = pred
                self.df.at[curr_i, 'Score'] = pred_score