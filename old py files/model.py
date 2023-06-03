import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from keras.callbacks import ModelCheckpoint
from cfg import Config 



def check_data():
    if os.path.isfile(config.p_path):
        print("Loading existing data for {} model".format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None
    
def get_conv_model():
    model = Sequential()
    
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1),
                     padding='same'))
    
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(NUM_CLASS, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer= Adam(learning_rate=LEARNING_RATE),
                  metrics=['acc'])
    return model

def get_recurrent_model():
    #shape of data for RNN is (n, time, feat)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(NUM_CLASS, activation = 'softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer= Adam(learning_rate=LEARNING_RATE),
                  metrics=['acc'])
    return model
    
def build_rand_feat():
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

def plot_evaluation(model_history):
    plt.figure(figsize=(12, 4))
    
    # Accuracy Curves
    plt.subplot(1, 2, 1)
    plt.plot(model_history.history['acc'], label='Training Accuracy')
    plt.plot(model_history.history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss Curves
    plt.subplot(1, 2, 2)
    plt.plot(model_history.history['loss'], label='Training Loss')
    plt.plot(model_history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

NUM_CLASS = 4
LEARNING_RATE = 0.0001
MFCC_FILTERS = 15
MFCC_FEATURES = 13
MFCC_FMAX = 1250
MODE = "time"
EPOCHS = 100
BATCH_SIZE = 47


df = pd.read_csv('metadata/metadata_clean.csv')
df.set_index('file_names', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f,'length'] = signal.shape[0]/rate

df['Class'] = df['Class'].apply(lambda x: x.strip())
classes = list(np.unique(df.Class))
class_dist = df.groupby(['Class'])['length'].mean()

n_samples = 2 * int(df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
# choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title("Class Distribution", y=1.08)
ax.pie(class_dist, labels=class_dist.index,
       autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = Config(mode=MODE, hfreq=MFCC_FMAX, nfilt=MFCC_FILTERS, nfeat=MFCC_FEATURES)

if config.mode == 'conv':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()
    
elif config.mode == 'time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_flat), y=y_flat)
class_weight = dict(enumerate(class_weight))

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc',
                             verbose=1, mode='max', save_best_only=True,
                             save_weight_only=False, save_freq="epoch")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                    shuffle=True, class_weight=class_weight, 
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint])

model.save(config.model_path)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

plot_evaluation(history)







