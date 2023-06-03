# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:17:18 2023

@author: Simon
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fishNet_nconv import  Evaluate, Model


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from cfg import Config
# Assuming you have X_train, X_test, y_train, y_test from train_test_split

MODE = "conv"
EPOCHS = 100
MFCC_FILTERS = 15
MFCC_FEATURES = 13
MFCC_NFFT = 1024
SAMPLE_RATE = 16000
MFCC_FMAX = 1250
TRANSFER_LEARN = False # default
LABELS = pd.read_csv(r"metadata/metadata_clean.csv" )
LABELS.set_index('file_names', inplace=True)

STEP = int(SAMPLE_RATE/4) # 250ms

# Train and evaluate your models
config100 = Config(mode=MODE,
                epochs=EPOCHS,
                nfilt=MFCC_FILTERS, 
                nfeat=MFCC_FEATURES,
                nfft=MFCC_NFFT,
                rate=SAMPLE_RATE, 
                hfreq=MFCC_FMAX,
                transferlearning=TRANSFER_LEARN,
                metadata=LABELS,
                step=STEP)
# =============================================================================
# 
# STEP = int(SAMPLE_RATE/8) # 125ms
# config125 = Config(mode=MODE,
#                 epochs=EPOCHS,
#                 nfilt=MFCC_FILTERS, 
#                 nfeat=MFCC_FEATURES,
#                 nfft=MFCC_NFFT,
#                 rate=SAMPLE_RATE, 
#                 hfreq=MFCC_FMAX,
#                 transferlearning=TRANSFER_LEARN,
#                 metadata=LABELS,
#                 step=STEP)
# 
# STEP = int(SAMPLE_RATE/5) # 200ms
# config200 = Config(mode=MODE,
#                 epochs=EPOCHS,
#                 nfilt=MFCC_FILTERS, 
#                 nfeat=MFCC_FEATURES,
#                 nfft=MFCC_NFFT,
#                 rate=SAMPLE_RATE, 
#                 hfreq=MFCC_FMAX,
#                 transferlearning=TRANSFER_LEARN,
#                 metadata=LABELS,
#                 step=STEP)
# 
# STEP = int(SAMPLE_RATE/4) # 250ms
# config250 = Config(mode=MODE,
#                 epochs=EPOCHS,
#                 nfilt=MFCC_FILTERS, 
#                 nfeat=MFCC_FEATURES,
#                 nfft=MFCC_NFFT,
#                 rate=SAMPLE_RATE, 
#                 hfreq=MFCC_FMAX,
#                 transferlearning=TRANSFER_LEARN,
#                 metadata=LABELS,
#                 step=STEP)
# =============================================================================
                
model100 = Model(config100)
model100.train_model()
# =============================================================================
# model125 = Model(config125)
# model125.train_model()
# model200 = Model(config200)
# model200.train_model()
# model250 = Model(config250)
# model250.train_model()
# =============================================================================


# Make predictions on the test set for each model
y_pred100 = model100.model.predict(model100.X_test)
# =============================================================================
# y_pred125 = model125.model.predict(model125.X_test)
# y_pred200 = model200.model.predict(model200.X_test)
# y_pred250 = model250.model.predict(model250.X_test)
# =============================================================================


# Calculate F2 scores for each model
# f2_scores = []
# f2_scores.append(fbeta_score(model100.y_test, y_pred100.round(), beta=2, average='weighted'))

# f2_scores.append(fbeta_score(model125.y_test, y_pred125.round(), beta=2, average='weighted'))
# f2_scores.append(fbeta_score(model200.y_test, y_pred200.round(), beta=2, average='weighted'))
# f2_scores.append(fbeta_score(model250.y_test, y_pred250.round(), beta=2, average='weighted'))

# Plot the F2 scores
models = ['2^2', '2^3', '2^4', '2^5']
f2_scores = [0.9316295303153687, 0.9725488340899626, 0.979206823697108 ,0.9515567342075746]
plt.bar(models, f2_scores, )
plt.xlabel('Models')
plt.ylabel('F2 Score')

for i, score in enumerate(f2_scores):
    plt.text(i, score, str(round(score, 2)), ha='center', va='bottom')
    
plt.title('F2 Scores of Varying CNN Filters')
plt.show()

# # Save the F2 scores to a CSV file
scores_df = pd.DataFrame({'Model': models, 'F2 Score': f2_scores})
scores_df.to_csv('f2_scores_n_filters.csv', index=False)
