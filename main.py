# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:51:02 2023

@author: Simon
"""

from fishNet import Model, Predict
from audioplayer import Player
from processors import AudioProcessor
from plotter import LabelWidgetImage

import pandas as pd
from cfg import Config
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5 import uic
import os
import threading as th
from time import perf_counter


#plotter melspec
import numpy as np
import pyqtgraph as pg
import pyaudio
# from PyQt5 import QtCore, QtGui use PyQt5.QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
import librosa
import datetime as t
from matplotlib import cm

df = pd.read_csv('metadata/metadata_clean.csv')
df.set_index('file_names', inplace=True)

MODE = "conv"
EPOCHS = 100
MFCC_FILTERS = 15
MFCC_FEATURES = 13
MFCC_NFFT = 1024
SAMPLE_RATE = 16000
MFCC_FMAX = 1250
TRANSFER_LEARN = False # default
STEP = int(SAMPLE_RATE/4)

# BATCH_SIZE = 32
# LABELS = df
CONFIG_PATH = r"C:\Users\Simon\fishsounds\models\16khz_conv_window-250ms\pickles\fishNet-conv-model_2023-05-31_23-53-00.p"
# TEST_AUDIO_DIR = "clean"

# config = Config(mode=MODE,
#                 epochs=EPOCHS,
#                 nfilt=MFCC_FILTERS, 
#                 nfeat=MFCC_FEATURES,
#                 nfft=MFCC_NFFT,
#                 rate=SAMPLE_RATE, 
#                 hfreq=MFCC_FMAX,
#                 transferlearning=TRANSFER_LEARN,
#                 metadata=LABELS,
#                 step=STEP)

# # train model
# print("Training")
# model = Model(config)
# model.train_model()


# predict using model
# print("Predicting")
predictor = Predict(CONFIG_PATH)
predictions = ...
# predictor.predict(audio_path=TEST_AUDIO_DIR)

class SpectrogramWidget(pg.ImageItem):
    def __init__(self):
        super(SpectrogramWidget, self).__init__()
        
        self.CHUNKSZ = 1024
        self.FS = 22050
        self.img_array = np.zeros((1000, 32))
        
        # colormap
        colormap = cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        
        # set colormap
        self.setLookupTable(lut)
        self.setLevels([-70,70])
        
        # setup the correct scaling for y-axis
        # freq = np.arange(librosa.mel_to_hz(13))
        # yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        # self.setTransform(QtGui.QTransform.fromScale((1./self.FS)*self.CHUNKSZ, yscale))

    def update_img(self, sig=None, sr=None):
        
        sig = librosa.resample(sig*np.hanning(1024)*100, orig_sr=sr, 
                                target_sr=round(sr/10), res_type='kaiser_best')
        spec = librosa.feature.melspectrogram(y=sig,sr=sr, n_fft=len(sig), hop_length=len(sig)//8,
                                              n_mels=32, center=True).T
        
        # spec = np.fft.rfft(sig)/1024*2
        # spec= 2595*np.log10(1+(spec/700))
        
        # get magnitude 
        psd = abs(spec)
        # convert to dB scale
        psd = 20* np.log10(psd)

        # roll down one and replace leading edge with new data
        self.img_array = np.roll(self.img_array, -9, 0)
        self.img_array[-9:] = psd

        self.setImage(self.img_array, autoLevels=False)
        

class MyGUI(QMainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi("GUI/pyQt/main.ui", self)
        
        self.listenButton.clicked.connect(self.listen)
        self.loadButton.clicked.connect(self.load)
        self.stopButton.clicked.connect(self.stop)
        self.playButton.clicked.connect(self.play_pause)
        self.nextButton.clicked.connect(self._next)
        self.prevButton.clicked.connect(self._prev)
        self.saveButton.clicked.connect(self.save)

        #---------------------------------------
        self.melspecPlot = SpectrogramWidget()
        
        self.plot1 = self.plotterView.addPlot()
        self.plot1.setMouseEnabled(x=True, y=False)
        self.plot1.hideAxis('left')
        self.plot1.hideAxis('bottom')
        self.plot1.hideAxis('right')
        self.plot1.hideAxis('top')
        self.plot1.setLimits(yMin=0)
        self.plot1.setLogMode(False, True)
        self.plot1.addItem(self.melspecPlot)
        
        #---------------------------------------
        self.audioListWidget.currentItemChanged.connect(lambda: self.select_song())
        self.audioListWidget.currentItemChanged.connect(lambda: self.seek())
        #---------------------------------------
        self.labelsPlot = LabelWidgetImage(predictor.CLASSES)
        
        self.plot2 = self.labelPlotterView.addPlot()
        
        self.plot2.setXLink(self.plot1)
        self.plot2.hideAxis('left')
        self.plot2.hideAxis('bottom')
        self.plot2.hideAxis('right')
        self.plot2.hideAxis('top')
        self.plot2.addItem(self.labelsPlot)
        
        print("plot window created")
        
        #---------------------------------------
        
        
    def get_class_id(self, current_frame):
        print(f"called on {current_frame}")
        for index in predictions.index:
            if predictions.at[index, 'Prediction']==None:
                continue
            if predictions.at[index, 'file_name']!=audio_player.current_audio:
                continue
            if current_frame >= predictions.at[index, 'event_start'] and current_frame < predictions.at[index, 'event_end']:
                try:
                    return predictor.CLASSES.index(predictions.at[index, 'Prediction'])
                except:
                    return None
            else:
                continue
                    
    def terminate(self):
        app.closeAllWindows()
        
    def listen(self):
        pass
    
    def seek(self):
        pass
        
    def select_song(self):
        print(f"Selected {self.audioListWidget.currentRow()}")
        audio_player.audio_list.index = self.audioListWidget.currentRow()
        self.load_events()
        
    def load_events(self):
        self.eventListWidget.clear()
        item = QtWidgets.QListWidgetItem(f"Events\t| Class\t| Score\t| Start\t| End", self.eventListWidget)
        for index in predictions.index:
            # if  predictions.at[index, 'events']==0 and predictions.at[index+1, 'events']==1:
            if  predictions.at[index, 'Prediction']==None:
                continue
            if predictions.at[index, 'file_name']!=audio_player.current_audio:
                continue
            event_id = f"{audio_player.current_audio[:5]}...{predictions.at[index, 'events']}"
            class_p = predictions.at[index, 'Prediction']
            score = str(predictions.at[index, 'Score'])[:3]
            start = str(predictions.at[index, 'event_start']/predictor.RATE)[:5]
            end = str(predictions.at[index, 'event_end']/predictor.RATE)[:5]
            item = f"{event_id}\t{class_p}\t{score}\t{start}\t{end}"
            item = QtWidgets.QListWidgetItem(item, self.eventListWidget)
            
        
    def load(self):
        self.audioListWidget.clear() 
        fname, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;WAV File (*.wav)")
        audio_dir = fname[:fname.rindex("/")]
        audio_player.set_playlist(audio_dir, fname=fname[fname.rindex("/")+1:])
        self.audioTrackLabel.setText(f"Audio Track {audio_player.audio_list.index+1}: {audio_player.current_audio}")
        self.audioListWidget.setCurrentRow(audio_player.audio_list.index)
        
        
        for index, audio in enumerate(audio_player.audio_list.list):
            pos = 0
            print(f"Segmenting {index+1}/{len(audio_player.audio_list.list)} : {audio}")
            audio_processed = AudioProcessor(audio_dir+"/"+audio)
            
            rate_factor = predictor.RATE/audio_processed.audiobackup.frame_rate
            
            loaded_audios.loc[len(loaded_audios)]={'file_name': audio,
                                                   'directory': audio_dir,
                                                   'events': 0,
                                                   'event_start': 0,
                                                   'event_end': int(audio_processed.audiobackup.frame_count()*rate_factor),
                                                   'Prediction': None}
            for indexx, event in  enumerate(audio_processed.audios):
                print(event)
                loaded_audios.loc[len(loaded_audios)]={'file_name': audio,
                                                       'directory': audio_dir,
                                                       'events': indexx+1,
                                                       'event_start': int(pos*rate_factor),
                                                       'event_end': int((pos + event.frame_count())*rate_factor),
                                                       'Prediction': None}
                pos += event.frame_count()
        for index, fname in enumerate(loaded_audios[loaded_audios['events']==0]['file_name']):
            item = QtWidgets.QListWidgetItem(f"{audio_player.audio_list.get_index(fname=fname)+1}. {fname}", self.audioListWidget)
        global predictions
        predictions = predictor.predict(test_set = loaded_audios)
        self.load_events()
        
        
        # for i in predictions.index:
            
        
    def stop(self):
        self.playButton.setChecked(False)
        audio_player.stop_audio()

    def save(self):
        datetime = t.datetime.strptime(str(t.datetime.now())[:16], '%Y-%m-%d %H:%M')
        datetime = str(datetime).replace(' ', '_').replace(':', '-')
        
        save_fn = os.path.join(loaded_audios['directory'][0], 'label.csv')
        
        loaded_audios.to_csv(save_fn, index=False)
    
    def play_pause(self):
        if self.playButton.isChecked():
            audio_player.play_audio()
            self.timeStampLabel.setText(self.format_time(audio_player.current_frame/audio_player.sr))
        else:
            audio_player.pause_audio()

    def _next(self):
        audio_player.next_audio()
        self.audioTrackLabel.setText(f"Audio Track {audio_player.audio_list.index+1}: {audio_player.current_audio}")
        self.audioListWidget.setCurrentRow(audio_player.audio_list.index)
        self.load_events()
        
    def _prev(self):
        audio_player.prev_audio()
        self.audioTrackLabel.setText(f"Audio Track {audio_player.audio_list.index+1}: {audio_player.current_audio}")
        self.audioListWidget.setCurrentRow(audio_player.audio_list.index)
        self.load_events()
        
    def train(self):
        print("Training")
        self.train_thread = th.Thread(self.train_parallel())
        self.train_thread.start()
        
    def train_parallel(self):
        self.model = Model()
        self.model.train_model(config)
    
    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}h{minutes:02d}m{seconds:06.3f}s"

    
if __name__ == '__main__':
    try:
        app = QtWidgets.QApplication([])
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
        app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        window = MyGUI()
        print("Player importing")
        audio_player = Player(window)
        print("Player imported")
        print("Creating audio database")
        loaded_audios = pd.DataFrame({'file_name': ['audio1', 'audio2'],
                                      'directory': ['None', 'None'],
                                      'events': ['event1', 'event2'],
                                      'event_start': [0.0,220],
                                      'event_end': [10,300],
                                      'Prediction': ['grunt', 'drums']
                                      })
        
        for audio in loaded_audios['file_name']:
            loaded_audios.drop(loaded_audios[loaded_audios['file_name']== audio].index, inplace=True)
        print(loaded_audios)
        print("Audio database ready")
        
        window.show()
        app.exec_()
    except:
        pass
    

    

