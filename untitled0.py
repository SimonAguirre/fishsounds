# -*- coding: utf-8 -*-
"""
Created on Mon May 29 23:15:24 2023

@author: Simon
"""

"""
Tested on Linux with python 3.7
Must have portaudio installed (e.g. dnf install portaudio-devel)
pip install pyqtgraph pyaudio PyQt5
"""
import numpy as np
import pyqtgraph as pg
import pyaudio
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication
import librosa
from matplotlib import cm




FS = 22050 #Hz
CHUNKSZ = 1024 #samples

class MicrophoneRecorder():
    def __init__(self, signal):
        self.signal = signal
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=FS,
                            input=True,
                            frames_per_buffer=CHUNKSZ)

    def read(self):
        data = self.stream.read(CHUNKSZ, exception_on_overflow=False)
        y = np.fromstring(data, 'int16')
        self.signal.emit(y)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class SpectrogramWidget(pg.PlotWidget):
    read_collected = QtCore.pyqtSignal(np.ndarray)
    def __init__(self):
        super(SpectrogramWidget, self).__init__()

        self.img = pg.ImageItem()
        self.addItem(self.img)
        
        self.img_array = np.zeros((1000//2, 513))

        # self.img_array = np.zeros((1000//2, (CHUNKSZ)*2+1))

        # # bipolar colormap
        # pos = np.array([0., 1., 0.5, 0.25, 0.75])
        # color = np.array([[0,255,255,255], [0,255,255,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        # cmap = pg.ColorMap(pos, color)
        # lut = cmap.getLookupTable(0.0, 1.0, 1024)

        # # set colormap
        # self.img.setLookupTable(lut)
        # self.img.setLevels([-100,50])
        
        # colormap
        colormap = cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-70,90])
        

        # setup the correct scaling for y-axis
        
        # self.plot1.getaxis('left').setTicks([[(self.freq, str(self.freq)) for freq in self.freq[]])
        freq = np.arange((CHUNKSZ/2)+1)/(float(CHUNKSZ)/FS)
        yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        self.img.setTransform(QtGui.QTransform.fromScale((1./FS)*CHUNKSZ, yscale))

        self.setLabel('left', 'Frequency', units='Hz',)
        self.setMouseEnabled(x=True, y=False)
        # prepare window for later use
        self.win = np.hanning(1024)
        self.show()

    def update(self, chunk):
        # normalized, windowed frequencies in data chunk
        
        # print(chunk.shape)
        # spec = np.fft.rfft(chunk*self.win, n=4096) / CHUNKSZ
        # print(spec.shape)
        # print(f"SAAAAAAAAAAAAAAAAAAAAAAAAAAAAA1{chunk.max()}")
        chunk = chunk/700
        # chunk = librosa.resample(chunk*np.hanning(1024), orig_sr=22050, target_sr=2205, res_type='kaiser_best')
        # print(f"SAAAAAAAAAAAAAAAAAAAAAAAAAAAAA2{chunk.max()}")
        
        # spec = librosa.feature.melspectrogram(y=chunk,
        #                                       n_fft=len(chunk), hop_length= len(chunk)//5,
        #                                       n_mels=16, center=True).T
        
        # spec = librosa.
        
        spec = np.fft.rfftfreq(chunk)/1024*2
        spec= abs( 2595*np.log10(1+(spec/700)))

        
        # get magnitude 
        psd = abs(spec)
        # convert to dB scale
        psd = 20* np.log10(psd)
        

        # roll down one and replace leading edge with new data
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = psd

        self.img.setImage(self.img_array, autoLevels=False)

if __name__ == '__main__':
    app = QApplication([])
    w = SpectrogramWidget()
    w.read_collected.connect(w.update)

    mic = MicrophoneRecorder(w.read_collected)

    # time (seconds) between reads
    interval = 1024
    t = QtCore.QTimer()
    t.timeout.connect(mic.read)
    t.start(10) #QTimer takes ms

    app.exec_()
    mic.close()