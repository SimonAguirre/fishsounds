# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 00:53:14 2023

@author: Simon
"""
import pyqtgraph as pg
import numpy as np
from matplotlib import cm
import librosa
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication


class SpectrogramWidgetImage(pg.ImageItem):
    def __init__(self):
        super(SpectrogramWidgetImage, self).__init__()
        
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
        
        if self.img_array.shape[1] != psd.shape[1]:
            self.img_array = np.zeros((1000, psd.shape[1]))
            
        # roll down one and replace leading edge with new data
        self.img_array = np.roll(self.img_array, -psd.shape[0], 0)
        self.img_array[-psd.shape[0]:] = psd

        self.setImage(self.img_array, autoLevels=False)
        
class LabelWidgetImage(pg.ImageItem):
    def __init__(self, class_list):
        super(LabelWidgetImage, self).__init__()
        self.class_list = class_list
        
        self.data_img = np.fromfunction(lambda i, j: 0*(i/i), (1000, 32))
        
        pos = np.array([0., 0.25, 0.5, 0.75, 1.])
        color = np.array([[0,0,0,255], # black
                          [0,0,255,255],# blue
                          [0,255,0,255], # green
                          [255, 255, 0, 255], # yellow
                          [255, 0, 0, 255]], # red
                         dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 255)

        # set colormap
        self.setLookupTable(lut)
        self.setLevels([0,1])
        
        # self.addColorBar( self.data_img, colorMap=cmap, values=(0, 1))
        
        # assert len(class_list)+1 == len(color)
        # # setup the correct scaling for y-axis
        # freq = np.arange(librosa.mel_to_hz(13))
        # yscale = 1.0/(self.data_img.shape[1]/freq[-1])
        # self.setTransform(QtGui.QTransform.fromScale(1.5, 1))
        
    def update_img(self, class_i, n_pixels):   
        
        if class_i == None:
            mapper = 0
        else:
            mapper = (class_i+1)/len(self.class_list)
            
        self.new_data = np.fromfunction(lambda i, j: mapper*(i/i), (n_pixels, 32))
        
        # roll down one and replace leading edge with new data
        self.data_img = np.roll(self.data_img, -n_pixels, 0)
        self.data_img[-n_pixels:] = self.new_data

        self.setImage(self.data_img, autoLevels=False)


# if __name__=='__main__':
#     app = QApplication([])
#     win = pg.PlotWidget()
#     l_plotter = LabelWidgetImage(['knocks', 'croak', 'drums', 'grunt'])
#     win.addItem(l_plotter)
#     win.show()
#     l_plotter.update_img(1,20)
#     l_plotter.update_img(0,20)
#     l_plotter.update_img(None,20)
#     l_plotter.update_img(3,20)
#     l_plotter.update_img(2,20)
#     # l_ploter = 
#     app.exec_()
    
    