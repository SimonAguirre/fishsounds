# -*- coding: utf-8 -*-
"""
Created on Sat May 27 00:52:20 2023

@author: Simon
"""
class Config:
    def __init__(self,
                 mode='conv',
                 epochs=10,
                 nfilt=15,
                 nfeat=13,
                 nfft=512, 
                 rate=8000, 
                 hfreq=4000,
                 transferlearning=False,
                 **kwargs
                 ):
        
        self.mode = mode
        self.epochs = epochs
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.hfreq = hfreq
        self.step = int(rate/5)
        self.transferlearning = transferlearning
        try:
            self.step = kwargs['step']
        except:
            pass
        try:
            self.model_path = kwargs['model_path']
        except:
            self.model_path = None
        try:
            self.p_path = kwargs['p_path']
        except:
            self.p_path = None
        try:
            self.metadata = kwargs['metadata']
        except:
            self.metadata = None
        try:
            self.min = kwargs['data_min']
        except:
            self.min = None
        try:
            self.max = kwargs['data_max']
        except:
            self.max = None
        try:
            self.data = kwargs['data']
        except:
            self.data = None

    def setTransferlearning(self, transferlearning):
        self.transferlearning = transferlearning
        
    def setMetadata(self, metadata):
        self.metadata = metadata