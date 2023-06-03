# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:16:52 2023

@author: Simon
"""
import os
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt

# audio_path = r"C:\Users\Simon\fishsounds\clean\817,883,etc_HawkinsT-2002_Melanogrammus-aeglefinus_Rumble_event_3.wav"
class AudioProcessor():
    def __init__(self, audio_path):
        self.audio_path = audio_path
        
        try:
            self.audio = AudioSegment.from_file(audio_path)
        except:
            self.audio = AudioSegment.from_mp3(audio_path)
            
        self.audiobackup = self.audio
        self.min_silence_len = 500
        self.silence_thresh = -30
        self.keep_silence = True
        self.audios = split_on_silence(self.audio, min_silence_len=self.min_silence_len, 
                                       silence_thresh=self.silence_thresh, 
                                       keep_silence=self.keep_silence)

# if __name__=="__main__":
#     processor = AudioProcessor(audio_path)
    