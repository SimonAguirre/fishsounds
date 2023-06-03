# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:53:40 2023

@author: Simon
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import threading as th
import os


class Player():
    def __init__(self, window):
        self.window = window
        self.blocksize = 1024
        self.current_frame = 0
        self.dtype = np.float32
        self.stop_flag = True
        self.audio_folder = os.getcwd()
        self.audio_list = self.playlistIterator(self.get_music_files(self.audio_folder))
        self.current_audio = self.audio_list.get_item()
        
        if self.current_audio != 'No-Audio':
            self.sig = np.zeros((220500, 1))
            self.sr = 22050
            self.n_samples, self.n_channels = self.sig.shape
        
        print("Player Initialized")
        
    def play_audio(self):
        print(f"Playing {self.current_audio} from {self.current_frame/self.sr}s")
        if not self.stop_flag:
            return
        self.stop_flag = False
        self.thread = self.playerThreadClass(self)
        self.thread.start()
        
    def stop_audio(self):
        self.current_frame = 0
        self.window.timeStampLabel.setText(self.window.format_time(self.current_frame/self.sr))
        self.thread.stop()
        self.stop_flag = True
    
    def pause_audio(self):
        self.current_frame = self.thread.current_frame
        self.window.timeStampLabel.setText(self.window.format_time(self.current_frame/self.sr))
        print(f"paused at {self.current_frame}")
        self.thread.stop()
        self.stop_flag = True
        
    def next_audio(self):
        resume = self.stop_flag
        if self.thread:
            self.stop_audio()
        self.current_audio = self.audio_list.get_next()
        self.update_data()
        if resume:
            self.play_audio()
                    
    def prev_audio(self):
        resume = not self.stop_flag
        if self.thread:
            self.stop_audio()
        self.current_audio = self.audio_list.get_prev()
        self.update_data()
        if resume:
            self.play_audio()
            
    def set_playlist(self, directory, **kwargs):
        self.audio_folder = directory
        self.audio_list = self.playlistIterator(self.get_music_files(self.audio_folder))
        print("debug: here")
        if len(kwargs) > 0:
            self.audio_list.set_index(kwargs['fname'])
        self.current_audio = self.audio_list.get_item()
        print(f"Loading {self.current_audio} and playlist")
        self.sig, self.sr = sf.read(os.path.join(self.audio_folder, self.current_audio), always_2d=True) # sig: 信号, sr: サンプリング周波数
        self.n_samples, self.n_channels = self.sig.shape
        print(f"Sig shape: {self.sig.shape}, Sr shape: {self.sr}")
        # self.window.plot_audio()
        
    
    def update_data(self):
        self.current_audio = self.audio_list.get_item()
        self.sig, self.sr = sf.read(os.path.join(self.audio_folder, self.current_audio), always_2d=True) # sig: 信号, sr: サンプリング周波数
        self.n_samples, self.n_channels = self.sig.shape
        # self.chunk = np.zeros((self.blocksize, self.n_channels))
        
        
    def get_music_files(self, directory):
        music_extensions = [".mp3", ".wav", ".flac", ".m4a"]  # Add more extensions if needed
        music_files = []
        
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                if any(file.lower().endswith(ext) for ext in music_extensions):
                    music_files.append(file)
    
        return music_files

    class playlistIterator():
        def __init__(self, playlist):
            self.index = 0
            self.list = playlist
            
        def get_next(self):
            self.index += 1
            if self.index < len(self.list):
                item = self.list[self.index]
                return item
            else:
                self.index -= 1
                item = self.list[self.index]
                return item
            
        def get_prev(self):
            if self.index == 0:
                item = self.list[self.index]
                return item
            else:
                self.index -= 1
                item = self.list[self.index]
                return item
            
        def get_item(self, **kwargs):
            try:
                if kwargs['index'] >= 0:
                    self.index = kwargs['index']
            except:
                try:
                    return self.list[self.index]
                except:
                    return "No-Audio"
            
        def get_index(self, **kwargs):
            try:
                return self.list.index(kwargs['fname'])
            except:
                return self.index
        
        def set_index(self, fname, **kwargs):
            self.index = self.list.index(fname)
            try:
                kwargs['player'].stop_audio()
                kwargs['player'].update_data()
            except:
                pass
            
            
    class playerThreadClass(th.Thread):
        def __init__(self, player):
            super(player.playerThreadClass, self).__init__()
            self.sig = player.sig
            self.sr  = player.sr
            self.n_samples, self.n_channels = player.sig.shape
            self.blocksize = player.blocksize
            self.current_frame = player.current_frame
            self.window = player.window
        
        def callback (self, indata, outdata, frames, time, status):
            chunksize = min(self.n_samples - self.current_frame, frames)
            outdata[:] *= 0.0
            self.window.timeStampLabel.setText(self.window.format_time(self.current_frame/self.sr))
            
            
            # チャンネルごとの信号処理
            for k in range(self.n_channels): 
                outdata[0:chunksize, k] = self.sig[self.current_frame:self.current_frame + chunksize, k]
            self.window.melspecPlot.update_img(sig=outdata.T[0],sr=self.sr)
            self.window.labelsPlot.update_img(self.window.get_class_id(self.current_frame), 9)
            
            if chunksize < frames:
                self.window.playButton.setChecked(False)
                self.window.stop()
                raise sd.CallbackStop()
            
            self.current_frame += chunksize
        
        def run(self):
            self.stop_flag = False
            self.event = th.Event()
            
            with sd.Stream(
                samplerate=self.sr, 
                blocksize=self.blocksize,
                channels=self.n_channels,
                callback=self.callback, 
                finished_callback=self.stop
            ):
                self.event.wait()
                
        def stop(self):
            self.stop_flag = True
            self.event.set()
        
# if __name__=="__main__":
#     player = Player()
#     player.play_audio()