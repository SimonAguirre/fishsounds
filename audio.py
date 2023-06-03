import sounddevice as sd
import soundfile as sf
import threading

class Player(threading.Thread):
    def __init__(self, sig, sr, blocksize = 1024):
        super(Player, self).__init__()
        
        self.sig = sig
        self.sr  = sr
        self.n_samples, self.n_channels = sig.shape
        self.blocksize = blocksize

    def callback(self, indata, outdata, frames, time, status):
        chunksize = min(self.n_samples - self.current_frame, frames)
    
        outdata[:] *= 0.0
        # チャンネルごとの信号処理
        for k in range(self.n_channels): 
            outdata[0:chunksize, k] = self.sig[self.current_frame:self.current_frame + chunksize, k]

        if chunksize < frames:
            raise sd.CallbackStop()
        
        self.current_frame += chunksize

    def run(self):
        self.current_frame = 0
        self.event = threading.Event()
        
        with sd.Stream(
            samplerate=self.sr, 
            blocksize=self.blocksize,
            channels=self.n_channels,
            callback=self.callback, 
            finished_callback=self.event.set
        ):
            self.event.wait()


filepath = "./audio.wav"
sig, sr = sf.read(filepath, always_2d=True)

player = Player(sig, sr)
player.start()

print("Start")

import code
console = code.InteractiveConsole(locals=locals())
console.interact()