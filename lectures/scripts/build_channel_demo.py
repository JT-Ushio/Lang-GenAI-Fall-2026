"""Generate PCM16 channel examples using only NumPy and the standard library."""
from pathlib import Path
import wave
import numpy as np

root = Path(__file__).resolve().parents[1]
rate = 48000
# Same tone: left only, equal left/right, right only, then smooth left-to-right pan.
t = np.arange(rate) / rate
beep = .2 * np.sin(2*np.pi*440*t) * np.minimum(t/.03, 1) * np.minimum((1-t)/.03, 1)
quiet = np.zeros((rate//2, 2))
left = np.column_stack([beep, np.zeros(rate)])
center = np.column_stack([beep/np.sqrt(2), beep/np.sqrt(2)])
right = left[:, ::-1]
u = np.arange(rate*3)/(rate*3-1)
pan_t = np.arange(rate*3)/rate
carrier = .2*np.sin(2*np.pi*440*pan_t)*np.minimum(pan_t/.03,1)*np.minimum((3-pan_t)/.03,1)
pan = carrier[:,None]*np.column_stack([np.cos(u*np.pi/2),np.sin(u*np.pi/2)])
stereo = np.vstack([left,quiet,center,quiet,right,quiet,pan])
for name, values in [('channels-stereo.wav',stereo),('channels-mono.wav',stereo.mean(axis=1,keepdims=True))]:
    with wave.open(str(root/'videos/lecture_02'/name),'wb') as f:
        f.setparams((values.shape[1],2,rate,0,'NONE','not compressed'))
        f.writeframes(np.rint(values*32767).astype('<i2').tobytes())
print('Created mono/stereo channel demos: 7.5 seconds, 48 kHz, PCM16.')
