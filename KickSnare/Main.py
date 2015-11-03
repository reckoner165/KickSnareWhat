__author__ = 'Srinivasan'

import librosa

import matplotlib.pyplot as plt
import numpy as np

path = '/Users/Srinivasan/Music/Samples/MafzDrumkitVS1/Kick/'
filename = 'K - Air.wav'
input = path + filename

y, sr = librosa.load(input)

#s = np.abs(librosa.stft(y))
pow = librosa.core.stft(y, n_fft=len(y))


print(pow)

librosa.display.specshow(librosa.logamplitude(np.abs(pow)**2,ref_power=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig('Plot.jpg')