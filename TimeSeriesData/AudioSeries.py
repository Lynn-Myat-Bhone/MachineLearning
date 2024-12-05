import librosa as lr
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
# List all the wav files in the folder
audio_files = glob("your data dir" + '/*.wav')

# Read in the first audio file, create the time array
audio, sfreq = lr.load(audio_files[0])
time = np.arange(0, len(audio)) / sfreq

# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()