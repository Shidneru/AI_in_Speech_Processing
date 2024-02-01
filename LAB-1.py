#!/usr/bin/env python
# coding: utf-8

# # V. SAI SHRUTHIK (BL.EN.U4AIE21135)

# In[1]:


# LOAD AND PLOT

import librosa
import matplotlib.pyplot as plt

y, sr = librosa.load('wavtrial.wav')

plt.figure(figsize=(10,5))
librosa.display.waveshow(y, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform of the Speech Signal")
plt.show()


# In[2]:


#LENGTH AND MAGNITUDE

import numpy as np 

signal_length_seconds = len(y) / sr
print("Length of the signal:", signal_length_seconds, "seconds")

magnitude_range = np.abs(y).max() - np.abs(y).min()
print("Magnitude range of the signal:", magnitude_range)


# In[3]:


# SMALL SEGMENT

import sounddevice as sd  

def play_segment(start_time, end_time):
    segment = y[int(start_time * sr):int(end_time * sr)]
    sd.play(segment, sr)
    sd.wait()
    
play_segment(2,6)


# In[4]:


print("Playing segment 1: 0 to 2 seconds")
play_segment(0, 2)

print("Playing segment 2: 2 to 4 seconds")
play_segment(2, 4)

print("Playing segment 3: 4 to 6 seconds")
play_segment(4, 6)

print("Playing segment 4: 6 to 8 seconds")
play_segment(6, 8)

print("Playing segment 5: 8 to 10 seconds")
play_segment(8, 10)

