# V.SAI SHRUTHIK [BL.EN.U4AIE21135]
#Lab-6


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt


# For Vowel sound 1

# In[2]:


vowel_a="vowels.ogg"
ipd.Audio(vowel_a)


# In[3]:


time_series_vowel_a, sample_rate_vowel_a = librosa.load(vowel_a)
librosa.display.waveshow(time_series_vowel_a)


# In[4]:


## Applying fft to the signal
fft_vowel_a=np.fft.fft(time_series_vowel_a, n=None, axis=-1, norm=None)
print(fft_vowel_a)


# In[5]:


amp_vowel_a=np.abs(fft_vowel_a)
plt.plot(amp_vowel_a)
plt.title('FFT Signal')
plt.show()


# For vowel sound 2

# In[6]:


vowel_i="vowels.ogg"
ipd.Audio(vowel_i)


# In[7]:


time_series_vowel_i, sample_rate_vowel_i = librosa.load(vowel_i)
librosa.display.waveshow(time_series_vowel_i)


# In[8]:


## Applying fft to the signal
fft_vowel_i=np.fft.fft(time_series_vowel_i, n=None, axis=-1, norm=None)
print(fft_vowel_i)


# In[9]:


amp_vowel_i=np.abs(fft_vowel_i)
plt.plot(amp_vowel_i)
plt.title('FFT Signal')
plt.show()


# **A2.**

# In[10]:


consonant_h="consonants.ogg"
ipd.Audio(consonant_h)


# In[11]:


time_signal_consonant_h, sample_rate_consonant_h = librosa.load(consonant_h)
librosa.display.waveshow(time_signal_consonant_h)


# In[12]:


## Applying fft to the signal
fft_consonant_h=np.fft.fft(time_signal_consonant_h, n=None, axis=-1, norm=None)
print(fft_consonant_h)


# In[13]:


amplitude_consonant_h = np.abs(fft_consonant_h)
plt.plot(amplitude_consonant_h)
plt.title('FFT Signal of consonant h')
plt.show()


# For 2nd consonant

# In[14]:


consonant_r="consonants.ogg"
ipd.Audio(consonant_r)


# In[15]:


time_signal_consonant_r, sample_rate_consonant_r = librosa.load(consonant_r)
librosa.display.waveshow(time_signal_consonant_r)


# In[16]:


## Applying fft to the signal
fft_consonant_r=np.fft.fft(time_signal_consonant_r, n=None, axis=-1, norm=None)
print(fft_consonant_r)


# In[17]:


amplitude_consonant_r=np.abs(fft_consonant_r)
plt.plot(amplitude_consonant_r)
plt.title('FFT Signal of consonat r')
plt.show()


# **A3.**

# In[18]:


silence="silence.ogg"
ipd.Audio(silence)


# In[19]:


time_signal_silence, sample_rate_silence = librosa.load(silence)
librosa.display.waveshow(time_signal_silence)


# In[20]:


## Applying fft to the signal
fft_silence=np.fft.fft(time_signal_silence, n=None, axis=-1, norm=None)
print(fft_silence)


# In[21]:


amplitude_silence_signal=np.abs(fft_silence)
plt.plot(amplitude_silence_signal)
plt.title('FFT Signal of silence signal')
plt.show()


# **For non-voice signal**

# In[22]:


nonvoice='consonants.ogg'
ipd.Audio(nonvoice)
time_signal_nonvoice, sample_rate_nonvoice = librosa.load(nonvoice)
librosa.display.waveshow(time_signal_nonvoice)

## Applying fft to the signal
fft_nonvoice=np.fft.fft(time_signal_nonvoice, n=None, axis=-1, norm=None)
print(fft_nonvoice)
amplitude_nonvoice_signal=np.abs(fft_nonvoice)
plt.plot(amplitude_nonvoice_signal)
plt.title('FFT Signal of non-voice signal')
plt.show()


# **A4.**

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

signal = 'wavtrial.wav'
time_signal, sample_rate = librosa.load(signal)

# Compute the STFT
n_fft = 2048  # FFT window size
hop_length = 512  # Hop length for STFT
spectrogram = np.abs(librosa.stft(time_signal, n_fft=n_fft, hop_length=hop_length))

# Plot the spectrogram
plt.figure(figsize=(12, 6))
librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of the signal')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()




