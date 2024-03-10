#!/usr/bin/env python
# coding: utf-8

# # V. SAI SHRUTHIK (BL.EN.U4AIE21135)

# In[1]:


import librosa
import librosa.display
import IPython.display as ipd

y, sr = librosa.load("wavtrial.wav")
librosa.display.waveshow(y);


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
FFT_Signal = np.fft.fft(y)
plt.plot(FFT_Signal.real); #Plotting the amplitude of the signal


# In[3]:


IFFT_Signal = np.fft.ifft(FFT_Signal);
plt.plot(IFFT_Signal);


# A2

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

def rectangular_window_filter(signal, cutoff_freq, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)  
    mask = np.abs(freqs) <= cutoff_freq 
    plt.plot(freqs,mask) # rectangular low pass filter
    filtered_signal = np.fft.irfft(np.fft.rfft(signal) * mask, n)  # Apply the mask in frequency domain
    return filtered_signal

# Apply rectangular window filter
cutoff_freq = 1000  # Cutoff frequency
filtered_signal = rectangular_window_filter(y, cutoff_freq, sr)

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(y, label='Original Signal')
plt.plot(filtered_signal, label='Filtered Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Rectangular Window Filter')
plt.legend()
plt.grid()
plt.show()


# In[5]:


plt.plot(np.fft.fft(y).real,label='Original Signal');
plt.plot(np.fft.fft(filtered_signal).real,label='Filtered Signal');
plt.legend();


# Original Audio

# In[6]:


ipd.Audio(y,rate=sr)


# Low Pass Rectangular Window Filtered Audio

# In[7]:


ipd.Audio(filtered_signal,rate=sr)


# Same For Bandpass filter

# In[8]:


import numpy as np
import matplotlib.pyplot as plt

def rectangular_window_filter(signal, lower_freq, higher_freq, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)  # Compute frequency values using dft
    mask = (np.abs(freqs) >= lower_freq) & (np.abs(freqs) <= higher_freq)  # Create a mask for frequencies 
    plt.plot(freqs,mask) #Plotting rectangular band pass filter
    print(mask.shape)
    print(mask)
    filtered_signal = np.fft.irfft(np.fft.rfft(signal) * mask, n)  # Apply the mask in frequency domain
    return filtered_signal

# Apply rectangular window filter
lower_freq = 1000  # Cutoff frequency 
higher_freq = 6000
filtered_signal = rectangular_window_filter(y, lower_freq, higher_freq, sr)

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(y, label='Original Signal')
plt.plot(filtered_signal, label='Filtered Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Rectangular Window Filter')
plt.legend()
plt.grid()
plt.show()


# In[9]:


plt.plot(np.fft.fft(y).real,label='Original Signal');
plt.plot(np.fft.fft(filtered_signal).real,label='Filtered Signal');
plt.legend();


# In[10]:


ipd.Audio(y,rate=sr)


# In[11]:


ipd.Audio(filtered_signal,rate=sr)


# Highpass filter

# In[12]:


import numpy as np
import matplotlib.pyplot as plt

def rectangular_window_filter(signal, cutoff_freq, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)  # Compute frequency values using dft
    mask = np.abs(freqs) >= cutoff_freq  # Create a mask for frequencies 
    plt.plot(freqs,mask) #Plotting rectangular high pass filter 
    filtered_signal = np.fft.irfft(np.fft.rfft(signal) * mask, n)  # Apply the mask in frequency domain
    return filtered_signal

# Apply rectangular window filter
cutoff_freq = 3000  # Cutoff frequency 
filtered_signal = rectangular_window_filter(y, cutoff_freq, sr)

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(y, label='Original Signal')
plt.plot(filtered_signal, label='Filtered Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Rectangular Window Filter')
plt.legend()
plt.grid()
plt.show()


# In[13]:


plt.plot(np.fft.fft(y).real,label='Original Signal');
plt.plot(np.fft.fft(filtered_signal).real,label='Filtered Signal');
plt.legend();


# In[14]:


ipd.Audio(y,rate=sr)


# In[15]:


ipd.Audio(filtered_signal,rate=sr)


# A3

# Cosine filters

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cosine

# Generate a cosine filter
cutoff_freq = 1000  # Cutoff frequency 
window_size = 4096  # Window size for the filter
cosine_filter = cosine(window_size, np.pi * cutoff_freq)

# Plot the cosine filter
plt.figure(figsize=(10, 6))
plt.plot(cosine_filter)
plt.title('Cosine Filter')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid()
plt.show()


# In[17]:


from scipy.signal import gaussian

# Generate a Gaussian filter
std_dev = 500  # Standard deviation of the Gaussian kernel
gaussian_filter = gaussian(window_size, std=std_dev)

# Plot the Gaussian filter
plt.figure(figsize=(10, 6))
plt.plot(gaussian_filter)
plt.title('Gaussian Filter')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid()
plt.show()


# In[18]:


import librosa
from scipy.signal import convolve

y_filtered_cosine = convolve(y, cosine_filter, mode='same')
y_filtered_gaussian = convolve(y, gaussian_filter, mode='same')
plt.plot(y,label="Original")
plt.plot(y_filtered_gaussian, label='Gaussian Filtered')
plt.plot(y_filtered_cosine, label='Cosine Filtered')
plt.legend()


# In[19]:


ipd.Audio(y,rate=sr)


# In[20]:


ipd.Audio(y_filtered_cosine,rate=sr)


# In[21]:


ipd.Audio(y_filtered_gaussian,rate=sr)

