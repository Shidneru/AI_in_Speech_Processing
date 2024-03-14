#!/usr/bin/env python
# coding: utf-8

# # V. SAI SHRUTHIK(BL.EN.U4AIE21135)

# Making recording ready to use

import librosa
from IPython.display import Audio

y, sr = librosa.load("wavtrial.wav")
Audio(data=y, rate=sr)
librosa.get_duration(y=y, sr=sr)


# In[3]:


librosa.display.waveshow(y,color = 'blue')


# In[4]:


yt, index = librosa.effects.trim(y)


# In[5]:


Audio(data=yt, rate=sr)


# In[6]:


librosa.get_duration(y=yt, sr=sr)


# In[7]:


librosa.display.waveshow(yt,color = 'blue')


# In[8]:


nonmutesections=librosa.effects.split(y,top_db=10)
nonmutesections


# In[9]:


segment = y[11776:19968]
Audio(segment,rate=sr)


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


sections = librosa.effects.split(y, top_db=10)

# Plot the split signal with deb=10
plt.figure(figsize=(12, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.5, color="red")
for i, section in enumerate(sections):
    plt.fill_betweenx([-1, 1], section[0]/sr, section[1]/sr, alpha=0.5, label=f'Section {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Split Signal')
plt.legend()
plt.show()


# In[12]:


sections = librosa.effects.split(y, top_db=3)

# Plot the split signal with topdb = 3
plt.figure(figsize=(12, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.5, color="red")
for i, section in enumerate(sections):
    plt.fill_betweenx([-1, 1], section[0]/sr, section[1]/sr, alpha=0.5, label=f'Section {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Split Signal')
plt.legend()
plt.show()


# In[13]:


N = 1102 # number of samples taken in 50 ms. 1s -> 22050 samples. 50 ms -> 1102 samples
Ek = []  #continuos average energy
for k in range(len(y)-N+1):
    sum = 0
    for i in range(k,N+k):
        sum += y[i]*y[i]
    Ek.append((1/N)*sum)


# In[14]:


import numpy as np
Ek_arr = np.array(Ek)
print(Ek_arr)


# In[15]:


import statistics
Em = []
mean = statistics.mean(Ek_arr)
var = statistics.variance(Ek_arr)
for i in range(len(Ek_arr)):
    Em.append((Ek_arr[i] - mean)/var)
Em_arr = np.array(Em)


# In[16]:


print(Em_arr)


# In[17]:


librosa.display.waveshow(Em_arr,color = 'red')


# In[18]:


Audio(Em_arr,rate = sr)


# In[19]:


zero_crossings = librosa.zero_crossings(Em_arr, pad=False)


# In[20]:

#plot the signal
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.subplot(2, 1, 1)
librosa.display.waveshow(Ek_arr, sr=sr, color="blue")
plt.title('Continuos average energy')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Plot the zero crossings with sampled index
plt.subplot(2, 1, 2)
plt.plot(zero_crossings, color='r')
plt.title('Zero Crossings')
plt.xlabel('Sample Index')
plt.ylabel('Zero Crossing')
plt.tight_layout()
plt.show()


# In[21]:


zero_crossings_Energy = np.where(np.diff(np.sign(Em_arr)))[0]
lobe_maxima_indices = []
lobe_boundaries = []

# loop to follow the zero crossing variables
for i in range(len(zero_crossings_Energy) - 1):
    lobe_start = zero_crossings_Energy[i]
    lobe_end = zero_crossings_Energy[i + 1]
    lobe_max_index = lobe_start + np.argmax(Em_arr[lobe_start:lobe_end])
    lobe_maxima_indices.append(lobe_max_index)
    lobe_boundaries.append((lobe_start, lobe_end))

plt.figure(figsize=(10, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.5, color="blue")

lobe_maxima_times = librosa.samples_to_time(lobe_maxima_indices, sr=sr)
lobe_boundaries_times = librosa.samples_to_time(np.array(lobe_boundaries).flatten(), sr=sr)

plt.scatter(lobe_maxima_times, y[lobe_maxima_indices], color='r', label='Lobe Maxima') #scatter plot for zero crossings

for start, end in zip(lobe_boundaries_times[::2], lobe_boundaries_times[1::2]):
    plt.axvline(x=start, color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=end, color='g', linestyle='--', alpha=0.5)

#plot with boundaries inclusion
plt.title('Audio Signal with Lobe Maxima and Boundaries')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

