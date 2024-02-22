#!/usr/bin/env python
# coding: utf-8

# # V.SAI SHRUTHIK(BL.EN.U4AIE21135)

# In[2]:


# import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import spectrogram

# Load the speech signal
speech_file = "wavtrial.wav"
speech_signal, your_sampling_rate = librosa.load(speech_file)

# A1. Spectral Analysis
fft_result = np.fft.fft(speech_signal)
freq_axis = np.fft.fftfreq(len(speech_signal), d=1/your_sampling_rate)
plt.plot(freq_axis[:len(freq_axis)//2], np.abs(fft_result[:len(freq_axis)//2]))
plt.title('Amplitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

# A2. Inverse Transform
inv_fft_result = np.fft.ifft(fft_result)
plt.plot(np.real(inv_fft_result))  # Take the real part for visualization
plt.title('Inverse FFT Result')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# A3. Spectral Analysis of a Word
word_start = int(2 * your_sampling_rate)  # Example: starting at 2 seconds
word_end = int(3 * your_sampling_rate)  # Example: ending at 3 seconds
word_signal = speech_signal[word_start:word_end]
word_fft_result = np.fft.fft(word_signal)
freq_axis_word = np.fft.fftfreq(len(word_signal), d=1/your_sampling_rate)
plt.plot(freq_axis_word[:len(freq_axis_word)//2], np.abs(word_fft_result[:len(freq_axis_word)//2]))
plt.title('Amplitude Spectrum of Word')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()


# A4. Spectral Analysis with Rectangular Window
window_size = int(0.02 * your_sampling_rate)  # 20 milliseconds
rectangular_window = np.ones(window_size)
rectangular_fft_result = np.fft.fft(speech_signal[:window_size] * rectangular_window)
freq_axis_rectangular = np.fft.fftfreq(window_size, d=1/your_sampling_rate)
plt.plot(freq_axis_rectangular[:len(freq_axis_rectangular)//2], np.abs(rectangular_fft_result[:len(freq_axis_rectangular)//2]))
plt.title('Amplitude Spectrum with Rectangular Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()


# A5. Break into Windows and Create Heatmap
window_length = int(0.02 * your_sampling_rate)  # 20 milliseconds
hop_length = int(0.01 * your_sampling_rate)  # 10 milliseconds
num_windows = int(len(speech_signal) / hop_length) - 1
spec_matrix = np.zeros((window_length // 2 + 1, num_windows))

for i in range(num_windows):
    window_data = speech_signal[i * hop_length : i * hop_length + window_length]
    spec_matrix[:, i] = np.abs(np.fft.rfft(window_data))

plt.imshow(np.log(spec_matrix), cmap='viridis', aspect='auto', origin='lower')
plt.title('Spectrogram Heatmap')
plt.xlabel('Time')
plt.ylabel('Frequency Bin')
plt.colorbar(format='%+2.0f dB')
plt.show()

# A6. Spectrogram using scipy
frequencies, times, spectrogram_result = spectrogram(speech_signal, fs=your_sampling_rate, nperseg=window_length, noverlap=hop_length)
plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_result), shading='auto')
plt.title('Spectrogram using scipy')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(label='Intensity (dB)')
plt.show()

