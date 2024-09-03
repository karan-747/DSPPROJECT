from google.colab import drive

drive.mount('/content/gdrive')

# Step 2: Upload the audio file to Colab
from google.colab import files

# Upload the "music.wav" file


import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
from scipy.signal import freqz #for bode plot generation only
def kaiser_window(beta, M):
    # Function to generate Kaiser window coefficients
    n = np.arange(0, M + 1)
    w = np.i0(beta * np.sqrt(1 - ((n - M / 2) / (M / 2))**2)) / np.i0(beta)
    return w

def fir_filter_design(cutoff_freq, filter_type, beta, order, Fs):
    # Function to design FIR filters using Kaiser window
    M = order

    #Normalize cutoff frequencies
    if isinstance(cutoff_freq, list):
        cutoff_freq = [f / (Fs / 2) for f in cutoff_freq]
    else:
        cutoff_freq = cutoff_freq / (Fs / 2)

    # Design filter coefficients
    if filter_type == 'lowpass':
        h = cutoff_freq*np.sinc(cutoff_freq * (np.arange(M + 1) - M / 2))

    elif filter_type == 'highpass':
        h = np.sinc(np.arange(M + 1) - M / 2) - cutoff_freq * np.sinc(cutoff_freq * (np.arange(M + 1) - M / 2))

    elif filter_type == 'bandpass':
        h = cutoff_freq[1] * (np.sinc( cutoff_freq[1] * (np.arange(M + 1) - M / 2)) - cutoff_freq[0] * np.sinc(cutoff_freq[0] * (np.arange(M + 1) - M / 2)))

    # Apply Kaiser window
    w = kaiser_window(beta, M)
    h = h * w

    return h

def apply_filter(input_signal, filter_coeffs):
    # Function to apply filter to input signal using convolution
    output_signal = np.convolve(input_signal, filter_coeffs, mode='full')
    return output_signal

# Read audio file
file_path = 'music.wav'
wave_file = wave.open(file_path, 'rb')
signal = np.frombuffer(wave_file.readframes(-1), dtype=np.int16)
wave_file.close()
fs, data = wavfile.read('/content/gdrive/My Drive/music.wav')
# Filter parameters
beta = 0.5  # Adjust the Kaiser window parameter
order = 80
Fs = fs  # Adjust the sampling frequency according to your audio file

Gain1 = 0.99
Gain2 = 0.01
Gain3 = 0.01
#to avoid clipping
signal = signal/3

# High-pass filter design
filter_coeffs_band1 = fir_filter_design(20, 'highpass', beta, order, fs) - fir_filter_design(200, 'highpass', beta, order, fs)
filter_coeffs_band2 = fir_filter_design(200, 'highpass', beta, order, fs) - fir_filter_design(2000, 'highpass', beta, order, fs)
filter_coeffs_band3 =  fir_filter_design(2000, 'highpass', beta, order, fs) - fir_filter_design(20000, 'highpass', beta, order, fs)

# Apply high-pass filter to the audio signal
filtered_signal = apply_filter(signal*Gain1, filter_coeffs_band1) +  apply_filter(signal*Gain2, filter_coeffs_band2) + apply_filter(signal*Gain3, filter_coeffs_band3)


wavfile.write('equalized_music.wav', fs, filtered_signal.astype(np.int16))



import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

freq_response_band1 = fft(filter_coeffs_band1, n=2048)
freq_response_band2 = fft(filter_coeffs_band2, n=2048)
freq_response_band3 = fft(filter_coeffs_band3, n=2048)
freq_axis = fftfreq(2048, d=1/fs)
plt.figure(figsize=(12, 6))
plt.plot(freq_axis, np.abs(freq_response_band1), label='20 - 200 Hz Band')
plt.plot(freq_axis, np.abs(freq_response_band2), label='200 - 2KHz Band')
plt.plot(freq_axis, np.abs(freq_response_band3), label='2KHz - 20KHz Band')
plt.title('Equalizer Bands Magnitude Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.show()

freq_response_band1 = fft(filter_coeffs_band1, n=2048)
freq_response_band2 = fft(filter_coeffs_band2, n=2048)
freq_response_band3 = fft(filter_coeffs_band3, n=2048)
freq_axis = fftfreq(2048, d=1/fs)
plt.figure(figsize=(12, 6))
plt.plot(freq_axis, np.abs(freq_response_band1), label='20 - 200 Hz Band', color='b')
plt.title('20 - 200 Hz Band Magnitude Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.axvline(x=20, color='r', linestyle='--', label='20 Hz')  # Add vertical line at 20 Hz
plt.axvline(x=200, color='g', linestyle='--', label='200 Hz')  # Add vertical line at 200 Hz
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(freq_axis, np.abs(freq_response_band2), label='200 - 2000 Hz Band', color='b')
plt.title('200 - 2000 Hz Band Magnitude Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.axvline(x=200, color='r', linestyle='--', label='200 Hz')  # Add vertical line at 200 Hz
plt.axvline(x=2000, color='g', linestyle='--', label='2000 Hz')  # Add vertical line at 2000 Hz
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(freq_axis, np.abs(freq_response_band3), label='2000 - 20000 Hz Band', color='b')
plt.title('2000 - 20000 Hz Band Magnitude Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.axvline(x=2000, color='r', linestyle='--', label='2000 Hz')  # Add vertical line at 2000 Hz
plt.axvline(x=20000, color='g', linestyle='--', label='20000 Hz')  # Add vertical line at 20000 Hz
plt.legend()
plt.show()




#Plot 2: Input Signal Spectrum and Filtered Signal Spectrum
plt.figure(figsize=(12, 8))

# Subplot 1: Input Signal Spectrum
plt.subplot(2, 1, 1)
plt.magnitude_spectrum(signal, Fs=fs, scale='dB', color='b', label='Input Signal')
plt.title('Input Signal Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()

# Subplot 2: Filtered Signal Spectrum
plt.subplot(2, 1, 2)
plt.magnitude_spectrum(filtered_signal, Fs=fs, scale='dB', color='r', label='Filtered Signal')
plt.title('Filtered Signal Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()

plt.tight_layout()
plt.show()

