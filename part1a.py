import numpy as np
import matplotlib.pyplot as plt
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

def bode_plot(system, fs=1.0, cutoff_freq=None, title='Bode Plot', xlabel='Frequency (Hz)'):
    frequencies, response = freqz(system, fs=fs)

    # Plot magnitude response
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies, 20 * np.log10(np.abs(response)))
    plt.title(title)
    plt.ylabel('Gain (dB)')
    plt.grid(True)

    # Plot phase response
    plt.subplot(2, 1, 2)
    plt.semilogx(frequencies, np.angle(response))
    plt.xlabel(xlabel)
    plt.ylabel('Phase (radians)')
    plt.grid(True)

    if cutoff_freq is not None:
        for freq in cutoff_freq:
            plt.subplot(2, 1, 1)
            plt.axvline(x=freq, color='r', linestyle='--', label=f'Cutoff Frequency: {freq} Hz')
            plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage:
beta = 0.5
order = 80
Fs = 8000  # 8 KHz sampling frequency

#Low-pass filter design

cutoff_freq_lowpass = 20  # 1 KHz cutoff frequency
h_lowpass = fir_filter_design(cutoff_freq_lowpass, 'lowpass', beta, order, Fs)
bode_plot(h_lowpass, Fs, cutoff_freq=[cutoff_freq_lowpass], title='Bode Plot of Low-Pass Filter', xlabel='Frequency (Hz)')


# #High-pass filter design

cutoff_freq_highpass = 2000  # 1 KHz cutoff frequency
h_highpass = fir_filter_design(cutoff_freq_highpass, 'highpass', beta, order, Fs)
bode_plot(h_highpass, Fs, cutoff_freq=[cutoff_freq_highpass], title='Bode Plot of High-pass Filter', xlabel='Frequency (Hz)')

# Band-pass filter design
cutoff_freq_bandpass = [200, 2000]  # 500 Hz to 1.5 KHz passband
h_bandpass = -fir_filter_design(200, 'lowpass', beta, order, Fs) + fir_filter_design(2000, 'lowpass', beta, order, Fs)
bode_plot(h_bandpass, Fs, cutoff_freq=cutoff_freq_bandpass, title='Bode Plot of Band-Pass Filter using low', xlabel='Frequency (Hz)')