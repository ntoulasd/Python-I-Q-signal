import numpy as np
import pyaudio
import struct

# Parameters
sampling_rate = 44100  # Sampling rate
duration = 5           # Duration of the audio in seconds

# QAM Demodulation function (reverse process of modulation)
def qam_demodulate(qam_signal, M=16):
    num_symbols = len(qam_signal)
    audio_signal = np.zeros(num_symbols * 2, dtype=np.float32)

    # Define QAM constellation points
    I_vals = np.array([-3, -1, 1, 3])  # For 16-QAM
    Q_vals = np.array([-3, -1, 1, 3])  # For 16-QAM

    for i in range(num_symbols):
        I_symbol = np.real(qam_signal[i])
        Q_symbol = np.imag(qam_signal[i])

        # Find closest constellation points
        I_index = np.argmin(np.abs(I_vals - I_symbol))
        Q_index = np.argmin(np.abs(Q_vals - Q_symbol))

        # Map to original signal range
        audio_signal[2 * i] = I_vals[I_index] / 3.0  # Normalize to original signal range
        audio_signal[2 * i + 1] = Q_vals[Q_index] / 3.0  # Normalize to original signal range

    return audio_signal

# Load the modulated signal from file
input_file = 'modulated_signal.bin'
with open(input_file, 'rb') as f:
    modulated_signal = np.frombuffer(f.read(), dtype=np.complex64)

# Demodulate the signal
demodulated_audio = qam_demodulate(modulated_signal)

# Convert to 16-bit PCM format
demodulated_audio_int16 = np.int16(demodulated_audio * 32768)

# Play the demodulated audio
p = pyaudio.PyAudio()

# Open the output stream to play sound
stream = p.open(format=pyaudio.paInt16, channels=1, rate=sampling_rate, output=True)

# Play the demodulated audio
stream.write(demodulated_audio_int16.tobytes())

# Close the stream
stream.stop_stream()
stream.close()
p.terminate()

print("Playback finished.")
