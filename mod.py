import numpy as np
import pyaudio
import wave
import struct

# Parameters
M = 16  # Use 16-QAM for modulation
sampling_rate = 44100  # Standard audio sampling rate
duration = 5  # Duration of the audio in seconds
num_samples = duration * sampling_rate

# Microphone Recording
p = pyaudio.PyAudio()

# Print all available input devices
print("Available Input Devices:")
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    print(f"{i}: {device_info['name']}")

# Ask the user to select the device index
device_index = int(input("Enter the device index for the microphone you'd like to use: "))

# Open the microphone stream for recording
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=sampling_rate,
                input=True,
                input_device_index=device_index,  # Select the desired device index
                frames_per_buffer=1024)

print(f"Recording from device index {device_index}...")


# Open the microphone stream for recording
#stream = p.open(format=pyaudio.paInt16, channels=1, rate=sampling_rate, input=True, frames_per_buffer=1024)

print("Recording audio...")

# Read data from the microphone
frames = []
for _ in range(0, int(sampling_rate / 1024 * duration)):
    data = stream.read(1024)
    frames.append(data)

print("Recording finished.")

# Close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Convert frames to numpy array
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

# Normalize the audio to a range from -1 to 1
audio_data_normalized = audio_data / 32768.0

# QAM Modulation
def qam_modulate(signal, M=16):
    # Create QAM symbols for M-ary QAM
    num_symbols = len(signal) // 2
    qam_signal = np.zeros(num_symbols, dtype=complex)

    # Map signal to QAM symbols
    for i in range(num_symbols):
        I = signal[2 * i]  # In-phase component
        Q = signal[2 * i + 1]  # Quadrature component

        # Normalize to QAM constellation points
        I_vals = np.array([-3, -1, 1, 3])  # For 16-QAM
        Q_vals = np.array([-3, -1, 1, 3])  # For 16-QAM

        I_symbol = I_vals[int((I + 1) // 2)]  # Convert to symbol
        Q_symbol = Q_vals[int((Q + 1) // 2)]  # Convert to symbol

        qam_signal[i] = I_symbol + 1j * Q_symbol

    return qam_signal

# Modulate the normalized audio signal
modulated_signal = qam_modulate(audio_data_normalized)

# Save modulated signal to file (you can save it as a binary file)
output_file = 'modulated_signal.bin'
with open(output_file, 'wb') as f:
    # Save as raw bytes
    f.write(modulated_signal.tobytes())

print(f"Modulated signal saved to {output_file}")
