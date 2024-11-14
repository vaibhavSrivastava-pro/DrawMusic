import matplotlib
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import aubio

matplotlib.use("TkAgg")

# Parameters for audio stream
BUFFER_SIZE = 1024
SAMPLERATE = 44100
FORMAT = pyaudio.paFloat32
CHANNELS = 1

# Initialize PyAudio and open stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLERATE, input=True, frames_per_buffer=BUFFER_SIZE)

# Initialize aubio pitch detection
tolerance = 0.8
win_s = 4096  # FFT size
hop_s = BUFFER_SIZE  # Hop size
pitch_o = aubio.pitch("default", win_s, hop_s, SAMPLERATE)
pitch_o.set_unit("Hz")  # Set unit to Hz for pitch frequency
pitch_o.set_tolerance(tolerance)

# Set up real-time plotting
plt.ion()  # Interactive mode on
fig, ax = plt.subplots()
ax.set_xlim(0.35, 1)  # Amplitude range [0, 1]
ax.set_ylim(0, 500)  # Pitch range in Hz (adjust as needed)
ax.set_xlabel("Amplitude")
ax.set_ylabel("Pitch (Hz)")
scat = ax.scatter([], [])

alpha = 0.3
smoothed_amplitude, smoothed_pitch = 0, 0

# Real-time plotting loop
while True:
    try:
        # Get amplitude and pitch data
        audiobuffer = stream.read(BUFFER_SIZE, exception_on_overflow=False)
        signal = np.frombuffer(audiobuffer, dtype=np.float32)
        
        # Calculate amplitude
        amplitude = np.abs(signal).mean()  # Take the average absolute amplitude
        amplitude_normalized = amplitude / np.max(signal) if np.max(signal) != 0 else 0
        
        # Calculate pitch
        pitch = pitch_o(signal)[0]
        confidence = pitch_o.get_confidence()
        
        smoothed_amplitude = alpha * amplitude_normalized + (1 - alpha) * smoothed_amplitude
        smoothed_pitch = alpha * pitch + (1 - alpha) * smoothed_pitch
        
        # print(amplitude_normalized, pitch)
        
        # Only plot if pitch confidence is reasonably high
        if smoothed_pitch > 10:  # Thresholds to filter out noise
            scat.set_offsets([[smoothed_amplitude, smoothed_pitch]])
            
            # Refresh plot
            plt.draw()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("Stopping...")
        break

# Close the stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()
plt.show(block=True)  # Keeps plot open after stopping
