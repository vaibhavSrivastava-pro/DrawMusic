import matplotlib
import pyaudio
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# Parameters for audio stream
CHUNK = 512       # Number of audio samples per frame
RATE = 44100       # Sampling rate in Hz
FORMAT = pyaudio.paInt16  # Audio format (16-bit signed integer)
CHANNELS = 1       # Number of audio channels (1 for mono)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def get_microphone_amplitude():
    try:
        # Read raw data from microphone
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Convert byte data to numpy array of int16
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Normalize amplitude to range [0, 1]
        normalized_amplitude = np.abs(audio_data) / 32768.0  # 32768 is the max amplitude for int16
        return normalized_amplitude
    except IOError as e:
        print(f"Error reading audio data: {e}")
        return np.zeros(CHUNK)  # Return an array of zeros on error

# Prepare real-time plotting
plt.ion()  # Interactive mode on
fig, ax = plt.subplots()
x_data = np.arange(0, CHUNK)  # X-axis for one chunk of samples
line, = ax.plot(x_data, np.zeros(CHUNK))
ax.set_ylim(0, 1)  # Set y-axis limits for normalized values

# Real-time data collection and plotting
while True:
    try:
        amplitude_data = get_microphone_amplitude()
        
        # Update the plot data
        print(amplitude_data)
        line.set_ydata(amplitude_data)
        ax.relim()    # Adjust the axes to display the new data
        ax.autoscale_view()

        plt.draw()
        plt.pause(0.01)
    except KeyboardInterrupt:
        print("*** Ctrl+C pressed, exiting")
        break
    

# Close the stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()


    
