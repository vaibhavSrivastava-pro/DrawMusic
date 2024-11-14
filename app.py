import matplotlib
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import aubio
import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Amplitude and Pitch Drawing Game")
clock = pygame.time.Clock()

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

alpha = 0.3
smoothed_amplitude, smoothed_pitch = 0, 0

running = True
while running:
    try:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
        
        # Only draw if pitch confidence is reasonably high
        if smoothed_pitch > 10:  # Thresholds to filter out noise
            x = int(smoothed_amplitude * screen.get_width())
            y = int(screen.get_height() - (smoothed_pitch / 500) * screen.get_height())
            pygame.draw.circle(screen, (255, 0, 0), (x, y), 2)
        
        pygame.display.flip()
        clock.tick(60)

    except KeyboardInterrupt:
        print("Stopping...")
        running = False

# Close the stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()