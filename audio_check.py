import wave
import numpy as np

file_path = "dataset/farhana_faruk/background/background_01.wav"

with wave.open(file_path, "rb") as audio:
    channels = audio.getnchannels()
    sample_rate = audio.getframerate()
    frames = audio.getnframes()

    raw_data = audio.readframes(frames)

audio_data = np.frombuffer(raw_data, dtype=np.int16)
audio_data = audio_data.reshape(-1, channels)

print("Audio file:", file_path)
print("Channels:", channels)
print("Sample rate:", sample_rate)
print("Duration:", frames / sample_rate, "seconds")
print("Audio shape:", audio_data.shape)
print("Minimum amplitude:", audio_data.min())
print("Maximum amplitude:", audio_data.max())
print("\nChannel analysis:")

for channel in range(channels):
    channel_data = audio_data[:, channel]

    rms = np.sqrt(np.mean(channel_data.astype(np.float64) ** 2))
    peak = np.max(np.abs(channel_data.astype(np.int32)))

    print(
        f"Channel {channel + 1}: "
        f"RMS = {rms:.2f}, "
        f"Peak = {peak}"
    )
print("\nClipping analysis:")

for channel in range(channels):
    channel_data = audio_data[:, channel].astype(np.int32)

    clipped = np.sum(np.abs(channel_data) >= 32760)
    percentage = (clipped / len(channel_data)) * 100

    print(
        f"Channel {channel + 1}: "
        f"Clipped samples = {clipped}, "
        f"Percentage = {percentage:.4f}%"
    )