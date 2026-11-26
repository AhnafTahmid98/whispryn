import wave
import numpy as np
from pathlib import Path

# Dataset location
DATASET_DIR = Path("dataset/farhana_faruk")

# Select Channel 5 (Python index 4)
SELECTED_CHANNEL = 4

# Output folder
OUTPUT_DIR = Path("processed_audio")
def convert_to_mono(input_file, output_file):

    with wave.open(str(input_file), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        frames = audio.getnframes()
        raw_data = audio.readframes(frames)

    audio_data = np.frombuffer(raw_data, dtype=np.int16)
    audio_data = audio_data.reshape(-1, channels)

    # Select Channel 5
    mono_data = audio_data[:, SELECTED_CHANNEL]

    # Save as a mono WAV file
    with wave.open(str(output_file), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(mono_data.tobytes())

    print(f"Processed: {input_file.name}")
    # Create output folder
OUTPUT_DIR.mkdir(exist_ok=True)

# Sound classes
CLASSES = ["clap", "whistle", "hey", "background"]

for sound_class in CLASSES:

    input_folder = DATASET_DIR / sound_class
    output_folder = OUTPUT_DIR / sound_class

    output_folder.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_folder.glob("*.wav"))

    print(f"\nProcessing {sound_class}: {len(wav_files)} files")

    for input_file in wav_files:

        output_file = output_folder / input_file.name

        convert_to_mono(input_file, output_file)

print("\nAll audio files processed!")
