
import soundfile as sf
from pathlib import Path

SOURCE = Path("dataset_new/farhana_faruk")
DESTINATION = Path("dataset_mono_test/farhana_faruk")

CLASSES = ["clap", "whistle", "hey", "background"]

CHANNEL_INDEX = 2

for class_name in CLASSES:
    source_folder = SOURCE / class_name
    destination_folder = DESTINATION / class_name
    destination_folder.mkdir(parents=True, exist_ok=True)

    for audio_file in sorted(source_folder.glob("*.wav")):
        audio, sample_rate = sf.read(audio_file)
        mono_audio = audio[:, CHANNEL_INDEX]

        output_file = destination_folder / audio_file.name
        sf.write(output_file, mono_audio, sample_rate)

        print(f"Created: {output_file}")

print("\nAll mono recordings created!")
