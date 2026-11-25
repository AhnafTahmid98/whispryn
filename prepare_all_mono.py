import soundfile as sf
from pathlib import Path

PEOPLE = ["farhana_faruk", "ahnaf_ahmed"]
CLASSES = ["clap", "whistle", "hey", "background"]

SOURCE_ROOT = Path("dataset_new")
DESTINATION_ROOT = Path("dataset_mono_test")
CHANNEL_INDEX = 2

for person in PEOPLE:
    for class_name in CLASSES:
        source_folder = SOURCE_ROOT / person / class_name
        destination_folder = DESTINATION_ROOT / person / class_name
        destination_folder.mkdir(parents=True, exist_ok=True)

        for audio_file in sorted(source_folder.glob("*.wav")):
            audio, sample_rate = sf.read(audio_file)

            if audio.ndim != 2 or audio.shape[1] <= CHANNEL_INDEX:
                print(f"Skipped (missing channel): {audio_file}")
                continue

            mono_audio = audio[:, CHANNEL_INDEX]
            output_file = destination_folder / audio_file.name
            sf.write(output_file, mono_audio, sample_rate)

            print(f"Created: {output_file}")

print("\nAll mono recordings created!")