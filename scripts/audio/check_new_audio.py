
import numpy as np
import soundfile as sf
from pathlib import Path

DATASET = Path("dataset_new/farhana_faruk")

for class_name in ["clap", "whistle", "hey", "background"]:
    print(f"\n===== {class_name.upper()} =====")

    folder = DATASET / class_name

    for audio_file in sorted(folder.glob("*.wav")):
        audio, sample_rate = sf.read(audio_file, dtype="int16")

        print(f"\nFile: {audio_file.name}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Channels: {audio.shape[1] if audio.ndim > 1 else 1}")
        print(f"Duration: {len(audio) / sample_rate:.2f} seconds")

        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        for channel in range(audio.shape[1]):
            samples = audio[:, channel].astype(np.int32)

            rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
            peak = np.max(np.abs(samples))
            clipped = np.sum(np.abs(samples) >= 32760)
            percentage = clipped / len(samples) * 100

            print(
                f"Channel {channel + 1}: "
                f"RMS={rms:.2f}, "
                f"Peak={peak}, "
                f"Clipped={percentage:.4f}%"
            )
