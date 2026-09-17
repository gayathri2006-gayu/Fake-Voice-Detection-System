import os
import time
import random
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

FAKE_DIR = "dataset/fake"
SAMPLE_RATE = 16000
RECORD_DURATION = 3  # seconds to record while the file plays

NUM_TO_PROCESS = 100  # how many existing fake files to re-record via mic


def get_next_index(folder, prefix):
    existing = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".wav")]
    max_num = 0
    for f in existing:
        try:
            num = int(f.replace(prefix, "").replace(".wav", ""))
            max_num = max(max_num, num)
        except ValueError:
            continue
    return max_num + 1


def play_and_record(source_path, out_path):
    """Plays source_path through the speaker while simultaneously recording from mic."""
    # Load the source audio to play
    data, sr = sf.read(source_path, dtype="float32")

    # Start recording (non-blocking) for RECORD_DURATION seconds
    recording = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)

    # Immediately start playback of the source audio (this plays through speakers)
    sd.play(data, sr)

    # Wait for the recording duration to finish
    sd.wait()

    # Stop playback in case it's still going
    sd.stop()

    write(out_path, SAMPLE_RATE, recording)


def main():
    existing_fake_files = [f for f in os.listdir(FAKE_DIR) if f.endswith(".wav") and not f.startswith("mic_")]

    if not existing_fake_files:
        print("No existing fake files found to re-record!")
        return

    random.shuffle(existing_fake_files)
    files_to_process = existing_fake_files[:NUM_TO_PROCESS]

    start_idx = get_next_index(FAKE_DIR, "mic_fake_")

    print(f"Found {len(existing_fake_files)} existing fake files.")
    print(f"Will re-record {len(files_to_process)} of them through speaker+mic.")
    print("\nIMPORTANT: Make sure your speaker volume is turned up,")
    print("and your mic can hear the speaker clearly.")
    print("Keep the room as quiet as possible (besides the TTS audio).\n")

    input("Press Enter to start the automated re-recording process...")

    for i, filename in enumerate(files_to_process):
        idx = start_idx + i
        source_path = os.path.join(FAKE_DIR, filename)
        out_filename = f"mic_fake_{idx}.wav"
        out_path = os.path.join(FAKE_DIR, out_filename)

        print(f"[{i+1}/{len(files_to_process)}] Playing {filename} -> recording as {out_filename}")

        try:
            play_and_record(source_path, out_path)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        time.sleep(0.5)  # short pause between recordings

    print("\nDone! Mic-recorded fake samples created.")
    total_fake = len([f for f in os.listdir(FAKE_DIR) if f.endswith(".wav")])
    print(f"Total fake files now: {total_fake}")


if __name__ == "__main__":
    main()