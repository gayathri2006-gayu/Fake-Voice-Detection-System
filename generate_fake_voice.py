
import os
import re
import asyncio
import random
import edge_tts
from pydub import AudioSegment

print("DATASET FIX STARTED")

fake_folder = "dataset/fake"
os.makedirs(fake_folder, exist_ok=True)

voices = [
    "en-US-AriaNeural",
    "en-GB-RyanNeural",
    "en-IN-NeerjaNeural",
    "ta-IN-PallaviNeural"
]

sentences = [
    "Hello this is a sample voice",
    "Artificial intelligence is powerful",
    "Voice authentication system test",
    "Machine learning detects fake voices",
    "Cyber security is important",
    "Deep learning audio processing",
    "Speech recognition experiment",
    "This is a synthetic voice sample"
]

def add_realism(audio):
    choice = random.choice(["speed", "pitch", "noise", "none"])

    if choice == "speed":
        factor = random.uniform(0.85, 1.15)
        audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * factor)
        }).set_frame_rate(audio.frame_rate)

    elif choice == "pitch":
        factor = random.uniform(0.9, 1.1)
        audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * factor)
        }).set_frame_rate(audio.frame_rate)

    elif choice == "noise":
        noise = AudioSegment.silent(duration=len(audio))
        audio = audio.overlay(noise - 20)

    return audio

existing = []
for file in os.listdir(fake_folder):
    m = re.match(r"fake_(\d+)\.wav", file)
    if m:
        existing.append(int(m.group(1)))

start_num = max(existing) + 1 if existing else 1
total_samples = 200

async def generate():
    for i in range(total_samples):
        file_num = start_num + i

        voice = voices[i % len(voices)]
        text = sentences[i % len(sentences)]

        print(f"Generating {i+1}/{total_samples}")

        mp3_path = f"temp_{file_num}.mp3"
        wav_path = os.path.join(fake_folder, f"fake_{file_num}.wav")

        try:
            communicate = edge_tts.Communicate(text=text, voice=voice)
            await communicate.save(mp3_path)

            audio = AudioSegment.from_mp3(mp3_path)

            # 🔥 IMPORTANT FIX
            audio = add_realism(audio)

            audio.export(wav_path, format="wav")

            os.remove(mp3_path)

            print(f"Saved {wav_path}")

        except Exception as e:
            print("Skipped:", e)

asyncio.run(generate())

print("DATASET FIX COMPLETE")
