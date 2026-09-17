import pyttsx3
import os

FAKE_DIR = "dataset/fake"
os.makedirs(FAKE_DIR, exist_ok=True)

SENTENCES = [
    "The weather today is bright and sunny.",
    "Artificial intelligence is changing the world.",
    "Please submit your assignment before the deadline.",
    "The train departs every hour from platform two.",
    "Our meeting has been rescheduled to Monday.",
    # இன்னும் sentences சேர்க்கலாம்
]

engine = pyttsx3.init()
voices = engine.getProperty('voices')

def get_next_index(folder, prefix):
    existing = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".wav")]
    nums = [int(f.replace(prefix, "").replace(".wav", "")) for f in existing if f.replace(prefix, "").replace(".wav", "").isdigit()]
    return max(nums) + 1 if nums else 1

start_idx = get_next_index(FAKE_DIR, "fake_")

count = 0
for i, text in enumerate(SENTENCES * 20):
    idx = start_idx + i
    filepath = os.path.join(FAKE_DIR, f"fake_{idx}.wav")
    voice = voices[i % len(voices)]
    engine.setProperty('voice', voice.id)
    engine.save_to_file(text, filepath)
    engine.runAndWait()
    count += 1
    print(f"Saved fake_{idx}.wav")

print(f"Done! Generated {count} files.")