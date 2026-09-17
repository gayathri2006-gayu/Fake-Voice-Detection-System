import os
import random
from gtts import gTTS

FAKE_DIR = "dataset/fake"
os.makedirs(FAKE_DIR, exist_ok=True)

SENTENCES = [
    "The weather today is bright and sunny with a gentle breeze.",
    "I am currently working on a project about voice detection technology.",
    "Artificial intelligence is changing the way we live and work.",
    "Please remember to submit your assignment before the deadline.",
    "The train to the city departs every hour from platform two.",
    "She enjoys reading books about science and history.",
    "Our team meeting has been rescheduled to next Monday morning.",
    "The recipe requires two cups of flour and a pinch of salt.",
    "Technology continues to evolve at a rapid and exciting pace.",
    "He walked quietly through the park enjoying the evening air.",
    "The new software update improves performance and battery life.",
    "Students are encouraged to ask questions during the lecture.",
    "The market was filled with fresh fruits and colorful vegetables.",
    "We need to finish the report before the end of the week.",
    "The mountains looked beautiful covered in fresh white snow.",
    "Could you please send me the file by tomorrow afternoon.",
    "The conference will be held online due to travel restrictions.",
    "A balanced diet and regular exercise keep the body healthy.",
    "The museum exhibit features paintings from the modern era.",
    "Traffic was heavy this morning because of the road construction.",
    "Hello, how are you doing today? I hope everything is fine.",
    "This is a sample sentence used for testing voice detection.",
    "The sun rises in the east and sets in the west every day.",
    "Learning new skills can open many doors in your career.",
    "The library remains open until nine in the evening on weekdays.",
]

TLDS = ["com", "co.in", "co.uk", "com.au", "ca", "de", "fr", "es", ]
NUM_TO_GENERATE = 1500


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


def main():
    start_idx = get_next_index(FAKE_DIR, "fake_")
    print(f"Starting from fake_{start_idx}.wav")
    print(f"Generating {NUM_TO_GENERATE} fake voice samples using gTTS...\n")

    generated = 0
    for i in range(NUM_TO_GENERATE):
        idx = start_idx + i
        text = random.choice(SENTENCES)
        tld = random.choice(TLDS)

        out_filename = f"fake_{idx}.wav"
        out_path = os.path.join(FAKE_DIR, out_filename)

        try:
            tts = gTTS(text=text, lang="en", tld=tld)
            tts.save(out_path)
            generated += 1
            print(f"[{i+1}/{NUM_TO_GENERATE}] Saved {out_filename} (accent: {tld})")
        except Exception as e:
            print(f"Failed to generate {out_filename}: {e}")

    print(f"\nDone! Generated {generated} new fake voice samples.")
    total_fake = len([f for f in os.listdir(FAKE_DIR) if f.endswith(".wav")])
    print(f"Total fake files now: {total_fake}")


if __name__ == "__main__":
    main()