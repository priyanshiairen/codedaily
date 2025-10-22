import csv
import os
from datetime import datetime, timedelta
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'sample_chats_100.csv')

random.seed(42)

POSITIVE = [
    "Great job on the release!",
    "Thanks for the quick fix, much appreciated",
    "This looks good to me",
    "Fantastic progress today",
    "Nice work team, keep it up",
    "I love this approach",
    "Brilliant solution",
    "Well done on the update",
    "The performance is excellent",
    "Super helpful, thank you!",
]

NEUTRAL = [
    "Please review the PR when you can",
    "Let's schedule a meeting for tomorrow",
    "Can you share the logs?",
    "What time works for everyone?",
    "I'll take a look later",
    "We need more information",
    "The tests are running",
    "Build is in progress",
    "Let's document this",
    "Noted",
]

NEGATIVE = [
    "The build failed again and I'm frustrated",
    "Not good. This is getting annoying",
    "This approach sucks and is terrible",
    "The service is down; this is awful",
    "This design is dumb and useless",
    "I'm angry this was not tested",
    "What a terrible experience",
    "This keeps breaking. Hate this",
    "WTF is happening here",
    "This code is trash",
]

LABEL_TO_POOL = {
    'positive': POSITIVE,
    'neutral': NEUTRAL,
    'negative': NEGATIVE,
}

SENDERS = ["Alex", "Priya", "Sam", "Jamie", "Taylor", "Jordan", "Avery"]


def generate_rows(n: int = 100):
    start = datetime(2025, 1, 2, 9, 0, 0)
    labels = ["positive", "neutral", "negative"] * (n // 3) + ["neutral"] * (n % 3)
    random.shuffle(labels)
    for i, label in enumerate(labels):
        sender = random.choice(SENDERS)
        ts = start + timedelta(minutes=i * 5)
        message = random.choice(LABEL_TO_POOL[label])
        yield {
            'sender': sender,
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'message_text': message,
            'label': label,
        }


def main():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sender','timestamp','message_text','label'])
        writer.writeheader()
        for row in generate_rows(100):
            writer.writerow(row)
    print(f"Wrote {DATA_PATH}")


if __name__ == '__main__':
    main()
