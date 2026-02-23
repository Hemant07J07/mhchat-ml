# src/ml/expand_dataset.py
import random, csv

templates = {
    "casual_chat": [
        "Hey, how are you?",
        "What's up?",
        "Tell me a joke",
        "Are you there?",
        "Thanks, that helped",
        "Nice to meet you",
    ],
    "mental_health_support": [
        "I'm feeling {feeling} about {topic}",
        "I can't sleep because I'm {feeling}",
        "I need help with anxiety about {topic}",
        "How do I calm down when I'm {feeling}?",
        "Can you give me tips for {topic}?",
    ],
    "crisis": [
        "I want to die",
        "I have a plan to kill myself",
        "I'm going to hurt myself",
        "I wish I was dead",
        "I don't want to live anymore",
    ]
}

feelings = ["anxious", "depressed", "panic", "overwhelmed", "down", "stressed"]
topics = ["exams", "work", "relationships", "school", "money", "family"]

def generate(n_per_class=100):
    rows = []
    for label, templs in templates.items():
        for i in range(n_per_class):
            t = random.choice(templs)
            txt = t.format(feeling=random.choice(feelings), topic=random.choice(topics))
            rows.append((txt, label))
    random.shuffle(rows)
    with open("data/intent_data_expanded.csv", "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["text","label"])
        w.writerows(rows)
    print("Saved data/intent_data_expanded.csv with", len(rows), "rows")

if __name__ == "__main__":
    generate(120)  # creates 120 examples per class -> ~360 rows