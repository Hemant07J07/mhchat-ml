CRISIS_KEYWORDS = [
    "kill myself", "i want to die", "end my life", "i might kill myself",
    "i'm going to hurt myself", "i want to die", "kill myself", "die", "end it all",
    "i can't go on", "i want to disappear", "i wish i was dead", "i don't want to exist",
    "i will kill myself", "i want to end it", "i can't live", "i might harm myself",
    "i want to hurt myself", "i want to harm myself", "hurt myself", "harm myself",
    "suicidal", "suicide", "self harm", "self-harm"
]

def contains_crisis_words(text:str) -> bool:
    t = text.lower()
    for kw in CRISIS_KEYWORDS:
        if kw in t:
            return True
    return False