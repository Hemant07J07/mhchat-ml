import os

from dotenv import load_dotenv

load_dotenv()

MODEL_PROFILES = {
    "LLAMA3_1B": {
        "chat": "llama3.2:1b",
        "embed": "embeddinggemma",
        "fallback": "llama3.2:1b",
    },
    "GEMMA4_2B": {
        "chat": "gemma4:e2b",
        "embed": "embeddinggemma",
        "fallback": "gemma3",
    },
    "GEMMA4_4B": {
        "chat": "gemma4:e4b",
        "embed": "embeddinggemma",
        "fallback": "gemma4:e2b",
    },
    "LEGACY": {
        "chat": "gemma3",
        "embed": "embeddinggemma",
        "fallback": "llama3.2:1b",
    },
}

DEFAULT_MODEL_PROFILE = "LLAMA3_1B"


def _resolve_profile(name: str):
    profile_name = (name or DEFAULT_MODEL_PROFILE).upper()
    return MODEL_PROFILES.get(profile_name, MODEL_PROFILES[DEFAULT_MODEL_PROFILE])


def get_model_config():
    profile_name = os.getenv("MODEL_PROFILE", DEFAULT_MODEL_PROFILE)
    profile = _resolve_profile(profile_name)

    return {
        "provider": os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        "profile": profile_name.upper(),
        "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "chat_model": os.getenv("OLLAMA_CHAT_MODEL", profile["chat"]),
        "embed_model": os.getenv("OLLAMA_EMBED_MODEL", profile["embed"]),
        "fallback_chat_model": os.getenv("OLLAMA_FALLBACK_CHAT_MODEL", profile["fallback"]),
    }
