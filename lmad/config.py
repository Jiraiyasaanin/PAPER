import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load .env from project root if it exists
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# Fallback models in order of preference if the selected model is unavailable or rate limited
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

# Debate configuration defaults
DEFAULT_MAX_ROUNDS = int(os.getenv("DEFAULT_MAX_ROUNDS", "3"))
DEFAULT_VERDICT_MODE = os.getenv("DEFAULT_VERDICT_MODE", "guilty_not_guilty")

# Supported verdict modes
VERDICT_MODES = {
    "guilty_not_guilty": ("Guilty", "Not Guilty"),
    "entails_does_not_entail": ("Entails", "Does Not Entail"),
    "liable_not_liable": ("Liable", "Not Liable"),
}
