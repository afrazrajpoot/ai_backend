import os
from dotenv import load_dotenv

load_dotenv()

class Settings:

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ALLOWED_ORIGINS = ["http://localhost:3000"]

settings = Settings()