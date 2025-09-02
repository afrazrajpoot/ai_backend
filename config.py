import os
from dotenv import load_dotenv

load_dotenv()

class Settings:

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ALLOWED_ORIGINS = ["http://localhost:3000","http://127.0.0.1:3000"]
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

settings = Settings()