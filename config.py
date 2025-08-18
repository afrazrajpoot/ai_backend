import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ALLOWED_ORIGINS = ["http://localhost:3000"]

settings = Settings()