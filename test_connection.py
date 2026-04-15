import requests
import os
from dotenv import load_dotenv

load_dotenv()  # загружает переменные из .env

API_KEY = os.getenv("YANDEX_API_KEY")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

# Тестовый запрос к YandexGPT
response = requests.post(
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
    headers={
        "Authorization": f"Api-Key {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt-lite",
        "completionOptions": {"temperature": 0.3, "maxTokens": 100},
        "messages": [{"role": "user", "text": "Скажи 'Связь установлена' и ничего больше"}]
    }
)

print(response.status_code)          # должно быть 200
print(response.json()["result"]["alternatives"][0]["message"]["text"])