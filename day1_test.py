from yandex_client import YandexGPTClient
import os
from dotenv import load_dotenv

load_dotenv()
client = YandexGPTClient(os.getenv("YANDEX_API_KEY"), os.getenv("YANDEX_FOLDER_ID"))

# Попробуй разные запросы — посмотри как меняются ответы
answer = client.chat([
    {"role": "user", "text": "Объясни что такое токен в LLM одним предложением"}
])
print(answer)