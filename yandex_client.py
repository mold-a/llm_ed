import requests  # библиотека для HTTP-запросов к внешним API


class YandexGPTClient:
    # Константы с URL-адресами API — выносим сюда чтобы не дублировать в методах
    CHAT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    EMBED_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"

    def __init__(self, api_key: str, folder_id: str):
        self.api_key = api_key        # ключ для авторизации в Yandex Cloud
        self.folder_id = folder_id    # ID папки в Yandex Cloud — нужен для указания модели

        # Заголовки которые отправляем с каждым запросом
        # Authorization — говорит API кто мы такие
        # Content-Type — говорит API что тело запроса в формате JSON
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 500) -> str:
        # messages — список словарей вида {"role": "user/system/assistant", "text": "..."}
        # temperature — от 0 (детерминированно) до 1 (творчески/случайно)
        # max_tokens — максимальная длина ответа в токенах

        # Тело запроса — то что отправляем в API
        payload = {
            # modelUri указывает какую именно модель использовать
            # yandexgpt-lite — лёгкая и дешёвая, для обучения достаточно
            "modelUri": f"gpt://{self.folder_id}/yandexgpt-lite",

            "completionOptions": {
                "temperature": temperature,   # управляет случайностью ответа
                "maxTokens": max_tokens       # ограничение длины ответа
            },

            "messages": messages  # передаём всю историю диалога
        }

        # Отправляем POST-запрос с заголовками и телом в формате JSON
        response = requests.post(self.CHAT_URL, headers=self.headers, json=payload)

        # Если API вернул ошибку (4xx, 5xx) — выбрасываем исключение сразу
        # Без этой строки ошибки API были бы скрыты и сложно диагностировались бы
        response.raise_for_status()

        # Достаём текст ответа из вложенной структуры JSON
        # result → alternatives → первый вариант → message → text
        return response.json()["result"]["alternatives"][0]["message"]["text"]

    def embed(self, text: str) -> list[float]:
        # Превращает текст в вектор чисел (embedding)
        # Этот вектор потом используется для семантического поиска в ChromaDB

        payload = {
            # text-search-doc — модель оптимизированная для поиска по документам
            # существует также text-search-query — для запросов пользователя
            "modelUri": f"emb://{self.folder_id}/text-search-doc",

            "text": text  # текст который хотим превратить в вектор
        }

        response = requests.post(self.EMBED_URL, headers=self.headers, json=payload)

        # Аналогично chat() — бросаем исключение при ошибке API
        response.raise_for_status()

        # Достаём вектор из ответа — это список из ~256 чисел типа float
        # Каждое число — координата текста в многомерном пространстве смыслов
        return response.json()["embedding"]