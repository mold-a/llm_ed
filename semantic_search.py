from yandex_client import YandexGPTClient  # наш клиент для работы с Yandex API
import numpy as np  # для быстрой работы с векторами
import os  # для чтения переменных окружения
from dotenv import load_dotenv  # для загрузки .env файла

load_dotenv()  # читаем .env — загружаем YANDEX_API_KEY и YANDEX_FOLDER_ID

# создаём клиент YandexGPT
client = YandexGPTClient(
    api_key=os.getenv("YANDEX_API_KEY"),
    folder_id=os.getenv("YANDEX_FOLDER_ID")
)

# база фраз — 15 штук на разные темы
# специально разнообразные чтобы видеть как embedding ловит смысл
PHRASES = [
    # еда
    "Паста карбонара с беконом и сыром пармезан",
    "Рецепт борща с говядиной и сметаной",
    "Лучший стейк из мраморной говядины на гриле",

    # спорт
    "Чемпионат мира по футболу прошёл в Катаре",
    "Тренировка ног в спортзале: приседания и жим",
    "Марафон это забег на 42 километра",

    # машинное обучение
    "Нейронные сети для распознавания изображений",
    "Градиентный спуск минимизирует функцию потерь",
    "Большие языковые модели GPT и BERT",

    # природа
    "Амурский тигр живёт в дальневосточной тайге",
    "Коралловые рифы Большого Барьерного рифа",

    # музыка
    "Битлз записали альбом Abbey Road в 1969 году",
    "Классическая музыка Моцарта и Бетховена",

    # города
    "Санкт-Петербург построен Петром Первым",
    "Токио — столица Японии и самый населённый город",

    # программирование
    "Python популярен для машинного обучения и анализа данных",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Косинусное сходство между двумя векторами.

    Формула: (a·b) / (|a| * |b|)
    Возвращает число от -1 до 1:
      1.0  — векторы идентичны (полное совпадение смысла)
      0.0  — векторы перпендикулярны (не связаны)
     -1.0  — векторы противоположны (противоположные смыслы)
    """
    # np.dot считает скалярное произведение быстрее чем sum(x*y for ...)
    dot_product = np.dot(a, b)

    # np.linalg.norm считает длину вектора (корень из суммы квадратов)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # защита от деления на ноль — если вектор нулевой
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def build_index(phrases: list[str]) -> list[np.ndarray]:
    """
    Получает embeddings для всех фраз при старте.
    Делается один раз — потом только сравниваем с запросами.
    """
    print(f"Индексируем {len(phrases)} фраз...")

    embeddings = []
    for i, phrase in enumerate(phrases, 1):
        # client.embed возвращает list[float] — конвертируем в numpy array
        # numpy быстрее работает с векторными операциями
        embedding = np.array(client.embed(phrase))
        embeddings.append(embedding)
        print(f"  [{i}/{len(phrases)}] {phrase[:50]}...")

    print("Индекс готов.\n")
    return embeddings


def search(query: str, phrases: list[str], embeddings: list[np.ndarray]) -> None:
    """
    Ищет похожие фразы в индексе.
    Выводит топ-3 самых похожих и антитоп-3 самых непохожих.
    """
    # получаем embedding запроса
    query_embedding = np.array(client.embed(query))

    # считаем similarity между запросом и каждой фразой
    # результат — список пар (фраза, score)
    results = []
    for phrase, phrase_embedding in zip(phrases, embeddings):
        score = cosine_similarity(query_embedding, phrase_embedding)
        results.append((phrase, score))
    print(results)
    # сортируем по убыванию — самые похожие в начале
    # key=lambda x: x[1] означает "сортируй по второму элементу кортежа" (по score)
    results.sort(key=lambda x: x[1], reverse=True)

    # топ-3 самых похожих
    print("🎯 ТОП-3 (самые похожие по смыслу):")
    for i, (phrase, score) in enumerate(results[:3], 1):
        print(f"  {i}. [{score:.3f}] {phrase}")

    # антитоп-3 — берём с конца списка (после сортировки по убыванию это самые низкие)
    # [-3:] — последние три элемента, [::-1] — разворачиваем чтобы от меньшего к большему
    print("\n🗑  АНТИТОП-3 (самые непохожие по смыслу):")
    for i, (phrase, score) in enumerate(results[-3:][::-1], 1):
        print(f"  {i}. [{score:.3f}] {phrase}")


if __name__ == "__main__":
    # строим индекс — получаем embeddings для всех фраз один раз
    embeddings = build_index(PHRASES)

    print("Семантический поиск запущен.")
    print("Введи запрос или 'выход' для завершения.")
    print("=" * 60)

    # бесконечный цикл запросов
    while True:
        query = input("\nЗапрос: ").strip()

        if query.lower() == "выход":
            print("Завершаю работу.")
            break

        if not query:
            print("Введи непустой запрос.")
            continue

        try:
            search(query, PHRASES, embeddings)
        except Exception as e:
            print(f"Ошибка: {e}")