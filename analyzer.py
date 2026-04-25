from yandex_client import YandexGPTClient
from pydantic import BaseModel, Field
from typing import Literal
from ddgs import DDGS
from sentence_transformers import CrossEncoder
import json
import re
import pprint
import numpy as np
import os
from dotenv import load_dotenv
#...
load_dotenv()

client = YandexGPTClient(
    api_key=os.getenv("YANDEX_API_KEY"),
    folder_id=os.getenv("YANDEX_FOLDER_ID")
)

print("Загружаем NLI модель...")
nli_model = CrossEncoder("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
print("NLI модель готова.\n")


# --- МОДЕЛИ ДАННЫХ ---

class AtomicFact(BaseModel):
    """Одно атомарное утверждение — минимальная единица для верификации."""
    text: str                                                     # само утверждение
    subject: str                                                   # субъект (кто/что)


class FactCheckResult(BaseModel):
    original_claim: str                                           # исходный факт
    atomic_facts: list[str]                                       # разбитые атомы
    search_queries: list[str]                                     # все поисковые запросы
    best_sentences: list[str]                                     # отобранные предложения
    verdict: Literal["confirmed", "unconfirmed", "contradicted"]
    confidence: float                                              # уверенность 0-1
    nli_scores: dict


class ArticleAnalysis(BaseModel):
    topic: str
    sentiment: Literal["positive", "negative", "neutral"]
    key_facts: list[str]
    summary: str


class FullAnalysis(BaseModel):
    topic: str
    sentiment: Literal["positive", "negative", "neutral"]
    key_facts: list[str]
    summary: str
    fact_checks: list[FactCheckResult]
    credibility_score: int = Field(ge=1, le=10)
    credibility_explanation: str


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def extract_json(text: str) -> dict | list:
    """Извлекает JSON из ответа модели."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        text = match.group()
    parsed = json.loads(text)
    if isinstance(parsed, list) and len(parsed) == 1:
        return parsed[0]
    return parsed


def get_structured_response(messages: list[dict], max_retries: int = 3) -> dict:
    """Отправляет запрос и повторяет если получили невалидный JSON."""
    for attempt in range(max_retries):
        response = client.chat(messages, temperature=0.1)
        try:
            return extract_json(response)
        except (json.JSONDecodeError, AttributeError) as e:
            if attempt < max_retries - 1:
                messages.append({"role": "assistant", "text": response})
                messages.append({
                    "role": "user",
                    "text": "Ошибка: невалидный JSON. Верни ТОЛЬКО чистый JSON."
                })
    raise ValueError(f"Не удалось получить JSON за {max_retries} попытки")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Вычисляет косинусное сходство между двумя векторами."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def is_english(text: str) -> bool:
    """Проверяет является ли текст преимущественно английским."""
    if not text:
        return False
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    total = sum(1 for c in text if c.isalpha())
    return (latin / total) > 0.6 if total > 0 else False


# --- ШАГ 1: ИЗВЛЕЧЕНИЕ ФАКТОВ ---

def extract_article_info(text: str) -> ArticleAnalysis:
    """YandexGPT извлекает структурированную информацию из текста."""
    print("[Шаг 1] Анализируем статью через YandexGPT...")

    messages = [
        {
            "role": "system",
            "text": (
                "Ты анализатор текста. Отвечай ТОЛЬКО одним валидным JSON-объектом без markdown. "
                "Первый символ — {, последний — }. "
                "Формат: {\"topic\": str, \"sentiment\": \"positive/negative/neutral\", "
                "\"key_facts\": [3-5 фактов], \"summary\": str}. "
                "Требования к key_facts: "
                "каждый факт — самодостаточное предложение с полным именем субъекта. "
                "Никогда не используй местоимения — всегда полное имя или название. "
                "Плохо: 'Учёный родился в 1879 году'. "
                "Хорошо: 'Альберт Эйнштейн родился в 1879 году в Германии'."
            )
        },
        {"role": "user", "text": f"Проанализируй статью как единое целое:\n\n{text}"}
    ]

    raw = get_structured_response(messages)
    if isinstance(raw, list):
        result = raw[0]
        all_facts = []
        for item in raw:
            all_facts.extend(item.get("key_facts", []))
        result["key_facts"] = all_facts
        raw = result

    return ArticleAnalysis(**raw)


# --- ШАГ 2: ДЕКОМПОЗИЦИЯ НА АТОМАРНЫЕ ФАКТЫ ---

def decompose_to_atomic_facts(claim: str) -> list[str]:
    """
    Разбивает сложное утверждение на атомарные факты.

    Зачем: NLI модели лучше работают с простыми утверждениями.
    'Эйнштейн получил Нобелевскую премию по физике в 1921 году за открытие эффекта'
    → ['Эйнштейн получил Нобелевскую премию',
       'Премия была по физике',
       'Премия получена в 1921 году',
       'Премия за открытие фотоэлектрического эффекта']
    """
    messages = [
        {
            "role": "system",
            "text": (
                "Разбей утверждение на минимальные атомарные факты. "
                "Каждый атом — одно простое утверждение которое можно проверить независимо. "
                "Каждый атом должен содержать имя субъекта (не местоимение). "
                "Если утверждение уже простое — верни его одним элементом. "
                "Отвечай ТОЛЬКО JSON: {\"atoms\": [\"факт 1\", \"факт 2\"]}"
            )
        },
        {"role": "user", "text": f"Утверждение: {claim}"}
    ]

    try:
        raw = get_structured_response(messages)
        atoms = raw.get("atoms", [claim])
        # если модель вернула не список — оборачиваем
        if isinstance(atoms, str):
            atoms = [atoms]
        return atoms if atoms else [claim]
    except Exception:
        return [claim]  # fallback — оригинальное утверждение


# --- ШАГ 3: MULTI-HOP ПОИСК ---

def generate_search_queries(claim: str, previous_queries: list[str] = None) -> list[str]:
    """
    Генерирует поисковые запросы для проверки утверждения.
    При повторном вызове учитывает предыдущие запросы чтобы не повторяться.
    """
    context = ""
    if previous_queries:
        context = (
            f"Уже использованные запросы (не повторяй): {', '.join(previous_queries)}. "
            "Сформулируй другие запросы."
        )

    messages = [
        {
            "role": "system",
            "text": (
                "Сформулируй 2 поисковых запроса для проверки утверждения. "
                "Запросы должны быть на русском языке, короткими (3-5 слов). "
                "Первый запрос — прямой поиск факта. "
                "Второй запрос — альтернативная формулировка или уточнение. "
                f"{context}"
                "Отвечай ТОЛЬКО JSON: {\"queries\": [\"запрос 1\", \"запрос 2\"]}"
            )
        },
        {"role": "user", "text": f"Утверждение: {claim}"}
    ]

    try:
        raw = get_structured_response(messages)
        queries = raw.get("queries", [claim[:50]])
        if isinstance(queries, str):
            queries = [queries]
        return [str(q) for q in queries]
    except Exception:
        return [claim[:50]]


def search_web(query: str) -> list[dict]:
    """Выполняет один поисковый запрос через DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="ru-ru", max_results=3)
        return results or []
    except Exception as e:
        print(f"    Ошибка поиска '{query}': {e}")
        return []


def multi_hop_search(claim: str, max_hops: int = 2) -> tuple[list[str], list[str]]:
    """
    Итеративный поиск доказательств.

    Шаг 1: генерируем запросы → ищем → собираем сниппеты
    Шаг 2: если не нашли достаточно — генерируем новые запросы и ищем ещё раз

    Возвращает (все_сниппеты, все_запросы)
    """
    all_snippets = []   # все найденные тексты
    all_queries = []    # все использованные запросы

    for hop in range(max_hops):
        # генерируем новые запросы с учётом уже использованных
        queries = generate_search_queries(claim, all_queries)

        for query in queries:
            if query in all_queries:
                continue  # не повторяем запросы

            all_queries.append(query)
            results = search_web(query)

            for r in results:
                snippet = f"{r.get('title', '')} {r.get('body', '')}".strip()
                if snippet and snippet not in all_snippets:
                    all_snippets.append(snippet)

        # если нашли достаточно текста — останавливаемся
        if len(all_snippets) >= 5:
            break

    return all_snippets, all_queries


# --- ШАГ 4: SENTENCE SELECTION ---

def select_best_sentences(claim: str, snippets: list[str], top_k: int = 3) -> list[str]:
    """
    Отбор релевантных предложений с сохранением контекста.
    Вместо обрезанных фраз берём предложения + их контекст.
    """
    if not snippets:
        return []

    # разбиваем каждый сниппет на предложения с индексами внутри сниппета
    # храним (sentence, snippet_index, position_in_snippet, full_snippet_sentences)
    sentence_map = []

    for snippet_idx, snippet in enumerate(snippets):
        sentences = re.split(r'(?<=[.!?])\s+', snippet)  # не теряем знак препинания
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        for pos, sent in enumerate(sentences):
            sentence_map.append({
                "text": sent,
                "snippet_idx": snippet_idx,
                "position": pos,
                "all_sentences": sentences,
            })

    if not sentence_map:
        return [snippets[0][:500]] if snippets else []

    # embedding утверждения
    try:
        claim_embedding = client.embed(claim)
    except Exception:
        return [s["text"] for s in sentence_map[:top_k]]

    # скорим каждое предложение
    scored = []
    for item in sentence_map:
        try:
            emb = client.embed(item["text"])
            score = cosine_similarity(claim_embedding, emb)
            scored.append((score, item))
        except Exception:
            continue

    if not scored:
        return [s["text"] for s in sentence_map[:top_k]]

    scored.sort(key=lambda x: x[0], reverse=True)

    # для топ-k берём не только само предложение, но и его контекст:
    # предыдущее + само + следующее (если есть)
    result = []
    seen = set()

    for score, item in scored[:top_k]:
        all_sents = item["all_sentences"]
        pos = item["position"]

        # собираем окно: [prev, current, next]
        start = max(0, pos - 1)
        end = min(len(all_sents), pos + 2)
        context = " ".join(all_sents[start:end])

        if context not in seen:
            seen.add(context)
            result.append(context)
            print(f"    Predложение (score={score:.3f}): {context[:120]}...")

    return result


# --- ШАГ 5: NLI ВЕРИФИКАЦИЯ ---

def softmax(scores: np.ndarray) -> np.ndarray:
    """Конвертирует логиты в вероятности."""
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def check_number_contradiction(claim: str, evidence: str) -> bool:
    """Проверяет числовые противоречия между утверждением и контекстом."""
    claim_numbers = set(re.findall(r'\b\d{4}\b', claim))
    evidence_numbers = set(re.findall(r'\b\d{4}\b', evidence))
    if not claim_numbers or not evidence_numbers:
        return False
    # противоречие: числа из утверждения не встречаются в контексте
    return claim_numbers.isdisjoint(evidence_numbers)


def verify_atomic_fact(atom: str, best_sentences: list[str]) -> dict:
    """
    Верифицирует атомарное утверждение через LLM-judge.
    LLM лучше NLI понимает русский и умеет рассуждать.
    """
    if not best_sentences:
        return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

    evidence = "\n".join(f"- {s}" for s in best_sentences)

    messages = [
        {
            "role": "system",
            "text": (
                "Ты строгий fact-checker. Твоя задача — определить подтверждают ли "
                "ДОКАЗАТЕЛЬСТВА конкретное УТВЕРЖДЕНИЕ. "
                "Анализируй только факты, не используй внешние знания. "
                ""
                "Три возможных вердикта:"
                ""
                "1) confirmed — доказательства ЯВНО подтверждают утверждение. "
                "Все конкретные данные (даты, числа, имена, места) из утверждения "
                "совпадают с доказательствами."
                ""
                "2) contradicted — доказательства ЯВНО противоречат утверждению. "
                "Хотя бы одна конкретная деталь (дата, число, имя) в утверждении "
                "не совпадает с доказательствами."
                ""
                "3) unconfirmed — в доказательствах нет достаточной информации "
                "чтобы подтвердить или опровергнуть утверждение."
                ""
                "Отвечай ТОЛЬКО JSON: "
                "{\"verdict\": \"confirmed/contradicted/unconfirmed\", "
                "\"confidence\": float 0-1, "
                "\"reasoning\": \"1-2 предложения почему\"}"
            )
        },
        {
            "role": "user",
            "text": (
                f"УТВЕРЖДЕНИЕ:\n{atom}\n\n"
                f"ДОКАЗАТЕЛЬСТВА:\n{evidence}"
            )
        }
    ]

    try:
        raw = get_structured_response(messages)
        verdict = raw.get("verdict", "unconfirmed")
        confidence = float(raw.get("confidence", 0.5))

        print(f"    LLM-judge: {verdict} ({confidence:.2f}) — {raw.get('reasoning', '')[:80]}")

        # конвертируем вердикт LLM в scores для совместимости
        if verdict == "confirmed":
            return {"entailment": confidence, "contradiction": 0.0,
                    "neutral": 1 - confidence}
        elif verdict == "contradicted":
            return {"entailment": 0.0, "contradiction": confidence,
                    "neutral": 1 - confidence}
        else:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": confidence}

    except Exception as e:
        print(f"    Ошибка LLM-judge: {e}")
        return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}


def aggregate_atomic_verdicts(atomic_scores: list[dict]) -> tuple[str, dict, float]:
    """
    Агрегирует вердикты атомов на основе большинства голосов.

    Новая логика:
    - считаем сколько атомов в каждой категории
    - побеждает категория с большинством голосов
    - при равенстве приоритет: contradicted > unconfirmed > confirmed
      (лучше ложное опровержение чем ложное подтверждение)
    """
    if not atomic_scores:
        return "unconfirmed", {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}, 0.0

    avg_scores = {
        "entailment": sum(s["entailment"] for s in atomic_scores) / len(atomic_scores),
        "contradiction": sum(s["contradiction"] for s in atomic_scores) / len(atomic_scores),
        "neutral": sum(s["neutral"] for s in atomic_scores) / len(atomic_scores),
    }

    # определяем категорию каждого атома по максимальному скору
    counts = {"confirmed": 0, "contradicted": 0, "unconfirmed": 0}
    for s in atomic_scores:
        if s["entailment"] >= s["contradiction"] and s["entailment"] >= s["neutral"]:
            counts["confirmed"] += 1
        elif s["contradiction"] >= s["neutral"]:
            counts["contradicted"] += 1
        else:
            counts["unconfirmed"] += 1

    total = len(atomic_scores)

    # критическое противоречие: больше 40% атомов явно опровергнуты
    if counts["contradicted"] / total > 0.4:
        return "contradicted", avg_scores, avg_scores["contradiction"]

    # большинство подтверждено: минимум 60% атомов confirmed
    if counts["confirmed"] / total >= 0.6:
        return "confirmed", avg_scores, avg_scores["entailment"]

    # иначе — unconfirmed
    return "unconfirmed", avg_scores, avg_scores["neutral"]


# --- ГЛАВНАЯ ФУНКЦИЯ ВЕРИФИКАЦИИ ОДНОГО ФАКТА ---

def verify_claim(claim: str) -> FactCheckResult:
    """
    Полный пайплайн верификации одного факта:
    1. Декомпозиция на атомарные факты
    2. Multi-hop поиск доказательств
    3. Sentence selection — отбор релевантных предложений
    4. NLI верификация каждого атома
    5. Агрегация вердиктов
    """

    # шаг 2а: декомпозиция
    atomic_facts = decompose_to_atomic_facts(claim)
    print(f"  Атомарных фактов: {len(atomic_facts)}")
    for i, atom in enumerate(atomic_facts, 1):
        print(f"    {i}. {atom[:70]}")

    # шаг 3: multi-hop поиск (общий для всех атомов этого факта)
    snippets, queries = multi_hop_search(claim)
    print(f"  Найдено сниппетов: {len(snippets)}, запросов: {len(queries)}")
    print(f"  Запросы: {queries}")

    # шаг 4: sentence selection — отбираем лучшие предложения для всего факта
    best_sentences = select_best_sentences(claim, snippets, top_k=3)
    print(f"  Отобрано предложений: {len(best_sentences)}")

    # шаг 5: NLI верификация каждого атома
    atomic_scores = []
    for atom in atomic_facts:
        scores = verify_atomic_fact(atom, best_sentences)
        atomic_scores.append(scores)

    # шаг 6: агрегация
    verdict, avg_scores, confidence = aggregate_atomic_verdicts(atomic_scores)

    # ДОПОЛНИТЕЛЬНО — если вердикт неуверенный, спрашиваем LLM напрямую
    sanity = final_sanity_check(claim, atomic_scores)
    if sanity and sanity["confidence"] > 0.8:
        print(f"  Sanity check переопределил вердикт: "
              f"{verdict} → {sanity['verdict']} ({sanity['reasoning'][:80]})")
        verdict = sanity["verdict"]
        confidence = sanity["confidence"]

    print(f"  Финальный вердикт: {verdict} (confidence={confidence:.2f})")

    return FactCheckResult(
        original_claim=claim,
        atomic_facts=atomic_facts,
        search_queries=queries,
        best_sentences=best_sentences,
        verdict=verdict,
        confidence=round(confidence, 3),
        nli_scores=avg_scores
    )


# --- ШАГ 1: ИЗВЛЕЧЕНИЕ + ВЕРИФИКАЦИЯ ---

def calculate_credibility(fact_checks: list[FactCheckResult]) -> tuple[int, str]:
    """Считает оценку достоверности математически."""
    if not fact_checks:
        return 5, "Нет проверяемых фактов"

    confirmed    = sum(1 for f in fact_checks if f.verdict == "confirmed")
    contradicted = sum(1 for f in fact_checks if f.verdict == "contradicted")
    unconfirmed  = sum(1 for f in fact_checks if f.verdict == "unconfirmed")

    score = 5 + (confirmed * 2) - (contradicted * 3)
    score = max(1, min(10, score))

    explanation = (
        f"Проверено фактов: {len(fact_checks)}. "
        f"Подтверждено: {confirmed}, "
        f"Опровергнуто: {contradicted}, "
        f"Не найдено: {unconfirmed}. "
        f"Итоговая оценка: {score}/10."
    )
    return score, explanation


def analyze_article(text: str) -> FullAnalysis:
    """Полный пайплайн анализа статьи."""

    article_info = extract_article_info(text)

    print(f"\n[Шаг 2-5] Верифицируем {len(article_info.key_facts)} фактов...\n")
    fact_checks = []

    for i, claim in enumerate(article_info.key_facts, 1):
        print(f"{'='*50}")
        print(f"Факт {i}: {claim[:80]}...")
        result = verify_claim(claim)
        fact_checks.append(result)
        print()

    score, explanation = calculate_credibility(fact_checks)

    return FullAnalysis(
        topic=article_info.topic,
        sentiment=article_info.sentiment,
        key_facts=article_info.key_facts,
        summary=article_info.summary,
        fact_checks=fact_checks,
        credibility_score=score,
        credibility_explanation=explanation
    )

def final_sanity_check(claim: str, atomic_verdicts: list[dict]) -> dict | None:
    """
    Финальная проверка: если агрегация не дала уверенного ответа,
    спрашиваем LLM напрямую, используя её общие знания + контекст.
    Возвращает None если проверка не нужна.
    """
    # проверяем нужна ли доп. проверка
    counts = {"confirmed": 0, "contradicted": 0, "unconfirmed": 0}
    for s in atomic_verdicts:
        if s["entailment"] >= max(s["contradiction"], s["neutral"]):
            counts["confirmed"] += 1
        elif s["contradiction"] >= s["neutral"]:
            counts["contradicted"] += 1
        else:
            counts["unconfirmed"] += 1

    # проверка нужна только если вердикт противоречивый или много unconfirmed
    has_mixed = counts["confirmed"] > 0 and counts["contradicted"] > 0
    mostly_unknown = counts["unconfirmed"] / len(atomic_verdicts) > 0.5

    if not (has_mixed or mostly_unknown):
        return None  # агрегация уверена — доп. проверка не нужна

    messages = [
        {
            "role": "system",
            "text": (
                "Ты fact-checker с широкими знаниями. "
                "Оцени фактологическую точность утверждения используя общеизвестные факты. "
                "Отвечай ТОЛЬКО JSON: "
                "{\"verdict\": \"confirmed/contradicted/unconfirmed\", "
                "\"confidence\": float 0-1, "
                "\"reasoning\": \"краткое объяснение\"}. "
                "confirmed — утверждение соответствует общеизвестным фактам. "
                "contradicted — утверждение противоречит общеизвестным фактам. "
                "unconfirmed — для оценки нужна специальная экспертиза."
            )
        },
        {"role": "user", "text": f"Утверждение для проверки: {claim}"}
    ]

    try:
        raw = get_structured_response(messages)
        return {
            "verdict": raw.get("verdict", "unconfirmed"),
            "confidence": float(raw.get("confidence", 0.5)),
            "reasoning": raw.get("reasoning", "")
        }
    except Exception:
        return None

# --- ТЕСТ ---

SAMPLE_TEXT = """
Альберт Эйнштейн получил Нобелевскую премию по физике в 1921 году за открытие 
закона фотоэлектрического эффекта. Учёный родился в 1879 году в Германии.
Теорию относительности он разработал в 1950 году, уже находясь в США.
Эйнштейн также известен тем, что изобрёл атомную бомбу в одиночку.
"""

if __name__ == "__main__":
    print("Запускаем полный анализ...\n")

    try:
        result = analyze_article(SAMPLE_TEXT)

        print("\n" + "=" * 60)
        print("ИТОГОВЫЙ РЕЗУЛЬТАТ:")
        print("=" * 60)
        print(f"Тема:          {result.topic}")
        print(f"Тональность:   {result.sentiment}")
        print(f"Резюме:        {result.summary}")
        print(f"\nДостоверность: {result.credibility_score}/10")
        print(f"Объяснение:    {result.credibility_explanation}")

        print(f"\nПроверка фактов ({len(result.fact_checks)} шт.):")
        icons = {"confirmed": "✅", "unconfirmed": "❓", "contradicted": "❌"}
        for i, fc in enumerate(result.fact_checks, 1):
            print(f"\n  {i}. {icons[fc.verdict]} [{fc.verdict}] "
                  f"(confidence={fc.confidence:.2f})")
            print(f"     Факт: {fc.original_claim[:70]}")
            print(f"     Атомы: {len(fc.atomic_facts)} шт.")
            print(f"     Лучшие предложения: {len(fc.best_sentences)} шт.")
            if fc.best_sentences:
                print(f"     Evidence: {fc.best_sentences[0][:100]}...")

    except ValueError as e:
        print(f"Критическая ошибка: {e}")