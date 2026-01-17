"""
Vision Agent - оптимизированный анализ веб-страниц

Улучшения:
- Более точные промпты
- Лучший парсинг JSON
- Умная фильтрация элементов
- Кэширование результатов
"""

import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

from ..llm.base import BaseLLMProvider
from ..browser.dom_extractor import Element


@dataclass
class PageAnalysis:
    """Результат анализа страницы"""
    page_type: str
    relevant_elements: List[str]
    observations: List[str]
    confidence: float
    context: str
    raw_response: str = ""  # Для отладки

    def to_dict(self) -> Dict:
        """Конвертация в словарь"""
        return asdict(self)

    def __repr__(self) -> str:
        return f"<PageAnalysis type={self.page_type} elements={len(self.relevant_elements)} conf={self.confidence:.2f}>"


class VisionAgent:
    """
    Оптимизированный Vision Agent для анализа страниц.

    Улучшения:
    1. Более структурированные промпты
    2. Валидация ответов от LLM
    3. Fallback стратегии при ошибках
    4. Кэширование анализа для одинаковых страниц
    5. Умная приоритизация элементов
    """

    # Улучшенный системный промпт
    SYSTEM_PROMPT = """You are a Vision Agent - expert in analyzing web pages for complex goals.

    Your role: Understand page structure, identify elements relevant to the user's goal, and provide step-by-step guidance.
    Do NOT perform actions, only analyze.

    Response format (strict JSON):
    {
      "page_type": "one of: search_page|search_results|article|form|homepage|product|auth|error|other",
      "relevant_elements": ["elem_X", "elem_Y"],
      "observations": ["Factual observation 1", "Factual observation 2"],
      "confidence": 0.0-1.0,
      "context": "Additional context for Action Agent",
      "steps": ["Optional step-by-step instructions for next actions"],
      "warnings": ["Optional issues found"]
    }

    Guidelines:
    - Step 1: Determine page_type based on structure and content.
    - Step 2: Identify only elements essential for achieving the goal (2-10 preferred).
    - Step 3: Note clear, factual observations that help achieve the goal.
    - Step 4: Provide optional next steps for Action Agent if relevant.
    - Step 5: Include warnings if something looks broken or unexpected.
    - Use the goal and page elements to guide your analysis.
    - Prioritize visible and interactive elements.
    - Return ONLY valid JSON, no markdown, no extra text."""

    def __init__(self, llm_provider: BaseLLMProvider, cache_size: int = 100):
        """
        Args:
            llm_provider: LLM провайдер
            cache_size: Размер кэша анализов
        """
        self.llm = llm_provider
        self._cache: Dict[str, PageAnalysis] = {}
        self._cache_size = cache_size

    def _create_cache_key(self, url: str, title: str, elements_count: int) -> str:
        """
        Создаёт ключ для кэша на основе характеристик страницы.

        Простая эвристика: если URL, title и количество элементов одинаковые,
        скорее всего это та же страница.
        """
        key_string = f"{url}|{title}|{elements_count}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """
        Надёжный парсинг JSON из ответа LLM.

        Стратегии:
        1. Попытка прямого парсинга
        2. Поиск JSON между { }
        3. Удаление markdown
        4. Очистка от лишних символов
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Стратегия 1: Прямой парсинг
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Стратегия 2: Удаляем markdown
        text = text.replace('```json', '').replace('```', '')
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Стратегия 3: Извлекаем JSON между скобками
        try:
            start = text.find('{')
            end = text.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Стратегия 4: Ищем первый валидный JSON объект
        depth = 0
        start_pos = -1

        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start_pos = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_pos >= 0:
                    try:
                        return json.loads(text[start_pos:i + 1])
                    except json.JSONDecodeError:
                        start_pos = -1

        return None

    def _validate_analysis(self, data: Dict) -> Tuple[bool, str]:
        """
        Валидирует структуру ответа от LLM.

        Returns:
            (is_valid, error_message)
        """
        required_fields = ['page_type', 'relevant_elements', 'observations', 'confidence']

        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        # Проверяем типы
        if not isinstance(data['page_type'], str):
            return False, "page_type must be string"

        if not isinstance(data['relevant_elements'], list):
            return False, "relevant_elements must be array"

        if not isinstance(data['observations'], list):
            return False, "observations must be array"

        if not isinstance(data['confidence'], (int, float)):
            return False, "confidence must be number"

        # Проверяем значения
        if not (0.0 <= data['confidence'] <= 1.0):
            return False, "confidence must be between 0 and 1"

        # Проверяем что элементы - строки
        if not all(isinstance(e, str) for e in data['relevant_elements']):
            return False, "relevant_elements must contain strings"

        return True, ""

    def _create_fallback_analysis(self, elements: List[Element]) -> PageAnalysis:
        """
        Создаёт базовый анализ если LLM не справился.

        Использует простую эвристику на основе элементов.
        """
        # Определяем тип страницы эвристически
        has_input = any(e.tag == 'input' and e.type == 'text' for e in elements)
        has_search = any('search' in (e.placeholder or '').lower() for e in elements)
        has_many_links = sum(1 for e in elements if e.tag == 'a') > 10

        if has_search or (has_input and not has_many_links):
            page_type = "search_page"
        elif has_many_links and not has_input:
            page_type = "search_results"
        else:
            page_type = "other"

        # Берём первые видимые интерактивные элементы
        relevant = [
                       e.id for e in elements
                       if e.is_in_viewport and e.tag in ['input', 'button', 'a']
                   ][:10]

        return PageAnalysis(
            page_type=page_type,
            relevant_elements=relevant,
            observations=["Fallback analysis - LLM response failed"],
            confidence=0.3,
            context="Using heuristic analysis due to LLM error"
        )

    def _format_elements_optimized(self, elements: List[Element]) -> str:
        """
        Оптимизированное форматирование элементов для минимизации токенов.

        Стратегия:
        1. Группируем одинаковые типы
        2. Показываем только ключевую информацию
        3. Приоритизируем видимые элементы
        """
        visible = [e for e in elements if e.is_in_viewport]
        hidden = [e for e in elements if not e.is_in_viewport]

        lines = []

        # Статистика (очень компактно)
        by_tag = {}
        for elem in elements:
            by_tag[elem.tag] = by_tag.get(elem.tag, 0) + 1

        stats = ", ".join([f"{tag}:{count}" for tag, count in sorted(by_tag.items())])
        lines.append(f"Total: {len(elements)} elements ({stats})")
        lines.append("")

        # Видимые элементы (детально)
        if visible:
            lines.append(f"VISIBLE ({len(visible)}):")
            for elem in visible[:30]:  # Топ 30 видимых
                parts = [elem.id, elem.tag]

                if elem.text:
                    text = elem.text[:35].replace('\n', ' ')
                    parts.append(f'"{text}"')

                if elem.placeholder:
                    parts.append(f'ph:"{elem.placeholder[:20]}"')

                if elem.type:
                    parts.append(f't:{elem.type}')

                if elem.href:
                    parts.append('link')

                lines.append(" | ".join(parts))

        # Скрытые элементы (кратко, только если мало видимых)
        if len(visible) < 5 and hidden:
            lines.append("")
            lines.append(f"BELOW FOLD ({len(hidden)}, showing top 10):")
            for elem in hidden[:10]:
                parts = [elem.id, elem.tag]
                if elem.text:
                    parts.append(f'"{elem.text[:25]}"')
                lines.append(" | ".join(parts))

        return "\n".join(lines)

    async def analyze_page(
            self,
            goal: str,
            url: str,
            title: str,
            elements: List[Element],
            use_cache: bool = True
    ) -> PageAnalysis:
        """
        Анализирует страницу относительно цели.

        Args:
            goal: Цель пользователя
            url: URL страницы
            title: Заголовок
            elements: Список элементов
            use_cache: Использовать кэш

        Returns:
            PageAnalysis
        """
        # Проверяем кэш
        if use_cache:
            cache_key = self._create_cache_key(url, title, len(elements))
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                # Обновляем только список релевантных элементов под новую цель
                # (тип страницы остаётся тем же)
                return cached

        # Форматируем элементы оптимально
        elements_str = self._format_elements_optimized(elements)

        # Создаём промпт
        user_message = f"""GOAL: {goal}

PAGE INFO:
URL: {url}
Title: {title}

ELEMENTS:
{elements_str}

Analyze this page for the given goal."""

        try:
            # Получаем анализ
            response = await self.llm.generate_simple(
                user_message=user_message,
                system_prompt=self.SYSTEM_PROMPT
            )

            # Парсим JSON
            data = self._parse_json_response(response.content)

            if not data:
                print("⚠️  Vision Agent: Could not parse JSON, using fallback")
                return self._create_fallback_analysis(elements)

            # Валидируем
            is_valid, error = self._validate_analysis(data)
            if not is_valid:
                print(f"⚠️  Vision Agent: Invalid response - {error}, using fallback")
                return self._create_fallback_analysis(elements)

            # Создаём анализ
            analysis = PageAnalysis(
                page_type=data['page_type'],
                relevant_elements=data['relevant_elements'],
                observations=data['observations'],
                confidence=data['confidence'],
                context=data.get('context', ''),
                raw_response=response.content
            )

            # Кэшируем
            if use_cache:
                cache_key = self._create_cache_key(url, title, len(elements))
                self._cache[cache_key] = analysis

                # Ограничиваем размер кэша
                if len(self._cache) > self._cache_size:
                    # Удаляем первый элемент (FIFO)
                    first_key = next(iter(self._cache))
                    del self._cache[first_key]

            return analysis

        except Exception as e:
            print(f"⚠️  Vision Agent error: {e}")
            return self._create_fallback_analysis(elements)

    def filter_elements(
            self,
            all_elements: List[Element],
            relevant_ids: List[str],
            max_elements: int = 20
    ) -> List[Element]:
        """
        Умная фильтрация элементов.

        Стратегия:
        1. Берём элементы из relevant_ids
        2. Если мало - добавляем видимые интерактивные
        3. Если всё ещё мало - добавляем любые видимые
        4. Ограничиваем max_elements
        """
        relevant = []
        relevant_id_set = set(relevant_ids)

        # Шаг 1: Берём релевантные элементы
        for elem in all_elements:
            if elem.id in relevant_id_set:
                relevant.append(elem)

        # Шаг 2: Если мало релевантных, добавляем видимые интерактивные
        if len(relevant) < 5:
            interactive_tags = {'input', 'button', 'a', 'select', 'textarea'}

            for elem in all_elements:
                if (elem.is_in_viewport and
                        elem.tag in interactive_tags and
                        elem not in relevant and
                        len(relevant) < max_elements):
                    relevant.append(elem)

        # Шаг 3: Если всё ещё мало, добавляем любые видимые
        if len(relevant) < 3:
            for elem in all_elements:
                if (elem.is_in_viewport and
                        elem not in relevant and
                        len(relevant) < max_elements):
                    relevant.append(elem)

        # Сортируем: видимые первыми, потом по позиции Y (сверху вниз)
        relevant.sort(key=lambda e: (
            not e.is_in_viewport,  # False (видимые) раньше True (скрытые)
            e.position.get('y', 0)  # Потом по вертикальной позиции
        ))

        return relevant[:max_elements]

    def clear_cache(self):
        """Очищает кэш анализов"""
        self._cache.clear()
