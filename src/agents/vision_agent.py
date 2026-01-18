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
    next_action_hint: Optional[str] = None
    element_priorities: Optional[Dict[str, float]] = None
    raw_response: str = ""

    def __post_init__(self):
        if self.element_priorities is None:
            self.element_priorities = {}

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_top_elements(self, n: int = 5) -> List[str]:
        """Возвращает топ N элементов по приоритету"""
        if not self.element_priorities:
            return self.relevant_elements[:n]

        sorted_elements = sorted(
            self.element_priorities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [elem_id for elem_id, _ in sorted_elements[:n]]

    def __repr__(self) -> str:
        return f"<PageAnalysis type={self.page_type} elements={len(self.relevant_elements)} conf={self.confidence:.2f}>"


class VisionAgent:
    """
    Оптимизированный Vision Agent с двухфазным анализом.

    Фаза 1: Быстрый структурный анализ (определение типа страницы)
    Фаза 2: Детальный анализ релевантных элементов
    """

    # Компактный промпт для фазы 1 (структурный анализ)
    STRUCTURE_PROMPT = """Analyze page structure and type. Return JSON:
{
  "page_type": "search_page|search_results|article|form|product|auth|email_inbox|email_list|vacancy_list|vacancy_detail|profile|other",
  "key_sections": ["header", "main_content", "sidebar"],
  "interactive_count": 15,
  "confidence": 0.9
}

Page types:
- search_page: has search input
- search_results: list of links/items
- article: long text content
- form: multiple inputs
- email_inbox: email list interface
- vacancy_list: job listings
- vacancy_detail: single job description
- profile: user profile page

Return ONLY JSON."""

    # Детальный промпт для фазы 2 (анализ элементов)
    DETAIL_PROMPT = """You are Vision Agent analyzing page elements for a goal.

CRITICAL: Your job is ONLY to identify relevant elements, NOT to decide actions.
Action Agent will use your analysis to make decisions.

Response format (strict JSON):
{
  "relevant_elements": ["elem_X", "elem_Y"],
  "element_priorities": {
    "elem_X": 0.95,
    "elem_Y": 0.80
  },
  "observations": [
    "Clear factual observation 1",
    "Clear factual observation 2"
  ],
  "next_action_hint": "Optional: suggest what type of action might help (click/type/scroll)",
  "confidence": 0.0-1.0,
  "context": "Additional context for Action Agent"
}

Guidelines:
1. Identify 3-10 elements most relevant to the goal
2. Assign priority (0.0-1.0) based on:
   - Visual prominence (size, position)
   - Semantic relevance to goal
   - Interactivity (buttons > links > text)
3. Observations should be factual, not interpretive
4. next_action_hint is optional but helpful
5. Only use element IDs from the provided list

Return ONLY valid JSON, no markdown."""

    def __init__(self, llm_provider: BaseLLMProvider, cache_size: int = 100):
        self.llm = llm_provider
        self._structure_cache: Dict[str, Dict] = {}
        self._detail_cache: Dict[str, PageAnalysis] = {}
        self._cache_size = cache_size

    def _create_structure_key(self, url: str, title: str, elem_count: int) -> str:
        """Ключ для кэша структурного анализа"""
        key_string = f"{url}|{title}|{elem_count}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _create_detail_key(self, structure_key: str, goal: str) -> str:
        """Ключ для кэша детального анализа (зависит от цели)"""
        key_string = f"{structure_key}|{goal}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Надёжный парсинг JSON"""
        if not text or not text.strip():
            return None

        text = text.strip()

        # Удаляем markdown если есть
        text = text.replace('```json', '').replace('```', '').strip()

        # Прямой парсинг
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Извлечение между { }
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        # Поиск первого валидного JSON
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

    def _collect_element_stats(self, elements: List[Element]) -> Dict:
        """Собирает статистику элементов"""
        stats = {
            'visible': sum(1 for e in elements if e.is_in_viewport),
            'inputs': 0,
            'input_types': set(),
            'buttons': 0,
            'links': 0,
            'forms': 0,
            'lists': 0
        }

        for elem in elements:
            if elem.tag == 'input':
                stats['inputs'] += 1
                if elem.type:
                    stats['input_types'].add(elem.type)
            elif elem.tag == 'button':
                stats['buttons'] += 1
            elif elem.tag == 'a':
                stats['links'] += 1
            elif elem.tag == 'form':
                stats['forms'] += 1
            elif elem.tag in ['ul', 'ol']:
                stats['lists'] += 1

        stats['input_types'] = list(stats['input_types'])
        return stats

    def _extract_key_text(self, elements: List[Element], max_chars: int = 200) -> str:
        """Извлекает ключевые текстовые фрагменты"""
        texts = []
        total_chars = 0

        # Приоритет: заголовки, кнопки, видимые элементы
        priority_elements = sorted(
            elements,
            key=lambda e: (
                e.tag not in ['h1', 'h2', 'h3'],
                e.tag != 'button',
                not e.is_in_viewport
            )
        )

        for elem in priority_elements:
            if elem.text and total_chars < max_chars:
                text = elem.text.strip()[:50]
                if text and text not in texts:
                    texts.append(text)
                    total_chars += len(text)

        return ' | '.join(texts[:5])

    def _heuristic_structure_analysis(self, stats: Dict) -> Dict:
        """Эвристическое определение типа страницы"""
        # Поиск
        if 'search' in stats['input_types'] or (stats['inputs'] == 1 and stats['buttons'] >= 1):
            return {'page_type': 'search_page', 'confidence': 0.7}

        # Результаты поиска
        if stats['links'] > 10 and stats['lists'] > 0:
            return {'page_type': 'search_results', 'confidence': 0.6}

        # Форма
        if stats['inputs'] >= 3 and stats['forms'] >= 1:
            return {'page_type': 'form', 'confidence': 0.7}

        # Статья (мало интерактивных элементов)
        if stats['links'] < 5 and stats['inputs'] == 0:
            return {'page_type': 'article', 'confidence': 0.5}

        return {'page_type': 'other', 'confidence': 0.3}

    async def _analyze_structure(
        self,
        url: str,
        title: str,
        elements: List[Element]
    ) -> Dict:
        """
        Фаза 1: Быстрый структурный анализ страницы.

        Определяет тип страницы без детального анализа элементов.
        """
        # Проверяем кэш
        cache_key = self._create_structure_key(url, title, len(elements))
        if cache_key in self._structure_cache:
            return self._structure_cache[cache_key]

        # Собираем статистику элементов
        stats = self._collect_element_stats(elements)

        # Компактное описание для LLM
        summary = f"""URL: {url}
Title: {title}

Element statistics:
- Total: {len(elements)} ({stats['visible']} visible)
- Inputs: {stats['inputs']} (types: {', '.join(stats['input_types'])})
- Buttons: {stats['buttons']}
- Links: {stats['links']}
- Forms: {stats['forms']}
- Lists: {stats['lists']}

Key text snippets:
{self._extract_key_text(elements, max_chars=200)}"""

        try:
            response = await self.llm.generate_simple(
                user_message=summary,
                system_prompt=self.STRUCTURE_PROMPT
            )

            data = self._parse_json_response(response.content)

            if data and 'page_type' in data:
                # Кэшируем
                self._structure_cache[cache_key] = data
                self._limit_cache(self._structure_cache)
                return data

        except Exception as e:
            print(f"⚠️  Structure analysis failed: {e}")

        # Fallback: эвристический анализ
        return self._heuristic_structure_analysis(stats)

    def _format_elements_for_detail_analysis(
        self,
        elements: List[Element],
        page_type: str
    ) -> str:
        """
        Форматирует элементы для детального анализа.

        Адаптивная детализация на основе типа страницы.
        """
        # Фильтруем по типу страницы
        if page_type == 'search_page':
            priority_tags = {'input', 'button', 'form'}
        elif page_type == 'search_results':
            priority_tags = {'a', 'button'}
        elif page_type in ['article', 'vacancy_detail']:
            priority_tags = {'h1', 'h2', 'h3', 'a', 'button'}
        elif page_type in ['email_inbox', 'email_list', 'vacancy_list']:
            priority_tags = {'a', 'button', 'input'}
        else:
            priority_tags = {'input', 'button', 'a'}

        # Сортируем: приоритетные теги → видимые → позиция
        sorted_elements = sorted(
            elements,
            key=lambda e: (
                e.tag not in priority_tags,
                not e.is_in_viewport,
                e.position.get('y', 0)
            )
        )

        lines = []
        for elem in sorted_elements[:20]:
            parts = [f"[{elem.id}]", elem.tag.upper()]

            if elem.text:
                text = elem.text[:40].replace('\n', ' ').strip()
                if text:
                    parts.append(f'"{text}"')

            if elem.placeholder:
                parts.append(f'placeholder="{elem.placeholder[:25]}"')

            if elem.type:
                parts.append(f'type={elem.type}')

            if elem.href:
                href = elem.href.split('/')[-1][:30] if '/' in elem.href else elem.href[:30]
                parts.append(f'href=.../{href}')

            if elem.is_in_viewport:
                parts.append('✓visible')

            lines.append(' '.join(parts))

        return '\n'.join(lines)

    def _validate_detail_response(self, data: Dict) -> bool:
        """Валидация ответа детального анализа"""
        required = ['relevant_elements', 'observations', 'confidence']

        for field in required:
            if field not in data:
                return False

        if not isinstance(data['relevant_elements'], list):
            return False

        if not isinstance(data['observations'], list):
            return False

        if not isinstance(data['confidence'], (int, float)):
            return False

        if not (0.0 <= data['confidence'] <= 1.0):
            return False

        return True

    def _create_fallback_analysis(
        self,
        elements: List[Element],
        page_type: str,
        confidence: float
    ) -> PageAnalysis:
        """Создаёт fallback анализ на основе эвристик"""
        # Берём видимые интерактивные элементы
        interactive = [
            e for e in elements
            if e.is_in_viewport and e.tag in ['input', 'button', 'a']
        ][:10]

        priorities = {}
        for i, elem in enumerate(interactive):
            base_priority = 1.0 - (i * 0.1)
            if elem.tag == 'button':
                base_priority += 0.1
            elif elem.tag == 'input':
                base_priority += 0.05

            priorities[elem.id] = min(base_priority, 1.0)

        return PageAnalysis(
            page_type=page_type,
            relevant_elements=[e.id for e in interactive],
            observations=["Fallback analysis - using heuristics"],
            confidence=confidence * 0.5,
            context="Heuristic element selection",
            element_priorities=priorities
        )

    async def analyze_page(
        self,
        goal: str,
        url: str,
        title: str,
        elements: List[Element],
        use_cache: bool = True
    ) -> PageAnalysis:
        """
        Двухфазный анализ страницы.

        Фаза 1: Определение структуры и типа
        Фаза 2: Детальный анализ релевантных элементов
        """
        # Фаза 1: Структурный анализ (кэшируется по странице)
        structure = await self._analyze_structure(url, title, elements)  # ИСПРАВЛЕНО: добавлен await
        page_type = structure.get('page_type', 'other')
        structure_confidence = structure.get('confidence', 0.5)

        # Проверяем кэш детального анализа
        if use_cache:
            structure_key = self._create_structure_key(url, title, len(elements))
            detail_key = self._create_detail_key(structure_key, goal)

            if detail_key in self._detail_cache:
                return self._detail_cache[detail_key]

        # Фаза 2: Детальный анализ элементов
        elements_str = self._format_elements_for_detail_analysis(elements, page_type)

        user_message = f"""GOAL: {goal}

PAGE TYPE: {page_type} (confidence: {structure_confidence:.2f})

ELEMENTS:
{elements_str}

Identify elements relevant to achieving the goal."""

        try:
            response = await self.llm.generate_simple(
                user_message=user_message,
                system_prompt=self.DETAIL_PROMPT
            )

            data = self._parse_json_response(response.content)

            if not data:
                print("⚠️  Vision Agent: JSON parse failed, using fallback")
                return self._create_fallback_analysis(elements, page_type, structure_confidence)

            # Валидация
            if not self._validate_detail_response(data):
                print("⚠️  Vision Agent: Invalid response structure")
                return self._create_fallback_analysis(elements, page_type, structure_confidence)

            # Создаём анализ
            analysis = PageAnalysis(
                page_type=page_type,
                relevant_elements=data.get('relevant_elements', []),
                observations=data.get('observations', []),
                confidence=min(data.get('confidence', 0.5), structure_confidence),
                context=data.get('context', ''),
                next_action_hint=data.get('next_action_hint'),
                element_priorities=data.get('element_priorities', {}),
                raw_response=response.content
            )

            # Кэшируем
            if use_cache:
                structure_key = self._create_structure_key(url, title, len(elements))
                detail_key = self._create_detail_key(structure_key, goal)
                self._detail_cache[detail_key] = analysis
                self._limit_cache(self._detail_cache)

            return analysis

        except Exception as e:
            print(f"⚠️  Vision Agent error: {e}")
            return self._create_fallback_analysis(elements, page_type, structure_confidence)

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
            not e.is_in_viewport,
            e.position.get('y', 0)
        ))

        return relevant[:max_elements]

    def _limit_cache(self, cache: Dict):
        """Ограничивает размер кэша (FIFO)"""
        if len(cache) > self._cache_size:
            to_remove = len(cache) - int(self._cache_size * 0.8)
            for _ in range(to_remove):
                cache.pop(next(iter(cache)))

    def clear_cache(self):
        """Очищает все кэши"""
        self._structure_cache.clear()
        self._detail_cache.clear()