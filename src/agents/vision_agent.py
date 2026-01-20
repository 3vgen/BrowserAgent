"""
Vision Agent - HOTFIX версия

Критические исправления:
1. Фильтрация submit/button элементов для type action
2. Приоритет REAL input полям над кнопками
3. Явное указание типов элементов в промпте
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
    subtask_achieved: bool = False

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
    Vision Agent с улучшенной фильтрацией элементов.
    """

    STRUCTURE_PROMPT = """Analyze page structure and type. Return JSON:
{
  "page_type": "search_page|search_results|article|form|product|auth|email_inbox|email_list|vacancy_list|vacancy_detail|profile|cart|checkout|other",
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
- cart: shopping cart page
- checkout: checkout/payment page
- product: single product page

Return ONLY JSON."""

    DETAIL_PROMPT = """You are Vision Agent analyzing page elements for a SPECIFIC SUBTASK.

CRITICAL ELEMENT TYPE AWARENESS:
- INPUT fields: type="text", type="search", type="email" - CAN be typed into
- BUTTON elements: type="submit", type="button", tag="button" - CANNOT be typed into (only clicked)
- LINKS: tag="a" - can be clicked
- SELECT: tag="select" - dropdowns

NEVER suggest typing into buttons or submit inputs!

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
  "next_action_hint": "Optional: suggest what type of action might help",
  "subtask_achieved": false,
  "confidence": 0.0-1.0,
  "context": "Additional context for Action Agent"
}

WHEN TO SET subtask_achieved=true:
- Current subtask: "Navigate to X" → page URL/title matches X
- Current subtask: "Search for X" → search results for X are visible
- Current subtask: "Add X to cart" → cart shows X was added OR "added to cart" confirmation visible
- Current subtask: "Open article about X" → article content about X is displayed
- Current subtask: "Find vacancy for X" → vacancy details about X are shown

Guidelines:
1. Identify 3-10 elements most relevant to the CURRENT SUBTASK
2. Assign priority based on:
   - Semantic relevance to CURRENT SUBTASK
   - Element type appropriateness (input > button for typing tasks)
   - Visual prominence
   - Interactivity
3. For search tasks: prioritize ACTUAL input fields (type="text" or "search"), NOT submit buttons!
4. Observations should be factual about what's visible NOW
5. Check if current subtask is already achieved on this page

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
        """Ключ для кэша детального анализа"""
        key_string = f"{structure_key}|{goal}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Надёжный парсинг JSON"""
        if not text or not text.strip():
            return None

        text = text.strip()
        text = text.replace('```json', '').replace('```', '').strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

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
        if 'search' in stats['input_types'] or (stats['inputs'] == 1 and stats['buttons'] >= 1):
            return {'page_type': 'search_page', 'confidence': 0.7}

        if stats['links'] > 10 and stats['lists'] > 0:
            return {'page_type': 'search_results', 'confidence': 0.6}

        if stats['inputs'] >= 3 and stats['forms'] >= 1:
            return {'page_type': 'form', 'confidence': 0.7}

        if stats['links'] < 5 and stats['inputs'] == 0:
            return {'page_type': 'article', 'confidence': 0.5}

        return {'page_type': 'other', 'confidence': 0.3}

    async def _analyze_structure(
        self,
        url: str,
        title: str,
        elements: List[Element]
    ) -> Dict:
        """Фаза 1: Быстрый структурный анализ страницы"""
        cache_key = self._create_structure_key(url, title, len(elements))
        if cache_key in self._structure_cache:
            return self._structure_cache[cache_key]

        stats = self._collect_element_stats(elements)

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
                self._structure_cache[cache_key] = data
                self._limit_cache(self._structure_cache)
                return data

        except Exception as e:
            print(f"⚠️  Structure analysis failed: {e}")

        return self._heuristic_structure_analysis(stats)

    def _format_elements_for_detail_analysis(
        self,
        elements: List[Element],
        page_type: str
    ) -> str:
        """
        Форматирует элементы с ЯВНЫМ указанием типов.

        КРИТИЧНО: Показываем разницу между input и button!
        """
        if page_type == 'search_page':
            priority_tags = {'input', 'button', 'form'}
        elif page_type == 'search_results':
            priority_tags = {'a', 'button'}
        elif page_type in ['article', 'vacancy_detail']:
            priority_tags = {'h1', 'h2', 'h3', 'a', 'button'}
        elif page_type in ['email_inbox', 'email_list', 'vacancy_list']:
            priority_tags = {'a', 'button', 'input'}
        elif page_type in ['cart', 'checkout']:
            priority_tags = {'button', 'input', 'a'}
        elif page_type == 'product':
            priority_tags = {'button', 'input', 'select'}
        else:
            priority_tags = {'input', 'button', 'a'}

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
            parts = [f"[{elem.id}]"]

            # КРИТИЧНО: Явно показываем тип элемента
            if elem.tag == 'input':
                if elem.type in ['submit', 'button']:
                    parts.append(f"BUTTON(submit)")  # ⚠️ Это кнопка, не input!
                elif elem.type in ['text', 'search', 'email', 'tel']:
                    parts.append(f"INPUT(text)")  # ✅ Это настоящий input
                else:
                    parts.append(f"INPUT({elem.type})")
            elif elem.tag == 'button':
                parts.append("BUTTON")
            elif elem.tag == 'a':
                parts.append("LINK")
            elif elem.tag == 'select':
                parts.append("SELECT")
            else:
                parts.append(elem.tag.upper())

            if elem.text:
                text = elem.text[:40].replace('\n', ' ').strip()
                if text:
                    parts.append(f'"{text}"')

            if elem.placeholder:
                parts.append(f'placeholder="{elem.placeholder[:25]}"')

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
        """Создаёт fallback анализ с умной фильтрацией"""
        # КРИТИЧНО: Приоритизируем НАСТОЯЩИЕ input поля
        interactive = []

        # Сначала ищем настоящие input поля
        for e in elements:
            if e.is_in_viewport and e.tag == 'input' and e.type in ['text', 'search', 'email', 'tel']:
                interactive.append(e)

        # Потом кнопки и ссылки
        for e in elements:
            if e.is_in_viewport and e.tag in ['button', 'a']:
                if e not in interactive:
                    interactive.append(e)

        interactive = interactive[:10]

        priorities = {}
        for i, elem in enumerate(interactive):
            base_priority = 1.0 - (i * 0.1)

            # КРИТИЧНО: Настоящие input поля имеют ВЫСШИЙ приоритет
            if elem.tag == 'input' and elem.type in ['text', 'search']:
                base_priority += 0.2
            elif elem.tag == 'button':
                base_priority += 0.05

            priorities[elem.id] = min(base_priority, 1.0)

        return PageAnalysis(
            page_type=page_type,
            relevant_elements=[e.id for e in interactive],
            observations=["Fallback analysis - using heuristics"],
            confidence=confidence * 0.5,
            context="Heuristic element selection",
            element_priorities=priorities,
            subtask_achieved=False
        )

    async def analyze_page(
        self,
        goal: str,
        url: str,
        title: str,
        elements: List[Element],
        use_cache: bool = True,
        task_context: str = ""
    ) -> PageAnalysis:
        """Двухфазный анализ страницы"""
        structure = await self._analyze_structure(url, title, elements)
        page_type = structure.get('page_type', 'other')
        structure_confidence = structure.get('confidence', 0.5)

        if use_cache and not task_context:
            structure_key = self._create_structure_key(url, title, len(elements))
            detail_key = self._create_detail_key(structure_key, goal)

            if detail_key in self._detail_cache:
                return self._detail_cache[detail_key]

        elements_str = self._format_elements_for_detail_analysis(elements, page_type)

        context_section = f"\n{task_context}\n" if task_context else ""

        user_message = f"""{context_section}PAGE TYPE: {page_type} (confidence: {structure_confidence:.2f})

CURRENT FOCUS: {goal}

ELEMENTS (with explicit types):
{elements_str}

IMPORTANT: Pay attention to element types!
- INPUT(text) or INPUT(search) can be typed into
- BUTTON(submit) or BUTTON cannot be typed into (only clicked)

Identify elements relevant to achieving the current focus and determine if it's already achieved."""

        try:
            response = await self.llm.generate_simple(
                user_message=user_message,
                system_prompt=self.DETAIL_PROMPT
            )

            data = self._parse_json_response(response.content)

            if not data:
                print("⚠️  Vision Agent: JSON parse failed, using fallback")
                return self._create_fallback_analysis(elements, page_type, structure_confidence)

            if not self._validate_detail_response(data):
                print("⚠️  Vision Agent: Invalid response structure")
                return self._create_fallback_analysis(elements, page_type, structure_confidence)

            analysis = PageAnalysis(
                page_type=page_type,
                relevant_elements=data.get('relevant_elements', []),
                observations=data.get('observations', []),
                confidence=min(data.get('confidence', 0.5), structure_confidence),
                context=data.get('context', ''),
                next_action_hint=data.get('next_action_hint'),
                element_priorities=data.get('element_priorities', {}),
                subtask_achieved=data.get('subtask_achieved', False),
                raw_response=response.content
            )

            if analysis.subtask_achieved:
                print(f"✓ Vision Agent: subtask appears to be achieved")

            if use_cache and not task_context:
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
        Умная фильтрация с приоритетом настоящим input полям
        """
        relevant = []
        relevant_id_set = set(relevant_ids)

        # Шаг 1: Берём релевантные элементы
        for elem in all_elements:
            if elem.id in relevant_id_set:
                relevant.append(elem)

        # Шаг 2: Если мало релевантных, добавляем НАСТОЯЩИЕ input поля
        if len(relevant) < 5:
            for elem in all_elements:
                if (elem.is_in_viewport and
                        elem.tag == 'input' and
                        elem.type in ['text', 'search', 'email', 'tel'] and
                        elem not in relevant and
                        len(relevant) < max_elements):
                    relevant.append(elem)

        # Шаг 3: Добавляем другие интерактивные элементы
        if len(relevant) < 5:
            interactive_tags = {'button', 'a', 'select'}

            for elem in all_elements:
                if (elem.is_in_viewport and
                        elem.tag in interactive_tags and
                        elem not in relevant and
                        len(relevant) < max_elements):
                    relevant.append(elem)

        # Шаг 4: Любые видимые
        if len(relevant) < 3:
            for elem in all_elements:
                if (elem.is_in_viewport and
                        elem not in relevant and
                        len(relevant) < max_elements):
                    relevant.append(elem)

        # Сортируем: настоящие input первыми
        relevant.sort(key=lambda e: (
            not (e.tag == 'input' and e.type in ['text', 'search']),
            not e.is_in_viewport,
            e.position.get('y', 0)
        ))

        return relevant[:max_elements]

    def _limit_cache(self, cache: Dict):
        """Ограничивает размер кэша"""
        if len(cache) > self._cache_size:
            to_remove = len(cache) - int(self._cache_size * 0.8)
            for _ in range(to_remove):
                cache.pop(next(iter(cache)))

    def clear_cache(self):
        """Очищает все кэши"""
        self._structure_cache.clear()
        self._detail_cache.clear()