"""
Action Agent - оптимизированное принятие решений

Ключевые улучшения:
- Интеграция с приоритетами элементов от Vision Agent
- Умная детекция зацикливания с адаптацией
- Предсказание успешности действия
- Graceful degradation при проблемах
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter, deque

from ..llm.base import BaseLLMProvider
from ..browser.dom_extractor import Element
from .vision_agent import PageAnalysis


@dataclass
class Action:
    """Действие для выполнения"""
    type: str
    params: Dict[str, Any]
    reasoning: str = ""
    confidence: float = 0.0
    expected_outcome: str = ""  # Что ожидаем после действия

    @classmethod
    def from_dict(cls, data: Dict) -> 'Action':
        return cls(
            type=data.get('type', 'wait'),
            params=data.get('params', {}),
            reasoning=data.get('reasoning', ''),
            confidence=data.get('confidence', 0.5),
            expected_outcome=data.get('expected_outcome', '')
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    def __repr__(self) -> str:
        return f"<Action {self.type} conf={self.confidence:.2f}>"

    def signature(self) -> str:
        """Уникальная сигнатура действия для детекции повторов"""
        return f"{self.type}:{json.dumps(self.params, sort_keys=True)}"


class ActionAgent:
    """
    Оптимизированный Action Agent с адаптивным поведением.
    """

    SYSTEM_PROMPT = """You are an Action Agent deciding browser actions to accomplish goals.

You receive analysis from Vision Agent including:
- page_type: type of current page
- relevant_elements: element IDs with priorities
- observations: factual observations about the page
- next_action_hint: optional suggestion
- confidence: Vision Agent's confidence

CRITICAL RULES:
1. Use ONLY element IDs from relevant_elements
2. If Vision Agent suggests page_type="article" AND confidence > 0.85 AND observations indicate goal is achieved → use "complete"
3. Consider element priorities (higher priority = more relevant)
4. Avoid repeating failed actions
5. If stuck after 3 similar attempts, try different approach or complete with partial result
6. expected_outcome helps verify if action succeeded

AVAILABLE ACTIONS:
- navigate: {"type": "navigate", "params": {"url": "https://..."}}
- click: {"type": "click", "params": {"element_id": "elem_X"}}
- type: {"type": "type", "params": {"element_id": "elem_X", "text": "..."}}
- press: {"type": "press", "params": {"key": "Enter"}}
- scroll: {"type": "scroll", "params": {"direction": "down", "amount": 500}}
- wait: {"type": "wait", "params": {"seconds": 2}}
- complete: {"type": "complete", "params": {"result": "description of what was achieved"}}

RESPONSE FORMAT (strict JSON):
{
  "thinking": "analyze situation step-by-step",
  "action": {
    "type": "click",
    "params": {"element_id": "elem_5"}
  },
  "reasoning": "why this action helps achieve goal",
  "confidence": 0.85,
  "expected_outcome": "what should happen after this action"
}

DECISION STRATEGY:
1. Check if goal is already achieved (complete immediately)
2. If Vision Agent has next_action_hint, consider it
3. Prioritize elements with higher priority scores
4. If previous action failed, try alternative approach
5. If making no progress, consider completing with partial result

Return ONLY valid JSON, no markdown."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        max_history: int = 10,
        loop_detection_window: int = 4
    ):
        self.llm = llm_provider
        self.max_history = max_history
        self.loop_detection_window = loop_detection_window

        self.action_history: deque = deque(maxlen=max_history * 2)
        self.failed_actions: List[Action] = []
        self.page_visit_count: Counter = Counter()  # Счётчик посещений страниц

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Надёжный парсинг JSON"""
        if not text or not text.strip():
            return None

        text = text.strip().replace('```json', '').replace('```', '').strip()

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

        return None

    def _validate_action(
        self,
        data: Dict,
        available_element_ids: List[str]
    ) -> Tuple[bool, str]:
        """Валидация действия"""
        if 'action' not in data:
            return False, "Missing 'action' field"

        action = data['action']
        if 'type' not in action or 'params' not in action:
            return False, "Invalid action structure"

        action_type = action['type']
        params = action['params']

        # Валидация по типу
        if action_type in ['click', 'type']:
            if 'element_id' not in params:
                return False, f"{action_type} requires element_id"

            elem_id = params['element_id']
            if elem_id not in available_element_ids:
                return False, f"element_id {elem_id} not available"

        elif action_type == 'type':
            if 'text' not in params:
                return False, "type requires text"

        elif action_type == 'navigate':
            if 'url' not in params:
                return False, "navigate requires url"

            url = params['url']
            if not url.startswith(('http://', 'https://')):
                return False, "invalid URL"

        elif action_type == 'press':
            params.setdefault('key', 'Enter')

        elif action_type == 'scroll':
            params.setdefault('direction', 'down')
            params.setdefault('amount', 500)

        elif action_type == 'wait':
            params.setdefault('seconds', 2)

        elif action_type == 'complete':
            params.setdefault('result', 'Task completed')

        elif action_type not in ['wait', 'complete']:
            return False, f"Unknown action type: {action_type}"

        return True, ""

    def _detect_loop(self) -> Tuple[bool, Optional[str]]:
        """
        Детектирует зацикливание.

        Returns:
            (is_loop, loop_type)
        """
        if len(self.action_history) < self.loop_detection_window:
            return False, None

        recent = list(self.action_history)[-self.loop_detection_window:]
        signatures = [a.signature() for a in recent]

        # Тип 1: Одно и то же действие повторяется
        sig_counts = Counter(signatures)
        if any(count >= 3 for count in sig_counts.values()):
            return True, "repeated_action"

        # Тип 2: Цикл из 2-3 действий (A→B→A→B)
        if len(set(signatures)) <= 2:
            return True, "action_cycle"

        # Тип 3: Все действия одного типа (только клики)
        action_types = [a.type for a in recent]
        if len(set(action_types)) == 1 and action_types[0] != 'wait':
            return True, "same_type_spam"

        return False, None

    def _format_elements_with_priorities(
        self,
        elements: List[Element],
        priorities: Dict[str, float]
    ) -> str:
        """Форматирует элементы с учётом приоритетов"""
        if not elements:
            return "No elements available"

        # Сортируем по приоритету
        sorted_elements = sorted(
            elements,
            key=lambda e: priorities.get(e.id, 0.0),
            reverse=True
        )

        lines = []
        for elem in sorted_elements[:12]:  # Топ 12
            priority = priorities.get(elem.id, 0.0)
            priority_marker = "★★★" if priority > 0.8 else "★★" if priority > 0.6 else "★"

            parts = [f"{priority_marker} [{elem.id}]", elem.tag.upper()]

            if elem.text:
                text = elem.text[:35].replace('\n', ' ').strip()
                if text:
                    parts.append(f'"{text}"')

            if elem.placeholder:
                parts.append(f'ph:"{elem.placeholder[:20]}"')

            if elem.type:
                parts.append(f't:{elem.type}')

            lines.append(' '.join(parts))

        return '\n'.join(lines)

    def _format_history_compact(self) -> str:
        """Компактное форматирование истории"""
        if not self.action_history:
            return "No previous actions"

        recent = list(self.action_history)[-self.max_history:]
        lines = []

        for i, action in enumerate(recent, 1):
            failed = action in self.failed_actions
            marker = "❌" if failed else "✓"

            conf_marker = "★" if action.confidence > 0.8 else "~" if action.confidence > 0.5 else "?"

            action_desc = action.type
            if action.type in ['click', 'type']:
                elem_id = action.params.get('element_id', '?')
                action_desc += f"({elem_id})"

            lines.append(f"{i}. {marker}{conf_marker} {action_desc}")

        # Предупреждения
        is_loop, loop_type = self._detect_loop()
        if is_loop:
            lines.append(f"\n⚠️  LOOP DETECTED: {loop_type}")

        return '\n'.join(lines)

    def _should_complete_early(
        self,
        goal: str,
        vision_analysis: PageAnalysis
    ) -> Tuple[bool, str]:
        """
        Определяет, достигнута ли цель (ранее завершение).

        Returns:
            (should_complete, reason)
        """
        # Высокая уверенность Vision Agent + тип страницы соответствует цели
        if vision_analysis.confidence > 0.85:
            page_type = vision_analysis.page_type

            # Если ищем статью и попали на статью
            if page_type == 'article':
                # Проверяем observations на совпадение с целью
                obs_text = ' '.join(vision_analysis.observations).lower()
                goal_lower = goal.lower()

                # Простая эвристика: есть ли ключевые слова из цели в наблюдениях
                goal_keywords = set(goal_lower.split()) - {'find', 'search', 'look', 'for', 'the', 'a', 'an'}
                if any(keyword in obs_text for keyword in goal_keywords):
                    return True, f"Found article matching goal (confidence: {vision_analysis.confidence:.2f})"

            # Если ищем профиль и попали на профиль
            if page_type == 'profile' and 'profile' in goal.lower():
                return True, f"Reached profile page (confidence: {vision_analysis.confidence:.2f})"

        return False, ""

    async def decide_action(
        self,
        goal: str,
        vision_analysis: PageAnalysis,
        relevant_elements: List[Element],
        step_number: int,
        max_steps: int,
        current_url: str = ""
    ) -> Optional[Action]:
        """
        Принимает решение о следующем действии.
        """
        # Проверяем раннее завершение
        should_complete, complete_reason = self._should_complete_early(goal, vision_analysis)
        if should_complete:
            print(f"✓ Early completion: {complete_reason}")
            return Action(
                type='complete',
                params={'result': complete_reason},
                reasoning=complete_reason,
                confidence=vision_analysis.confidence
            )

        # Отслеживаем посещения страниц
        if current_url:
            self.page_visit_count[current_url] += 1
            if self.page_visit_count[current_url] > 3:
                print(f"⚠️  Visited {current_url} {self.page_visit_count[current_url]} times")

        # Форматируем элементы с приоритетами
        elements_str = self._format_elements_with_priorities(
            relevant_elements,
            vision_analysis.element_priorities or {}
        )
        element_ids = [e.id for e in relevant_elements]

        # История
        history_str = self._format_history_compact()

        # Детекция зацикливания
        is_loop, loop_type = self._detect_loop()
        loop_warning = f"\n⚠️  LOOP DETECTED ({loop_type}): Try different approach or complete!" if is_loop else ""

        # Предупреждение о лимите шагов
        steps_remaining = max_steps - step_number
        steps_warning = ""
        if steps_remaining <= 3:
            steps_warning = f"\n⚠️  Only {steps_remaining} steps left! Consider completing."

        # Подсказка от Vision Agent
        hint_section = ""
        if vision_analysis.next_action_hint:
            hint_section = f"\nVision Agent suggests: {vision_analysis.next_action_hint}"

        # Промпт
        user_message = f"""GOAL: {goal}

PROGRESS: Step {step_number}/{max_steps}{steps_warning}{loop_warning}

VISION ANALYSIS:
- Page type: {vision_analysis.page_type}
- Confidence: {vision_analysis.confidence:.2f}
- Context: {vision_analysis.context}{hint_section}

Observations:
{chr(10).join('  • ' + obs for obs in vision_analysis.observations)}

AVAILABLE ELEMENTS (sorted by priority):
{elements_str}

ACTION HISTORY:
{history_str}

Decide next action to achieve the goal."""

        try:
            response = await self.llm.generate_simple(
                user_message=user_message,
                system_prompt=self.SYSTEM_PROMPT
            )

            data = self._parse_json_response(response.content)
            if not data:
                print("⚠️  Action Agent: JSON parse failed")
                return self._create_fallback_action(vision_analysis, relevant_elements)

            # Показываем размышления
            if 'thinking' in data:
                thinking = data['thinking'][:150]
                print(f"💭 {thinking}{'...' if len(data['thinking']) > 150 else ''}")

            # Валидация
            is_valid, error = self._validate_action(data, element_ids)
            if not is_valid:
                print(f"⚠️  Invalid action: {error}")
                return self._create_fallback_action(vision_analysis, relevant_elements)

            # Создаём действие
            action = Action.from_dict({
                **data['action'],
                'reasoning': data.get('reasoning', ''),
                'confidence': data.get('confidence', 0.5),
                'expected_outcome': data.get('expected_outcome', '')
            })

            # Проверяем на повтор неудачного действия
            if action in self.failed_actions:
                print("⚠️  Attempting previously failed action, trying fallback")
                return self._create_fallback_action(vision_analysis, relevant_elements)

            # Сохраняем в историю
            self.action_history.append(action)

            return action

        except Exception as e:
            print(f"⚠️  Action Agent error: {e}")
            return self._create_fallback_action(vision_analysis, relevant_elements)

    def _create_fallback_action(
        self,
        vision_analysis: PageAnalysis,
        elements: List[Element]
    ) -> Action:
        """
        Создаёт fallback действие на основе эвристик.
        """
        # Если есть элементы с высоким приоритетом, кликаем на первый
        if vision_analysis.element_priorities:
            top_elem = max(
                vision_analysis.element_priorities.items(),
                key=lambda x: x[1]
            )[0]

            return Action(
                type='click',
                params={'element_id': top_elem},
                reasoning='Fallback: clicking highest priority element',
                confidence=0.4
            )

        # Иначе просто ждём
        return Action(
            type='wait',
            params={'seconds': 2},
            reasoning='Fallback: waiting',
            confidence=0.3
        )

    def mark_action_failed(self, action: Action):
        """Помечает действие как неуспешное"""
        if action not in self.failed_actions:
            self.failed_actions.append(action)

    def reset_history(self):
        """Сбрасывает историю"""
        self.action_history.clear()
        self.failed_actions.clear()
        self.page_visit_count.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Статистика работы"""
        if not self.action_history:
            return {"total_actions": 0}

        action_types = Counter(a.type for a in self.action_history)
        avg_confidence = sum(a.confidence for a in self.action_history) / len(self.action_history)

        is_loop, loop_type = self._detect_loop()

        return {
            "total_actions": len(self.action_history),
            "failed_actions": len(self.failed_actions),
            "success_rate": 1.0 - (len(self.failed_actions) / len(self.action_history)) if self.action_history else 0.0,
            "average_confidence": avg_confidence,
            "action_types": dict(action_types),
            "loop_detected": is_loop,
            "loop_type": loop_type,
            "page_revisits": dict(self.page_visit_count.most_common(3))
        }