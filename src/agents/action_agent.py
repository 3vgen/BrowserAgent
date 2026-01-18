"""
Action Agent - оптимизированное принятие решений

Улучшения:
- Более точные промпты с примерами
- Валидация действий
- Предотвращение зацикливания
- История с анализом паттернов
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

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

    @classmethod
    def from_dict(cls, data: Dict) -> 'Action':
        return cls(
            type=data.get('type', 'wait'),
            params=data.get('params', {}),
            reasoning=data.get('reasoning', ''),
            confidence=data.get('confidence', 0.5)
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    def __repr__(self) -> str:
        return f"<Action {self.type} conf={self.confidence:.2f}>"

    def __eq__(self, other) -> bool:
        """Проверка на одинаковые действия (для детекции зацикливания)"""
        if not isinstance(other, Action):
            return False
        return (self.type == other.type and
                self.params == other.params)


class ActionAgent:
    """
    Оптимизированный Action Agent.

    Улучшения:
    1. Более детальные промпты с примерами
    2. Валидация действий перед выполнением
    3. Детекция зацикливания
    4. Адаптивная уверенность на основе истории
    5. Graceful degradation при ошибках
    """

    # Улучшенный промпт с примерами
    SYSTEM_PROMPT = """You are an Action Agent - you decide what browser actions to take to accomplish goals.

You work with Vision Agent who analyzed the page. Use their JSON insights to make smart decisions.  
Vision Agent provides: page_type, relevant_elements, observations, steps, and warnings.

CRITICAL RULES:
1. Use ONLY element IDs from relevant_elements provided by Vision Agent.
2. Choose ONE action per response.
3. Think step-by-step: what brings you closer to the goal?
4. Use "complete" when goal is clearly achieved or partially achieved.
   - If Vision Agent indicates page_type = article AND confidence >= 0.9
     AND content clearly matches the goal, immediately use "complete".
5. Do not repeat failed actions.
6. If stuck, try a different approach or complete with partial result.
7. Consider Vision Agent's "steps" suggestions to guide your decision.
8. Confidence reflects how sure you are this action moves toward the goal (0.0-1.0).

AVAILABLE ACTIONS:
1. navigate    - Go to a URL
   {"type": "navigate", "params": {"url": "https://example.com"}}

2. click       - Click an element by ID
   {"type": "click", "params": {"element_id": "elem_5"}}

3. type        - Type text into an input field
   {"type": "type", "params": {"element_id": "elem_3", "text": "search query"}}

4. press       - Press a keyboard key
   {"type": "press", "params": {"key": "Enter"}}

5. scroll      - Scroll page (visible portion only)
   {"type": "scroll", "params": {"direction": "down", "amount": 500}}

6. wait        - Wait for a few seconds
   {"type": "wait", "params": {"seconds": 2}}

7. complete    - Task is done
   {"type": "complete", "params": {"result": "successfully found article about Python"}}

RESPONSE FORMAT (strict JSON):
{
  "thinking": "analyze current situation and what needs to happen",
  "action": {
    "type": "click",
    "params": {"element_id": "elem_5"}
  },
  "reasoning": "why this specific action helps achieve the goal",
  "confidence": 0.85
}

EXAMPLES:

Example 1 - Article clearly matches goal:
Goal: "Find Wikipedia article 'Король и Шут'"
Vision: {
  "page_type": "article",
  "relevant_elements": ["link1", "link2"],
  "observations": ["Title matches search query"],
  "steps": ["Goal achieved — content matches search query. Use 'complete' action"],
  "warnings": [],
  "confidence": 0.90
}
Response:
{
  "thinking": "Article clearly matches user's search query. Task is complete.",
  "action": {"type": "complete", "params": {"result": "Successfully found article 'Король и Шут'"}},
  "reasoning": "Vision Agent indicates high confidence article matches goal",
  "confidence": 0.95
}

Return ONLY valid JSON, no markdown, no extra text.
"""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        max_history: int = 10,
        loop_detection_window: int = 3
    ):
        """
        Args:
            llm_provider: LLM провайдер
            max_history: Максимум действий в истории для контекста
            loop_detection_window: Окно для детекции зацикливания
        """
        self.llm = llm_provider
        self.max_history = max_history
        self.loop_detection_window = loop_detection_window

        self.action_history: List[Action] = []
        self.failed_actions: List[Action] = []

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Надёжный парсинг JSON (аналогично Vision Agent)"""
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

        # Стратегия 3: Извлекаем между { }
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
        """
        Валидирует действие перед выполнением.

        Returns:
            (is_valid, error_message)
        """
        if 'action' not in data:
            return False, "Missing 'action' field"

        action = data['action']

        if 'type' not in action:
            return False, "Missing action type"

        if 'params' not in action:
            return False, "Missing action params"

        action_type = action['type']
        params = action['params']

        # Валидация специфичная для типа действия
        if action_type == 'click':
            if 'element_id' not in params:
                return False, "click requires element_id"

            elem_id = params['element_id']
            if elem_id not in available_element_ids:
                return False, f"element_id {elem_id} not in available elements"

        elif action_type == 'type':
            if 'element_id' not in params:
                return False, "type requires element_id"
            if 'text' not in params:
                return False, "type requires text"

            elem_id = params['element_id']
            if elem_id not in available_element_ids:
                return False, f"element_id {elem_id} not in available elements"

        elif action_type == 'navigate':
            if 'url' not in params:
                return False, "navigate requires url"

            url = params['url']
            if not url.startswith(('http://', 'https://')):
                return False, "url must start with http:// or https://"

        elif action_type == 'press':
            if 'key' not in params:
                params['key'] = 'Enter'  # Default

        elif action_type == 'scroll':
            if 'direction' not in params:
                params['direction'] = 'down'  # Default

        elif action_type == 'wait':
            if 'seconds' not in params:
                params['seconds'] = 2  # Default

        elif action_type == 'complete':
            if 'result' not in params:
                params['result'] = 'Task completed'  # Default

        else:
            return False, f"Unknown action type: {action_type}"

        return True, ""

    def _detect_loop(self) -> bool:
        """
        Детектирует зацикливание в последних действиях.

        Returns:
            True если обнаружено зацикливание
        """
        if len(self.action_history) < self.loop_detection_window:
            return False

        # Берём последние N действий
        recent = self.action_history[-self.loop_detection_window:]

        # Проверяем на одинаковые действия
        action_types = [a.type for a in recent]
        type_counts = Counter(action_types)

        # Если одно действие повторяется слишком часто
        if any(count >= self.loop_detection_window for count in type_counts.values()):
            # Проверяем что это именно одинаковые действия (не просто тип)
            if len(set(str(a.to_dict()) for a in recent)) <= 2:
                return True

        return False

    def _format_elements_compact(self, elements: List[Element]) -> str:
        """Компактное форматирование элементов"""
        if not elements:
            return "No elements available"

        lines = []
        for elem in elements[:15]:  # Топ 15
            parts = [elem.id, elem.tag.upper()]

            if elem.text:
                text = elem.text[:30].replace('\n', ' ')
                parts.append(f'"{text}"')

            if elem.placeholder:
                parts.append(f'ph:"{elem.placeholder[:20]}"')

            if elem.type:
                parts.append(f't:{elem.type}')

            lines.append(" | ".join(parts))

        if len(elements) > 15:
            lines.append(f"... +{len(elements)-15} more")

        return "\n".join(lines)

    def _format_history_smart(self) -> str:
        """Умное форматирование истории с акцентом на паттерны"""
        if not self.action_history:
            return "No previous actions"

        recent = self.action_history[-self.max_history:]

        lines = []
        for i, action in enumerate(recent, 1):
            # Маркер успеха/неуспеха
            failed = action in self.failed_actions
            marker = "❌" if failed else "✓"

            # Маркер уверенности
            conf_marker = "★" if action.confidence > 0.8 else "~"

            action_desc = f"{action.type}"
            if action.type in ['click', 'type']:
                elem_id = action.params.get('element_id', '?')
                action_desc += f"({elem_id})"

            lines.append(
                f"{i}. {marker}{conf_marker} {action_desc}: {action.reasoning[:40]}"
            )

        # Добавляем предупреждение если есть петля
        if self._detect_loop():
            lines.append("\n⚠️  WARNING: Possible action loop detected!")

        return "\n".join(lines)

    async def decide_action(
        self,
        goal: str,
        vision_analysis: PageAnalysis,
        relevant_elements: List[Element],
        step_number: int,
        max_steps: int
    ) -> Optional[Action]:
        """
        Принимает решение о следующем действии.

        Args:
            goal: Цель
            vision_analysis: Анализ от Vision Agent
            relevant_elements: Отфильтрованные элементы
            step_number: Текущий шаг
            max_steps: Максимум шагов

        Returns:
            Action или None
        """
        # Форматируем элементы
        elements_str = self._format_elements_compact(relevant_elements)
        element_ids = [e.id for e in relevant_elements]

        # Форматируем историю
        history_str = self._format_history_smart()

        # Предупреждение если близко к лимиту
        steps_warning = ""
        if step_number > max_steps * 0.8:
            steps_warning = f"\n⚠️  WARNING: Only {max_steps - step_number} steps remaining! Consider completing soon."

        # Создаём промпт
        user_message = f"""GOAL: {goal}

PROGRESS: Step {step_number}/{max_steps}{steps_warning}

VISION ANALYSIS:
Type: {vision_analysis.page_type} (confidence: {vision_analysis.confidence:.2f})
Context: {vision_analysis.context}

Observations:
{chr(10).join('  • ' + obs for obs in vision_analysis.observations)}

AVAILABLE ELEMENTS:
{elements_str}

ACTION HISTORY:
{history_str}

What action should I take next to achieve the goal?"""

        try:
            # Получаем решение
            response = await self.llm.generate_simple(
                user_message=user_message,
                system_prompt=self.SYSTEM_PROMPT
            )

            # Парсим
            data = self._parse_json_response(response.content)

            if not data:
                print("⚠️  Action Agent: Could not parse JSON")
                return None

            # Показываем размышления
            if 'thinking' in data:
                thinking = data['thinking']
                print(f"💭 {thinking[:120]}{'...' if len(thinking) > 120 else ''}")

            # Валидируем
            is_valid, error = self._validate_action(data, element_ids)
            if not is_valid:
                print(f"⚠️  Action Agent: Invalid action - {error}")
                return None

            # Создаём действие
            action = Action.from_dict({
                **data['action'],
                'reasoning': data.get('reasoning', ''),
                'confidence': data.get('confidence', 0.5)
            })

            # Сохраняем в историю
            self.action_history.append(action)

            # Ограничиваем размер истории
            if len(self.action_history) > self.max_history * 2:
                self.action_history = self.action_history[-self.max_history:]

            return action

        except Exception as e:
            print(f"⚠️  Action Agent error: {e}")
            return None

    def mark_action_failed(self, action: Action):
        """Помечает действие как неуспешное"""
        if action not in self.failed_actions:
            self.failed_actions.append(action)

    def reset_history(self):
        """Сбрасывает историю"""
        self.action_history = []
        self.failed_actions = []

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы агента"""
        if not self.action_history:
            return {"total_actions": 0}

        action_types = Counter(a.type for a in self.action_history)
        avg_confidence = sum(a.confidence for a in self.action_history) / len(self.action_history)

        return {
            "total_actions": len(self.action_history),
            "failed_actions": len(self.failed_actions),
            "success_rate": 1.0 - (len(self.failed_actions) / len(self.action_history)),
            "average_confidence": avg_confidence,
            "action_types": dict(action_types),
            "loop_detected": self._detect_loop()
        }