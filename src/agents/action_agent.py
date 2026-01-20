"""
Action Agent - с улучшенной защитой от ошибок

Критические исправления:
1. Не пытаемся набирать текст в submit/button элементы
2. Строгая детекция зацикливания на ошибках
3. Альтернативные стратегии при повторяющихся ошибках
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
    expected_outcome: str = ""
    subtask_complete: bool = False

    @classmethod
    def from_dict(cls, data: Dict) -> 'Action':
        return cls(
            type=data.get('type', 'wait'),
            params=data.get('params', {}),
            reasoning=data.get('reasoning', ''),
            confidence=data.get('confidence', 0.5),
            expected_outcome=data.get('expected_outcome', ''),
            subtask_complete=data.get('subtask_complete', False)
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    def __repr__(self) -> str:
        complete_marker = " [COMPLETE]" if self.subtask_complete else ""
        return f"<Action {self.type} conf={self.confidence:.2f}{complete_marker}>"

    def signature(self) -> str:
        """Уникальная сигнатура действия"""
        return f"{self.type}:{json.dumps(self.params, sort_keys=True)}"


class ActionAgent:
    """Action Agent с защитой от зацикливания"""

    SYSTEM_PROMPT = """You are an Action Agent deciding browser actions to accomplish SUBTASKS.

CRITICAL ELEMENT TYPE RULES:
1. NEVER use "type" action on elements with type="submit" or type="button"
2. NEVER use "type" action on <button> tags
3. For typing text, ONLY use elements with:
   - tag="input" AND type in ["text", "search", "email", "tel"]
   - tag="textarea"
4. Use "click" for buttons and submit inputs
5. Use "click" for links

If you see an error like "Input of type submit cannot be filled":
- This means you tried to type into a button
- Look for the ACTUAL input field (usually nearby)
- Use "type" action on the input field
- Then "click" the submit button OR "press" Enter key

CRITICAL CONTEXT:
- You receive a CURRENT SUBTASK - focus ONLY on this specific subtask
- You also receive TASK CONTEXT showing what's already done and what's remaining
- Your job is to complete the CURRENT SUBTASK, not the entire goal

Vision Agent provides:
- page_type: type of current page
- relevant_elements: element IDs with priorities
- observations: factual observations about the page
- next_action_hint: optional suggestion
- subtask_achieved: whether Vision Agent thinks subtask is done
- confidence: Vision Agent's confidence

CRITICAL RULES:
1. Focus ONLY on the CURRENT SUBTASK (ignore overall goal and future subtasks)
2. Use ONLY element IDs from relevant_elements
3. Set subtask_complete=true when you believe the current subtask is finished
4. Consider element priorities (higher priority = more relevant)
5. Avoid repeating failed actions
6. If stuck after 3 similar attempts, try DIFFERENT approach or mark complete

WHEN TO SET subtask_complete=true:
- Vision Agent says subtask_achieved=true AND you take a confirming action (like "wait")
- After clicking "Add to cart" for "Add X to cart" subtask
- After pressing Enter for "Search for X" subtask
- After page loads for "Navigate to X" subtask
- When clear visual confirmation appears

AVAILABLE ACTIONS:
- navigate: {"type": "navigate", "params": {"url": "https://..."}}
- click: {"type": "click", "params": {"element_id": "elem_X"}}
- type: {"type": "type", "params": {"element_id": "elem_X", "text": "..."}}
- press: {"type": "press", "params": {"key": "Enter"}}
- scroll: {"type": "scroll", "params": {"direction": "down", "amount": 500}}
- wait: {"type": "wait", "params": {"seconds": 2}}

RESPONSE FORMAT (strict JSON):
{
  "thinking": "analyze CURRENT SUBTASK step-by-step, check element types!",
  "action": {
    "type": "type",
    "params": {"element_id": "elem_5", "text": "search query"}
  },
  "reasoning": "why this helps complete CURRENT SUBTASK",
  "confidence": 0.85,
  "expected_outcome": "what should happen",
  "subtask_complete": false
}

DECISION STRATEGY:
1. Check if Vision Agent says subtask_achieved=true
2. Check element types before deciding to "type"
3. If you see "cannot be filled" error in history - find the REAL input field
4. Prioritize elements with higher priority scores
5. If previous action failed, try alternative approach
6. If stuck: scroll, wait, or navigate directly

EXAMPLES:

Correct:
elem_5 INPUT(text) "search box" → type "query" into elem_5 ✅
elem_3 BUTTON(submit) "Search" → click elem_3 ✅

Wrong:
elem_3 BUTTON(submit) → type "query" into elem_3 ❌ (will fail!)
Instead: find INPUT(text) field first

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
        self.page_visit_count: Counter = Counter()
        self.error_count_by_type: Counter = Counter()  # НОВОЕ: счётчик ошибок по типу

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Надёжный парсинг JSON"""
        if not text or not text.strip():
            return None

        text = text.strip().replace('```json', '').replace('```', '').strip()

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

        return None

    def _validate_action(
        self,
        data: Dict,
        available_element_ids: List[str],
        all_elements: List[Element]  # НОВОЕ: для проверки типов
    ) -> Tuple[bool, str]:
        """Валидация действия с проверкой типов элементов"""
        if 'action' not in data:
            return False, "Missing 'action' field"

        action = data['action']
        if 'type' not in action or 'params' not in action:
            return False, "Invalid action structure"

        action_type = action['type']
        params = action['params']

        # КРИТИЧНО: Проверяем тип action "type"
        if action_type == 'type':
            if 'element_id' not in params:
                return False, "type requires element_id"

            elem_id = params['element_id']
            if elem_id not in available_element_ids:
                return False, f"element_id {elem_id} not available"

            # НОВОЕ: Проверяем что это не кнопка
            elem = next((e for e in all_elements if e.id == elem_id), None)
            if elem:
                if elem.tag == 'button':
                    return False, f"Cannot type into <button> element {elem_id}. Use 'click' instead."

                if elem.tag == 'input' and elem.type in ['submit', 'button', 'reset']:
                    return False, f"Cannot type into input[type='{elem.type}'] {elem_id}. Use 'click' instead."

            if 'text' not in params:
                return False, "type requires text"

        elif action_type == 'click':
            if 'element_id' not in params:
                return False, "click requires element_id"

            elem_id = params['element_id']
            if elem_id not in available_element_ids:
                return False, f"element_id {elem_id} not available"

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

        elif action_type not in ['wait']:
            return False, f"Unknown action type: {action_type}"

        return True, ""

    def _detect_loop(self) -> Tuple[bool, Optional[str]]:
        """Детектирует зацикливание"""
        if len(self.action_history) < self.loop_detection_window:
            return False, None

        recent = list(self.action_history)[-self.loop_detection_window:]
        signatures = [a.signature() for a in recent]

        # Тип 1: Одно и то же действие повторяется
        sig_counts = Counter(signatures)
        if any(count >= 3 for count in sig_counts.values()):
            return True, "repeated_action"

        # Тип 2: Цикл из 2-3 действий
        if len(set(signatures)) <= 2:
            return True, "action_cycle"

        # Тип 3: Все действия одного типа
        action_types = [a.type for a in recent]
        if len(set(action_types)) == 1 and action_types[0] != 'wait':
            return True, "same_type_spam"

        # НОВОЕ: Тип 4: Повторяющиеся ошибки одного типа
        if self.error_count_by_type.most_common(1):
            most_common_error, count = self.error_count_by_type.most_common(1)[0]
            if count >= 3:
                return True, f"repeated_error: {most_common_error}"

        return False, None

    def _format_elements_with_priorities(
        self,
        elements: List[Element],
        priorities: Dict[str, float]
    ) -> str:
        """Форматирует элементы с типами и приоритетами"""
        if not elements:
            return "No elements available"

        sorted_elements = sorted(
            elements,
            key=lambda e: priorities.get(e.id, 0.0),
            reverse=True
        )

        lines = []
        for elem in sorted_elements[:12]:
            priority = priorities.get(elem.id, 0.0)
            priority_marker = "★★★" if priority > 0.8 else "★★" if priority > 0.6 else "★"

            parts = [f"{priority_marker} [{elem.id}]"]

            # КРИТИЧНО: Явно показываем тип
            if elem.tag == 'input':
                if elem.type in ['submit', 'button']:
                    parts.append("BUTTON(submit)")
                elif elem.type in ['text', 'search', 'email']:
                    parts.append("INPUT(text)")
                else:
                    parts.append(f"INPUT({elem.type})")
            elif elem.tag == 'button':
                parts.append("BUTTON")
            elif elem.tag == 'a':
                parts.append("LINK")
            else:
                parts.append(elem.tag.upper())

            if elem.text:
                text = elem.text[:35].replace('\n', ' ').strip()
                if text:
                    parts.append(f'"{text}"')

            if elem.placeholder:
                parts.append(f'ph:"{elem.placeholder[:20]}"')

            lines.append(' '.join(parts))

        return '\n'.join(lines)

    def _format_history_compact(self) -> str:
        """Компактное форматирование истории с ошибками"""
        if not self.action_history:
            return "No previous actions for current subtask"

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
            elif action.type == 'type':
                text = action.params.get('text', '')[:20]
                action_desc += f'("{text}")'

            lines.append(f"{i}. {marker}{conf_marker} {action_desc}")

        # Предупреждения
        is_loop, loop_type = self._detect_loop()
        if is_loop:
            lines.append(f"\n⚠️  LOOP DETECTED: {loop_type}")
            lines.append("   → Try DIFFERENT approach (scroll, navigate directly, or try other elements)!")

        return '\n'.join(lines)

    async def decide_action(
        self,
        current_subtask: str,
        task_context: str,
        vision_analysis: PageAnalysis,
        relevant_elements: List[Element],
        step_number: int,
        max_steps: int,
        current_url: str = ""
    ) -> Optional[Action]:
        """Принимает решение о следующем действии"""
        if current_url:
            self.page_visit_count[current_url] += 1
            if self.page_visit_count[current_url] > 5:
                print(f"⚠️  Visited {current_url} {self.page_visit_count[current_url]} times - consider navigation!")

        elements_str = self._format_elements_with_priorities(
            relevant_elements,
            vision_analysis.element_priorities or {}
        )
        element_ids = [e.id for e in relevant_elements]

        history_str = self._format_history_compact()

        is_loop, loop_type = self._detect_loop()
        loop_warning = ""
        if is_loop:
            loop_warning = f"\n⚠️  CRITICAL: LOOP DETECTED ({loop_type})!\n   You MUST try a DIFFERENT approach!"

            # Сбрасываем счётчики ошибок при детекции цикла
            if "repeated_error" in str(loop_type):
                self.error_count_by_type.clear()

        steps_remaining = max_steps - step_number
        steps_warning = ""
        if steps_remaining <= 3:
            steps_warning = f"\n⚠️  Only {steps_remaining} steps left for this subtask!"

        hint_section = ""
        if vision_analysis.next_action_hint:
            hint_section = f"\nVision Agent suggests: {vision_analysis.next_action_hint}"

        vision_complete = ""
        if vision_analysis.subtask_achieved:
            vision_complete = f"\n✓ Vision Agent signals: subtask appears ACHIEVED!"

        user_message = f"""{task_context}

{'-' * 60}

VISION ANALYSIS:
- Page type: {vision_analysis.page_type}
- Confidence: {vision_analysis.confidence:.2f}
- Subtask achieved: {vision_analysis.subtask_achieved}{vision_complete}
- Context: {vision_analysis.context}{hint_section}

Observations:
{chr(10).join('  • ' + obs for obs in vision_analysis.observations)}

AVAILABLE ELEMENTS (with explicit types!):
{elements_str}

RECENT ACTIONS:
{history_str}

Step {step_number}/{max_steps}{steps_warning}{loop_warning}

Remember: Check element TYPES before deciding action!
- Use "type" ONLY for INPUT(text) elements
- Use "click" for BUTTON or BUTTON(submit) elements

Decide next action to complete the CURRENT SUBTASK."""

        try:
            response = await self.llm.generate_simple(
                user_message=user_message,
                system_prompt=self.SYSTEM_PROMPT
            )

            data = self._parse_json_response(response.content)
            if not data:
                print("⚠️  Action Agent: JSON parse failed")
                return self._create_fallback_action(vision_analysis, relevant_elements)

            if 'thinking' in data:
                thinking = data['thinking'][:150]
                print(f"💭 {thinking}{'...' if len(data['thinking']) > 150 else ''}")

            # Валидация с проверкой типов
            is_valid, error = self._validate_action(data, element_ids, relevant_elements)
            if not is_valid:
                print(f"⚠️  Invalid action: {error}")

                # Запоминаем тип ошибки
                if "Cannot type into" in error:
                    self.error_count_by_type["type_into_button"] += 1

                return self._create_fallback_action(vision_analysis, relevant_elements)

            action = Action.from_dict({
                **data['action'],
                'reasoning': data.get('reasoning', ''),
                'confidence': data.get('confidence', 0.5),
                'expected_outcome': data.get('expected_outcome', ''),
                'subtask_complete': data.get('subtask_complete', False)
            })

            if action.subtask_complete:
                print(f"✓ Action Agent signals: SUBTASK COMPLETE")

            if action in self.failed_actions:
                print("⚠️  Attempting previously failed action, trying fallback")
                return self._create_fallback_action(vision_analysis, relevant_elements)

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
        """Создаёт fallback действие"""
        if vision_analysis.subtask_achieved:
            return Action(
                type='wait',
                params={'seconds': 1},
                reasoning='Vision Agent confirms subtask achieved',
                confidence=0.8,
                subtask_complete=True
            )

        # Ищем НАСТОЯЩЕЕ input поле
        text_inputs = [
            e for e in elements
            if e.tag == 'input' and e.type in ['text', 'search', 'email', 'tel']
        ]

        if text_inputs and vision_analysis.element_priorities:
            # Берём input с наивысшим приоритетом
            top_input = max(
                text_inputs,
                key=lambda e: vision_analysis.element_priorities.get(e.id, 0.0)
            )

            return Action(
                type='scroll',
                params={'direction': 'down', 'amount': 300},
                reasoning='Fallback: scrolling to reveal more elements',
                confidence=0.3
            )

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
        self.error_count_by_type.clear()

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
            "page_revisits": dict(self.page_visit_count.most_common(3)),
            "error_types": dict(self.error_count_by_type)
        }