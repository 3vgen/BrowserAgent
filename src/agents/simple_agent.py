"""
Simple Agent - первый автономный AI-агент
Анализирует страницу и выполняет одно действие за раз
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.llm.base import BaseLLMProvider, Message
from src.browser.manager import BrowserManager


@dataclass
class Action:
    """Действие которое нужно выполнить"""
    type: str  # navigate, click, type, scroll, wait, complete
    params: Dict[str, Any]
    reasoning: str = ""  # Почему агент выбрал это действие

    @classmethod
    def from_dict(cls, data: Dict) -> 'Action':
        """Создаёт Action из словаря"""
        return cls(
            type=data.get('type', 'wait'),
            params=data.get('params', {}),
            reasoning=data.get('reasoning', '')
        )


class SimpleAgent:
    """
    Простой AI-агент для управления браузером.

    Работает в цикле:
    1. Получает текущее состояние страницы
    2. Спрашивает LLM что делать дальше
    3. Выполняет действие
    4. Повторяет пока задача не выполнена
    """

    # Системный промпт для агента
    SYSTEM_PROMPT = """You are a web browser automation agent. Your job is to help users accomplish tasks on websites.

You can perform these actions:
- navigate: {"type": "navigate", "params": {"url": "https://..."}}
- click: {"type": "click", "params": {"element_id": "elem_X"}}
- type: {"type": "type", "params": {"element_id": "elem_X", "text": "..."}}
- scroll: {"type": "scroll", "params": {"direction": "down"}}
- wait: {"type": "wait", "params": {"seconds": 2}}
- complete: {"type": "complete", "params": {"result": "task completed successfully"}}

IMPORTANT RULES:
1. You can ONLY interact with elements by their ID (elem_0, elem_1, etc)
2. DO NOT use CSS selectors or XPath
3. Choose ONE action at a time
4. Think step-by-step
5. When the task is done, use the "complete" action

Your response MUST be valid JSON in this format:
{
  "thinking": "your reasoning about the current situation",
  "action": {
    "type": "...",
    "params": {...}
  },
  "reasoning": "why you chose this action"
}

Return ONLY the JSON object, no other text."""

    def __init__(
            self,
            llm_provider: BaseLLMProvider,
            browser: BrowserManager,
            max_steps: int = 20
    ):
        """
        Args:
            llm_provider: LLM провайдер (Ollama, Claude, etc)
            browser: Менеджер браузера
            max_steps: Максимум шагов для одной задачи
        """
        self.llm = llm_provider
        self.browser = browser
        self.max_steps = max_steps

        # История действий
        self.action_history = []

    def _parse_llm_response(self, response_text: str) -> Optional[Dict]:
        """
        Парсит JSON из ответа LLM.

        LLM иногда добавляет текст вокруг JSON, поэтому ищем фигурные скобки.
        """
        try:
            # Убираем markdown code blocks если есть
            text = response_text.strip()
            text = text.replace('```json', '').replace('```', '')

            # Ищем JSON
            start = text.find('{')
            end = text.rfind('}') + 1

            if 0 <= start < end:
                json_str = text[start:end]
                data = json.loads(json_str)
                return data

            return None

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"Response was: {response_text[:200]}")
            return None

    async def _decide_next_action(
            self,
            goal: str,
            page_state: Dict[str, Any],
            step_number: int
    ) -> Optional[Action]:
        """
        Спрашивает LLM какое действие выполнить следующим.

        Args:
            goal: Цель которую нужно достичь
            page_state: Текущее состояние страницы
            step_number: Номер текущего шага

        Returns:
            Action или None если не удалось распарсить
        """
        # Формируем промпт
        user_message = f"""GOAL: {goal}

CURRENT SITUATION:
- Step: {step_number}/{self.max_steps}
- URL: {page_state['url']}
- Page title: {page_state['title']}

{page_state['elements_formatted']}

PREVIOUS ACTIONS:
{self._format_history()}

What should I do next to achieve the goal?"""

        # Получаем ответ от LLM
        print(f"\n🤔 Asking LLM for decision...")

        response = await self.llm.generate_simple(
            user_message=user_message,
            system_prompt=self.SYSTEM_PROMPT
        )

        # Парсим ответ
        data = self._parse_llm_response(response.content)

        if not data:
            print(f"❌ Could not parse LLM response")
            return None

        # Показываем размышления агента
        if 'thinking' in data:
            print(f"💭 Thinking: {data['thinking']}")

        # Извлекаем действие
        if 'action' not in data:
            print(f"❌ No action in response")
            return None

        action = Action.from_dict({
            **data['action'],
            'reasoning': data.get('reasoning', '')
        })

        return action

    def _format_history(self) -> str:
        """Форматирует историю действий для промпта"""
        if not self.action_history:
            return "No previous actions"

        # Показываем последние 3 действия
        recent = self.action_history[-3:]
        lines = []

        for i, action in enumerate(recent, 1):
            lines.append(f"{i}. {action.type} - {action.reasoning[:60]}")

        return "\n".join(lines)

    async def _execute_action(self, action: Action) -> Dict[str, Any]:
        """
        Выполняет действие в браузере.

        Args:
            action: Действие для выполнения

        Returns:
            Результат выполнения
        """
        action_type = action.type
        params = action.params

        print(f"\n⚡ Executing: {action_type}")
        if action.reasoning:
            print(f"   Why: {action.reasoning}")

        # Навигация
        if action_type == "navigate":
            url = params.get("url")
            return await self.browser.navigate(url)

        # Клик
        elif action_type == "click":
            element_id = params.get("element_id")
            return await self.browser.click(element_id)

        # Ввод текста
        elif action_type == "type":
            element_id = params.get("element_id")
            text = params.get("text")
            return await self.browser.type_text(element_id, text)

        # Прокрутка
        elif action_type == "scroll":
            direction = params.get("direction", "down")
            return await self.browser.scroll(direction)

        # Ожидание
        elif action_type == "wait":
            seconds = params.get("seconds", 2)
            return await self.browser.wait(seconds)

        # Завершение
        elif action_type == "complete":
            return {
                "success": True,
                "completed": True,
                "result": params.get("result", "Task completed")
            }

        else:
            return {
                "success": False,
                "error": f"Unknown action type: {action_type}"
            }

    async def execute_task(self, goal: str, start_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Выполняет задачу автономно.

        Args:
            goal: Описание задачи
            start_url: Начальный URL (опционально)

        Returns:
            Результат выполнения задачи
        """
        print(f"\n{'=' * 80}")
        print(f"🎯 GOAL: {goal}")
        print(f"{'=' * 80}\n")

        # Сбрасываем историю
        self.action_history = []

        # Переходим на стартовый URL если указан
        if start_url:
            print(f"🌐 Starting at: {start_url}")
            result = await self.browser.navigate(start_url)
            if not result['success']:
                return {
                    "success": False,
                    "error": f"Failed to navigate to {start_url}: {result.get('error')}"
                }

        # Основной цикл
        for step in range(1, self.max_steps + 1):
            print(f"\n{'─' * 80}")
            print(f"📍 STEP {step}/{self.max_steps}")
            print(f"{'─' * 80}")

            # Получаем состояние страницы
            page_state = await self.browser.get_page_state()
            print(f"📄 Page: {page_state['title']}")
            print(f"🔗 URL: {page_state['url']}")

            # Решаем что делать
            action = await self._decide_next_action(goal, page_state, step)

            if not action:
                return {
                    "success": False,
                    "error": "Failed to decide next action",
                    "steps_completed": step
                }

            # Сохраняем в историю
            self.action_history.append(action)

            # Выполняем действие
            result = await self._execute_action(action)

            # Проверяем результат
            if not result.get('success'):
                print(f"\n❌ Action failed: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get('error'),
                    "steps_completed": step
                }

            print(f"✅ Action completed")

            # Проверяем завершение
            if result.get('completed'):
                print(f"\n{'=' * 80}")
                print(f"✅ TASK COMPLETED!")
                print(f"📋 Result: {result.get('result')}")
                print(f"📊 Steps: {step}/{self.max_steps}")
                print(f"{'=' * 80}\n")

                return {
                    "success": True,
                    "result": result.get('result'),
                    "steps_completed": step
                }

            # Небольшая пауза между шагами
            await self.browser.wait(1)

        # Достигли максимума шагов
        print(f"\n⚠️  Maximum steps ({self.max_steps}) reached")
        return {
            "success": False,
            "error": "Maximum steps reached",
            "steps_completed": self.max_steps
        }


# Пример использования
if __name__ == "__main__":
    import asyncio
    from src.llm.ollama_provider import create_ollama_provider
    from src.browser.manager import BrowserManager


    async def test_simple_agent():
        """Тестовая функция"""

        print("\n" + "=" * 80)
        print("SIMPLE AGENT TEST")
        print("=" * 80 + "\n")

        # Создаём LLM провайдер
        print("📍 Setting up LLM provider...")
        llm = await create_ollama_provider(model="qwen2.5:7b")
        print("✅ LLM ready\n")

        # Создаём браузер
        print("📍 Starting browser...")
        browser = BrowserManager(headless=False, slow_mo=500)
        await browser.start()
        print("✅ Browser ready\n")

        # Создаём агента
        agent = SimpleAgent(
            llm_provider=llm,
            browser=browser,
            max_steps=15
        )

        try:
            # Тест 1: Простой поиск
            print("\n" + "=" * 80)
            print("TEST 1: Simple search on Google")
            print("=" * 80)

            result = await agent.execute_task(
                goal="Search for 'AI agents' on Google and show me the results",
                start_url="https://google.com"
            )

            if result['success']:
                print(f"\n✅ Test 1 passed! Result: {result['result']}")
            else:
                print(f"\n❌ Test 1 failed: {result.get('error')}")

            # Даём время посмотреть результат
            await asyncio.sleep(3)

            # Тест 2: Wikipedia
            print("\n" + "=" * 80)
            print("TEST 2: Find article on Wikipedia")
            print("=" * 80)

            result = await agent.execute_task(
                goal="Go to Wikipedia, search for 'Python programming', and open the article",
                start_url="https://wikipedia.org"
            )

            if result['success']:
                print(f"\n✅ Test 2 passed! Result: {result['result']}")
            else:
                print(f"\n❌ Test 2 failed: {result.get('error')}")

        finally:
            await browser.close()
            await llm.close()


    asyncio.run(test_simple_agent())