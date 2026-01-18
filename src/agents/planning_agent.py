"""
Planning Agent - разбивает сложные задачи на простые шаги

Решает проблему: Агенты теряются в многошаговых задачах
Решение: Создаём план заранее, следуем ему пошагово
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..llm.base import BaseLLMProvider


class StepStatus(Enum):
    """Статус шага плана"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """Один шаг в плане"""
    step_number: int
    description: str
    success_criteria: str
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0

    def to_dict(self) -> Dict:
        return {
            "step": self.step_number,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "status": self.status.value,
            "attempts": self.attempts
        }


@dataclass
class TaskPlan:
    """План выполнения задачи"""
    goal: str
    steps: List[PlanStep]
    current_step_index: int = 0

    def get_current_step(self) -> Optional[PlanStep]:
        """Возвращает текущий шаг"""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def mark_step_completed(self):
        """Помечает текущий шаг как завершённый"""
        current = self.get_current_step()
        if current:
            current.status = StepStatus.COMPLETED
            self.current_step_index += 1

    def mark_step_failed(self):
        """Помечает текущий шаг как проваленный"""
        current = self.get_current_step()
        if current:
            current.status = StepStatus.FAILED
            current.attempts += 1

    def is_completed(self) -> bool:
        """Проверяет завершён ли весь план"""
        return self.current_step_index >= len(self.steps)

    def get_progress_summary(self) -> str:
        """Возвращает краткую сводку прогресса"""
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        total = len(self.steps)

        lines = [f"Progress: {completed}/{total} steps completed\n"]

        for step in self.steps:
            status_emoji = {
                StepStatus.PENDING: "⏸️",
                StepStatus.IN_PROGRESS: "▶️",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️"
            }

            emoji = status_emoji.get(step.status, "?")
            lines.append(f"{emoji} Step {step.step_number}: {step.description}")

        return "\n".join(lines)


class PlanningAgent:
    """
    Planning Agent - разбивает задачи на шаги.

    Почему это помогает:
    1. Агент знает куда он идёт (есть план)
    2. Агент знает когда закончить (план завершён)
    3. Агент не теряется (следует плану)
    4. Можно track прогресс (шаг X из Y)
    """

    SYSTEM_PROMPT = """You are a Planning Agent - expert at breaking down complex web tasks into simple steps.

Your job: Given a user goal, create a detailed step-by-step plan.

IMPORTANT RULES:
1. Each step should be SIMPLE and ATOMIC (one clear action)
2. Steps should be SEQUENTIAL (step N depends on step N-1)
3. Include clear SUCCESS CRITERIA for each step
4. Plan should have 3-8 steps (not too few, not too many)
5. Last step should ALWAYS be verification/completion

GOOD STEPS:
✅ "Navigate to Google homepage"
✅ "Type 'Python programming' in search box"
✅ "Click search button"
✅ "Verify results are displayed"

BAD STEPS:
❌ "Search for Python" (too vague - what are the sub-actions?)
❌ "Find information" (not specific)
❌ "Do research" (too broad)

Response format (strict JSON):
{
  "thinking": "analyze the task and think about what steps are needed",
  "steps": [
    {
      "step": 1,
      "description": "Navigate to google.com",
      "success_criteria": "Google homepage is loaded with search box visible"
    },
    {
      "step": 2,
      "description": "Type search query in search box",
      "success_criteria": "Query text is visible in search input field"
    },
    {
      "step": 3,
      "description": "Click search button or press Enter",
      "success_criteria": "Search results page is displayed"
    },
    {
      "step": 4,
      "description": "Verify results are shown",
      "success_criteria": "At least 5 search results are visible on page"
    }
  ],
  "estimated_actions": 6,
  "completion_criteria": "Search results for the query are successfully displayed"
}

EXAMPLES:

Example 1:
Goal: "Find Python tutorial on Wikipedia"
Steps:
1. Navigate to wikipedia.org → Homepage loaded
2. Locate search box → Search box is visible
3. Type "Python programming" → Text entered
4. Submit search → Search results shown
5. Click on Python article → Article page opened
6. Verify article content → Article about Python is displayed

Example 2:
Goal: "Search for 'AI agents' on Google"
Steps:
1. Navigate to google.com → Google homepage loaded
2. Find search input → Search box visible
3. Type "AI agents" → Query entered
4. Execute search → Results page loaded
5. Verify results → Search results visible

Return ONLY valid JSON, no markdown, no extra text."""

    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Парсит JSON из ответа"""
        if not text:
            return None

        text = text.strip()

        # Удаляем markdown
        text = text.replace('```json', '').replace('```', '')

        try:
            return json.loads(text.strip())
        except:
            pass

        # Ищем JSON между { }
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except:
            pass

        return None

    async def create_plan(self, goal: str) -> Optional[TaskPlan]:
        """
        Создаёт план для достижения цели.

        Args:
            goal: Цель пользователя

        Returns:
            TaskPlan или None если не удалось создать
        """
        print(f"\n📋 Planning Agent creating plan for: {goal}")

        user_message = f"""Goal: {goal}

Create a detailed step-by-step plan to accomplish this goal."""

        try:
            response = await self.llm.generate_simple(
                user_message=user_message,
                system_prompt=self.SYSTEM_PROMPT
            )

            data = self._parse_json(response.content)

            if not data or 'steps' not in data:
                print("⚠️  Planning Agent: Could not parse plan")
                return None

            # Показываем размышления
            if 'thinking' in data:
                print(f"💭 {data['thinking'][:150]}...")

            # Создаём шаги
            steps = []
            for step_data in data['steps']:
                step = PlanStep(
                    step_number=step_data.get('step', len(steps) + 1),
                    description=step_data.get('description', ''),
                    success_criteria=step_data.get('success_criteria', '')
                )
                steps.append(step)

            plan = TaskPlan(goal=goal, steps=steps)

            print(f"\n✅ Plan created with {len(steps)} steps:")
            print(plan.get_progress_summary())

            if 'completion_criteria' in data:
                print(f"\n🎯 Completion criteria: {data['completion_criteria']}")

            return plan

        except Exception as e:
            print(f"⚠️  Planning Agent error: {e}")
            return None

    async def should_step_be_complete(
            self,
            step: PlanStep,
            current_situation: str
    ) -> bool:
        """
        Проверяет выполнен ли шаг на основе текущей ситуации.

        Args:
            step: Шаг плана
            current_situation: Описание текущего состояния

        Returns:
            True если шаг выполнен
        """
        verification_prompt = f"""You are checking if a step is completed.

STEP: {step.description}
SUCCESS CRITERIA: {step.success_criteria}

CURRENT SITUATION:
{current_situation}

Question: Is the step completed according to the success criteria?

Response format (JSON):
{{
  "is_completed": true/false,
  "reasoning": "why you think it is or isn't completed",
  "confidence": 0.0-1.0
}}

Be strict - only mark as completed if success criteria is clearly met."""

        try:
            response = await self.llm.generate_simple(
                user_message=verification_prompt,
                system_prompt="You are a verification agent. Answer only in JSON."
            )

            data = self._parse_json(response.content)

            if data and 'is_completed' in data:
                is_complete = data['is_completed']
                reasoning = data.get('reasoning', '')
                confidence = data.get('confidence', 0.5)

                if confidence < 0.6:
                    # Низкая уверенность - считаем незавершённым
                    return False

                return is_complete

            return False

        except:
            return False


# Тест
if __name__ == "__main__":
    import asyncio
    from ..llm.ollama_provider import create_ollama_provider


    async def test_planning():
        print("\n" + "=" * 80)
        print("PLANNING AGENT TEST")
        print("=" * 80)

        llm = await create_ollama_provider(model="qwen2.5:7b")
        planner = PlanningAgent(llm_provider=llm)

        # Тест 1: Простая задача
        plan = await planner.create_plan(
            "Search for 'autonomous AI agents' on Google"
        )

        if plan:
            print("\n" + "=" * 80)
            print("PLAN CREATED SUCCESSFULLY")
            print("=" * 80)
            print(plan.get_progress_summary())

        # Тест 2: Сложная задача
        print("\n" + "=" * 80)
        plan2 = await planner.create_plan(
            "Go to Wikipedia, search for Python programming, and read the first paragraph"
        )

        if plan2:
            print("\n" + "=" * 80)
            print("COMPLEX PLAN")
            print("=" * 80)
            print(plan2.get_progress_summary())

        await llm.close()


    asyncio.run(test_planning())