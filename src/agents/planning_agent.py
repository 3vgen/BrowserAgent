"""
Planning Agent - разбивает сложные задачи на атомарные подзадачи

Ключевые улучшения:
- Декомпозиция с учётом многошаговых сценариев (e-commerce, поиск и т.д.)
- Атомарные шаги (один шаг = одно действие)
- Чёткие критерии успеха
- Интеграция с Vision/Action агентами
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.llm.base import BaseLLMProvider


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
    max_attempts: int = 3
    result: str = ""

    def to_dict(self) -> Dict:
        return {
            "step": self.step_number,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "status": self.status.value,
            "attempts": self.attempts,
            "result": self.result
        }

    def is_failed(self) -> bool:
        """Проверяет провален ли шаг"""
        return self.status == StepStatus.FAILED or self.attempts >= self.max_attempts


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

    def get_next_pending_step(self) -> Optional[PlanStep]:
        """Возвращает следующий ожидающий шаг"""
        for step in self.steps[self.current_step_index:]:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def mark_step_completed(self, result: str = ""):
        """Помечает текущий шаг как завершённый"""
        current = self.get_current_step()
        if current:
            current.status = StepStatus.COMPLETED
            current.result = result
            print(f"✓ Step {current.step_number} completed: {current.description}")
            self.current_step_index += 1

            # Активируем следующий шаг
            next_step = self.get_next_pending_step()
            if next_step:
                next_step.status = StepStatus.IN_PROGRESS
                print(f"→ Starting step {next_step.step_number}: {next_step.description}")

    def mark_step_failed(self, reason: str = ""):
        """Помечает текущий шаг как проваленный"""
        current = self.get_current_step()
        if current:
            current.attempts += 1
            current.result = reason

            if current.is_failed():
                current.status = StepStatus.FAILED
                print(f"✗ Step {current.step_number} failed after {current.attempts} attempts")
            else:
                print(f"⚠️  Step {current.step_number} attempt {current.attempts} failed, retrying...")

    def is_completed(self) -> bool:
        """Проверяет завершён ли весь план"""
        return all(s.status == StepStatus.COMPLETED for s in self.steps)

    def has_failed(self) -> bool:
        """Проверяет есть ли проваленные шаги"""
        return any(s.is_failed() for s in self.steps)

    def get_progress(self) -> Dict[str, int]:
        """Возвращает прогресс выполнения"""
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.is_failed())

        return {
            'total': len(self.steps),
            'completed': completed,
            'failed': failed,
            'in_progress': 1 if self.get_current_step() and self.get_current_step().status == StepStatus.IN_PROGRESS else 0,
            'pending': len(self.steps) - completed - failed - (1 if self.get_current_step() and self.get_current_step().status == StepStatus.IN_PROGRESS else 0),
            'progress_percent': (completed / len(self.steps) * 100) if self.steps else 0
        }

    def get_progress_summary(self) -> str:
        """Возвращает краткую сводку прогресса"""
        progress = self.get_progress()

        lines = [f"Progress: {progress['completed']}/{progress['total']} steps completed ({progress['progress_percent']:.0f}%)\n"]

        for step in self.steps:
            status_emoji = {
                StepStatus.PENDING: "⏸️",
                StepStatus.IN_PROGRESS: "▶️",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️"
            }

            emoji = status_emoji.get(step.status, "?")
            result_info = f" → {step.result}" if step.result else ""
            lines.append(f"{emoji} Step {step.step_number}: {step.description}{result_info}")

        return "\n".join(lines)

    def get_context_for_agents(self) -> str:
        """Форматирует контекст для Vision/Action агентов"""
        if not self.steps:
            return ""

        current = self.get_current_step()
        if not current:
            return "All steps completed"

        progress = self.get_progress()

        lines = [
            f"OVERALL GOAL: {self.goal}",
            f"PROGRESS: {progress['completed']}/{progress['total']} steps completed",
            f"\nCURRENT STEP: {current.description}",
            f"SUCCESS CRITERIA: {current.success_criteria}",
        ]

        # Показываем что уже сделано
        completed_steps = [s for s in self.steps if s.status == StepStatus.COMPLETED]
        if completed_steps:
            lines.append("\nCOMPLETED:")
            for s in completed_steps:
                result = f" → {s.result}" if s.result else ""
                lines.append(f"  ✓ {s.description}{result}")

        # Показываем что осталось
        pending = [s for s in self.steps if s.status == StepStatus.PENDING]
        if pending:
            lines.append(f"\nREMAINING: {len(pending)} steps")

        return '\n'.join(lines)


class PlanningAgent:
    """
    Planning Agent - разбивает задачи на атомарные шаги.

    Ключевое отличие от старой версии:
    - Акцент на АТОМАРНОСТЬ (один шаг = одно действие)
    - Разделение многопредметных задач (купить X и Y → отдельные шаги для каждого)
    - Чёткие критерии успеха для каждого шага
    """

    SYSTEM_PROMPT = """You are a Planning Agent - expert at breaking down complex web tasks into ATOMIC steps.

CRITICAL: Each step must be ATOMIC - one clear, focused action that can be completed independently.

ANTI-PATTERN (what NOT to do):
❌ "Search for BBQ burger and fries" - This is TWO items, should be TWO separate steps!
❌ "Add items to cart" - Which items? Be specific!
❌ "Find information and save it" - Two actions, split them!

CORRECT PATTERN:
✅ "Search for BBQ burger"
✅ "Add BBQ burger to cart"
✅ "Search for fries"
✅ "Add fries to cart"

RULES:
1. ONE action per step (search, click, type, add, navigate)
2. If task involves multiple items → separate step for EACH item
3. For e-commerce: "search X" → "add X to cart" → "search Y" → "add Y to cart"
4. Each step has clear SUCCESS CRITERIA (how to verify it's done)
5. Steps are SEQUENTIAL (later steps may depend on earlier ones)
6. Typical plan: 3-10 steps (not too few, not too many)

Response format (strict JSON):
{
  "thinking": "analyze the task: how many distinct items/actions? what order?",
  "steps": [
    {
      "step": 1,
      "description": "Navigate to Yandex Lavka",
      "success_criteria": "Yandex Lavka homepage is loaded with search visible"
    },
    {
      "step": 2,
      "description": "Search for BBQ burger",
      "success_criteria": "Search results for BBQ burger are displayed"
    },
    {
      "step": 3,
      "description": "Add BBQ burger to cart",
      "success_criteria": "BBQ burger is in cart (cart shows 1 item or confirmation visible)"
    }
  ],
  "estimated_actions": 12,
  "completion_criteria": "All required items are in the cart"
}

EXAMPLES:

Example 1 - E-commerce (IMPORTANT!):
Goal: "Buy BBQ burger and fries on Yandex Lavka"
Thinking: "Two items (BBQ burger, fries) → need separate search and add for each"
Steps:
1. Navigate to Yandex Lavka → Homepage loaded
2. Search for BBQ burger → Search results shown
3. Add BBQ burger to cart → Item in cart
4. Search for fries → Search results shown
5. Add fries to cart → Item in cart
6. Proceed to checkout → Checkout page visible

Example 2 - Information gathering:
Goal: "Find Python tutorial on Wikipedia and save the URL"
Steps:
1. Navigate to wikipedia.org → Homepage loaded
2. Search for "Python programming" → Search results shown
3. Click on Python article → Article page opened
4. Copy article URL → URL copied to clipboard

Example 3 - Simple search:
Goal: "Search for AI agents on Google"
Steps:
1. Navigate to google.com → Google homepage loaded
2. Type "AI agents" in search → Query entered
3. Press Enter or click search → Results page loaded
4. Verify results → At least 5 results visible

Return ONLY valid JSON, no markdown."""

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

    def _is_atomic_goal(self, goal: str) -> bool:
        """Проверяет является ли цель атомарной (не требует декомпозиции)"""
        # Эвристики для простых целей
        simple_keywords = [
            'navigate to',
            'open',
            'click',
            'go to'
        ]

        goal_lower = goal.lower()

        # Если цель начинается с простого действия
        if any(goal_lower.startswith(kw) for kw in simple_keywords):
            # И не содержит "and" / "then"
            if ' and ' not in goal_lower and ' then ' not in goal_lower:
                return True

        # Очень короткая цель
        if len(goal.split()) <= 4:
            return True

        return False

    async def create_plan(self, goal: str) -> Optional[TaskPlan]:
        """
        Создаёт план для достижения цели.

        Args:
            goal: Цель пользователя

        Returns:
            TaskPlan или None если не удалось создать
        """
        print(f"\n📋 Planning Agent creating plan for: {goal}")

        # Проверяем нужна ли декомпозиция
        if self._is_atomic_goal(goal):
            print("ℹ️  Goal is atomic, creating single-step plan")
            step = PlanStep(
                step_number=1,
                description=goal,
                success_criteria="Goal is achieved",
                status=StepStatus.IN_PROGRESS
            )
            plan = TaskPlan(goal=goal, steps=[step])
            print(f"✅ Plan created with 1 step")
            return plan

        user_message = f"""Goal: {goal}

Create a detailed step-by-step plan with ATOMIC steps.

Remember:
- If goal involves multiple items (e.g., "buy X and Y"), create SEPARATE steps for each item
- Each step should be one clear action
- Include success criteria for verification"""

        try:
            response = await self.llm.generate_simple(
                user_message=user_message,
                system_prompt=self.SYSTEM_PROMPT
            )

            data = self._parse_json(response.content)

            if not data or 'steps' not in data:
                print("⚠️  Planning Agent: Could not parse plan, using atomic fallback")
                # Fallback: одношаговый план
                step = PlanStep(
                    step_number=1,
                    description=goal,
                    success_criteria="Goal is achieved",
                    status=StepStatus.IN_PROGRESS
                )
                return TaskPlan(goal=goal, steps=[step])

            # Показываем размышления
            if 'thinking' in data:
                thinking = data['thinking'][:200]
                print(f"💭 {thinking}{'...' if len(data['thinking']) > 200 else ''}")

            # Создаём шаги
            steps = []
            for step_data in data['steps']:
                step = PlanStep(
                    step_number=step_data.get('step', len(steps) + 1),
                    description=step_data.get('description', ''),
                    success_criteria=step_data.get('success_criteria', '')
                )
                steps.append(step)

            if not steps:
                print("⚠️  No steps created, using fallback")
                step = PlanStep(
                    step_number=1,
                    description=goal,
                    success_criteria="Goal is achieved",
                    status=StepStatus.IN_PROGRESS
                )
                return TaskPlan(goal=goal, steps=[step])

            # Первый шаг - в процессе
            steps[0].status = StepStatus.IN_PROGRESS

            plan = TaskPlan(goal=goal, steps=steps)

            print(f"\n✅ Plan created with {len(steps)} steps:")
            for i, step in enumerate(steps, 1):
                print(f"   {i}. {step.description}")

            if 'completion_criteria' in data:
                print(f"\n🎯 Completion criteria: {data['completion_criteria']}")

            return plan

        except Exception as e:
            print(f"⚠️  Planning Agent error: {e}")
            # Fallback
            step = PlanStep(
                step_number=1,
                description=goal,
                success_criteria="Goal is achieved",
                status=StepStatus.IN_PROGRESS
            )
            return TaskPlan(goal=goal, steps=[step])

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

Be strict - only mark as completed if success criteria is CLEARLY met."""

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

    def get_stats(self) -> Dict[str, Any]:
        """Статистика (для совместимости с Task Manager)"""
        return {
            "agent_type": "planning_agent"
        }


# Тест
if __name__ == "__main__":
    import asyncio
    from src.llm.ollama_provider_request import create_ollama_provider
    from src.llm.openrouter_provider import create_openrouter_provider

    async def test_planning():
        print("\n" + "=" * 80)
        print("PLANNING AGENT TEST - ATOMIC STEPS")
        print("=" * 80)

        # llm = await create_ollama_provider(model="qwen2.5:7b")
        llm = await create_openrouter_provider(model="mistralai/devstral-2512:free")
        planner = PlanningAgent(llm_provider=llm)

        # Тест 1: E-commerce (критический случай!)
        print("\n" + "=" * 80)
        print("TEST 1: E-commerce with multiple items")
        print("=" * 80)

        plan = await planner.create_plan(
            "купить на Яндекс Лавке BBQ бургер и картошку фри"
        )

        if plan:
            print("\n" + plan.get_progress_summary())
            print("\nContext for agents:")
            print(plan.get_context_for_agents())

        # Тест 2: Простая задача
        print("\n" + "=" * 80)
        print("TEST 2: Simple search")
        print("=" * 80)

        plan2 = await planner.create_plan(
            "Search for 'autonomous AI agents' on Google"
        )

        if plan2:
            print("\n" + plan2.get_progress_summary())

        # Тест 3: Атомарная цель
        print("\n" + "=" * 80)
        print("TEST 3: Atomic goal")
        print("=" * 80)

        plan3 = await planner.create_plan(
            "Navigate to google.com"
        )

        if plan3:
            print("\n" + plan3.get_progress_summary())

        await llm.close()


    asyncio.run(test_planning())