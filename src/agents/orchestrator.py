"""
Orchestrator - координатор с Planning Agent

Архитектура:
- Planning Agent для декомпозиции задач
- Vision Agent анализирует в контексте текущего шага
- Action Agent фокусируется на текущем шаге
- Детальное логирование
- Адаптивное поведение при зацикливании
"""

import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..llm.base import BaseLLMProvider
from ..browser.manager import BrowserManager
from ..agents.vision_agent import VisionAgent
from ..agents.action_agent import ActionAgent, Action
from ..agents.planning_agent import PlanningAgent, TaskPlan, StepStatus
from ..utils.logging import AgentLogger, create_session_logger


@dataclass
class TaskResult:
    """Результат выполнения задачи"""
    success: bool
    result: str
    steps_completed: int
    total_time: float
    plan_steps_completed: int = 0
    plan_steps_total: int = 0
    error: Optional[str] = None
    stats: Optional[Dict] = None

    def __repr__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"<TaskResult {status} plan={self.plan_steps_completed}/{self.plan_steps_total} steps={self.steps_completed} time={self.total_time:.1f}s>"


class Orchestrator:
    """
    Оркестратор с Planning Agent для многошаговых задач.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        browser: BrowserManager,
        max_steps_per_plan_step: int = 10,
        max_total_steps: int = 50,
        verbose: bool = True,
        use_planning: bool = True,
        logger: Optional[AgentLogger] = None
    ):
        """
        Args:
            llm_provider: LLM провайдер
            browser: Менеджер браузера
            max_steps_per_plan_step: Максимум действий на один шаг плана
            max_total_steps: Максимум действий всего
            verbose: Показывать логи в консоль
            use_planning: Использовать Planning Agent
            logger: Внешний логгер
        """
        self.browser = browser
        self.max_steps_per_plan_step = max_steps_per_plan_step
        self.max_total_steps = max_total_steps
        self.verbose = verbose
        self.use_planning = use_planning

        # Логгер
        self.logger = logger if logger else create_session_logger()

        # Агенты
        self.vision_agent = VisionAgent(llm_provider=llm_provider)
        self.action_agent = ActionAgent(llm_provider=llm_provider)

        if use_planning:
            self.planning_agent = PlanningAgent(llm_provider=llm_provider)
        else:
            self.planning_agent = None

        # Состояние
        self.current_plan: Optional[TaskPlan] = None
        self.total_steps = 0
        self.plan_step_actions = 0
        self.consecutive_errors = 0
        self.loop_attempts = 0

    def _log(self, message: str):
        """Логирование если verbose"""
        if self.verbose:
            print(message)

    async def _execute_action_safe(self, action: Action) -> Dict[str, Any]:
        """Выполняет действие с логированием"""
        action_type = action.type
        params = action.params

        self.logger.log_action_execution(action_type, True, retry_attempt=0)

        try:
            if action_type == "navigate":
                url = params.get("url")
                result = await self.browser.navigate(url)

            elif action_type == "click":
                element_id = params.get("element_id")
                result = await self.browser.click(element_id)

            elif action_type == "type":
                element_id = params.get("element_id")
                text = params.get("text")
                result = await self.browser.type_text(element_id, text)

            elif action_type == "press":
                key = params.get("key", "Enter")
                result = await self.browser.press_key(key)

            elif action_type == "scroll":
                direction = params.get("direction", "down")
                result = await self.browser.scroll(direction)

            elif action_type == "wait":
                seconds = params.get("seconds", 2)
                result = await self.browser.wait(seconds)

            else:
                result = {
                    "success": False,
                    "error": f"Unknown action: {action_type}"
                }

            if result.get('success'):
                self.logger.log_action_execution(action_type, True)
            else:
                self.logger.log_action_execution(
                    action_type,
                    False,
                    error=result.get('error')
                )

            return result

        except Exception as e:
            error_msg = str(e)
            self.logger.log_action_execution(action_type, False, error=error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    async def execute_task(
        self,
        goal: str,
        start_url: Optional[str] = None
    ) -> TaskResult:
        """
        Выполняет задачу с декомпозицией на шаги.
        """
        start_time = time.time()

        self.logger.log_goal(goal)
        self._log(f"\n{'='*70}")
        self._log(f"🎯 GOAL: {goal}")
        self._log(f"{'='*70}\n")

        # Сброс состояния
        self.action_agent.reset_history()
        self.total_steps = 0
        self.plan_step_actions = 0
        self.consecutive_errors = 0
        self.loop_attempts = 0

        # Шаг 1: Создание плана
        if self.use_planning and self.planning_agent:
            self._log("📋 Planning Agent: creating plan...")

            self.current_plan = await self.planning_agent.create_plan(goal)

            if not self.current_plan:
                elapsed = time.time() - start_time
                error_msg = "Failed to create plan"
                self.logger.log_error(error_msg)

                return TaskResult(
                    success=False,
                    result="",
                    steps_completed=0,
                    total_time=elapsed,
                    error=error_msg
                )

            # Логируем план
            self.logger.log_plan({
                "steps": [s.to_dict() for s in self.current_plan.steps]
            })

        # Шаг 2: Начальная навигация
        if start_url:
            self._log(f"🌐 Starting at: {start_url}")
            result = await self.browser.navigate(start_url)

            if not result['success']:
                elapsed = time.time() - start_time
                error_msg = f"Failed to navigate: {result.get('error')}"
                self.logger.log_error(error_msg)

                return TaskResult(
                    success=False,
                    result="",
                    steps_completed=0,
                    total_time=elapsed,
                    error=error_msg
                )

        # Шаг 3: Основной цикл
        while self.total_steps < self.max_total_steps:
            # Проверяем завершение плана
            if self.current_plan and self.current_plan.is_completed():
                elapsed = time.time() - start_time
                progress = self.current_plan.get_progress()
                result_msg = f"All {progress['total']} plan steps completed!"

                self._log(f"\n{'='*70}")
                self._log(f"✅ SUCCESS: {result_msg}")
                self._log(f"{'='*70}\n")

                self.logger.log_task_completion(
                    True,
                    result_msg,
                    self.total_steps,
                    elapsed,
                    self.action_agent.get_stats()
                )

                return TaskResult(
                    success=True,
                    result=result_msg,
                    steps_completed=self.total_steps,
                    total_time=elapsed,
                    plan_steps_completed=progress['completed'],
                    plan_steps_total=progress['total'],
                    stats=self.action_agent.get_stats()
                )

            # Получаем текущий шаг плана
            current_plan_step = None
            if self.current_plan:
                current_plan_step = self.current_plan.get_current_step()

                if not current_plan_step:
                    if self.current_plan.has_failed():
                        elapsed = time.time() - start_time
                        error_msg = "Some plan steps failed"

                        return TaskResult(
                            success=False,
                            result="",
                            steps_completed=self.total_steps,
                            total_time=elapsed,
                            error=error_msg,
                            stats=self.action_agent.get_stats()
                        )
                    break

            # Выполняем итерацию текущего шага плана
            step_result = await self._execute_plan_step_iteration(
                current_plan_step.description if current_plan_step else goal,
                current_plan_step.success_criteria if current_plan_step else "Goal achieved"
            )

            # Обработка результата
            if step_result.get('error'):
                self.consecutive_errors += 1

                if self.consecutive_errors >= 5:
                    elapsed = time.time() - start_time
                    error_msg = f"Emergency stop: {self.consecutive_errors} errors"

                    return TaskResult(
                        success=False,
                        result="",
                        steps_completed=self.total_steps,
                        total_time=elapsed,
                        error=error_msg
                    )
            else:
                self.consecutive_errors = 0

            # Проверяем завершение шага плана
            if step_result.get('step_complete'):
                if self.current_plan and current_plan_step:
                    self.current_plan.mark_step_completed(
                        result=step_result.get('result', '')
                    )

                    # Сброс для нового шага
                    self.plan_step_actions = 0
                    self.action_agent.reset_history()
                    self.logger.log_step_completion(current_plan_step.description)

            # Лимит действий на шаг плана
            elif self.plan_step_actions >= self.max_steps_per_plan_step:
                self._log(f"⚠️  Max actions reached for current plan step")

                if self.current_plan and current_plan_step:
                    self.current_plan.mark_step_failed(
                        reason=f"Max actions ({self.max_steps_per_plan_step}) reached"
                    )

                    if current_plan_step.is_failed():
                        self._log(f"✗ Skipping failed plan step")
                        self.plan_step_actions = 0
                        self.action_agent.reset_history()

        # Максимум шагов достигнут
        elapsed = time.time() - start_time

        if self.current_plan:
            progress = self.current_plan.get_progress()

            return TaskResult(
                success=False,
                result=f"Partial: {progress['completed']}/{progress['total']} steps",
                steps_completed=self.total_steps,
                total_time=elapsed,
                plan_steps_completed=progress['completed'],
                plan_steps_total=progress['total'],
                error="Maximum steps reached",
                stats=self.action_agent.get_stats()
            )
        else:
            return TaskResult(
                success=False,
                result="",
                steps_completed=self.total_steps,
                total_time=elapsed,
                error="Maximum steps reached",
                stats=self.action_agent.get_stats()
            )

    async def _execute_plan_step_iteration(
        self,
        current_step_description: str,
        success_criteria: str
    ) -> Dict[str, Any]:
        """
        Выполняет одну итерацию текущего шага плана.
        """
        self.total_steps += 1
        self.plan_step_actions += 1

        self.logger.log_step_start(self.total_steps, self.max_total_steps)
        self._log(f"\n{'─'*70}")
        self._log(f"Action {self.total_steps}/{self.max_total_steps} (plan step action {self.plan_step_actions}/{self.max_steps_per_plan_step})")
        self._log(f"{'─'*70}")

        try:
            # 1. Получаем состояние
            page_state = await self.browser.get_page_state()

            self.logger.log_page_state(
                page_state['url'],
                page_state['title'],
                len(page_state['elements'])
            )

            # 2. Получаем контекст плана
            task_context = ""
            if self.current_plan:
                task_context = self.current_plan.get_context_for_agents()

            # 3. Vision Agent анализирует
            self._log("\n👁️  Vision Agent analyzing...")

            vision_analysis = await self.vision_agent.analyze_page(
                goal=current_step_description,
                url=page_state['url'],
                title=page_state['title'],
                elements=page_state['elements'],
                task_context=task_context
            )

            self.logger.log_vision_analysis(
                vision_analysis.page_type,
                vision_analysis.confidence,
                vision_analysis.observations,
                len(vision_analysis.relevant_elements)
            )

            if hasattr(vision_analysis, 'raw_response'):
                self.logger.log_thinking(
                    "vision_agent",
                    vision_analysis.raw_response,
                    {
                        "page_type": vision_analysis.page_type,
                        "confidence": vision_analysis.confidence,
                        "subtask_achieved": vision_analysis.subtask_achieved
                    }
                )

            if vision_analysis.subtask_achieved:
                self._log("✓ Vision Agent: step appears achieved")

            # 4. Фильтруем элементы
            relevant_elements = self.vision_agent.filter_elements(
                page_state['elements'],
                vision_analysis.relevant_elements
            )

            # 5. Action Agent решает
            self._log("\n🤖 Action Agent deciding...")

            action = await self.action_agent.decide_action(
                current_subtask=current_step_description,
                task_context=task_context,
                vision_analysis=vision_analysis,
                relevant_elements=relevant_elements,
                step_number=self.plan_step_actions,
                max_steps=self.max_steps_per_plan_step,
                current_url=page_state['url']
            )

            if not action:
                self.logger.log_error("Action Agent failed to decide")
                return {'error': 'Action decision failed'}

            self.logger.log_action_decision(
                action.type,
                action.params,
                action.reasoning,
                action.confidence
            )

            # 6. Детекция зацикливания
            is_loop, loop_type = self.action_agent._detect_loop()
            if is_loop:
                self.logger.log_loop_detected()
                self.loop_attempts += 1

                if self.loop_attempts >= 2:
                    self._log("⚠️  Breaking loop with scroll...")
                    await self.browser.scroll("down")
                    self.loop_attempts = 0

            # 7. Выполняем действие
            result = await self._execute_action_safe(action)

            if not result.get('success'):
                self.action_agent.mark_action_failed(action)
                return {'error': result.get('error', 'Action failed')}

            self.loop_attempts = 0

            # 8. Проверяем завершение шага
            step_complete = (
                action.subtask_complete or
                vision_analysis.subtask_achieved
            )

            if step_complete:
                self._log("✅ Plan step marked as COMPLETE")

            return {
                'step_complete': step_complete,
                'result': result.get('result', ''),
                'error': None
            }

        except Exception as e:
            self.logger.log_error(str(e), f"Action {self.total_steps}")
            return {'error': str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Статистика выполнения"""
        stats = {
            'total_steps': self.total_steps,
            'action_agent': self.action_agent.get_stats()
        }

        if self.current_plan:
            stats['plan'] = {
                'steps': [s.to_dict() for s in self.current_plan.steps],
                'progress': self.current_plan.get_progress()
            }

        return stats