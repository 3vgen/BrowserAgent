"""
Orchestrator - УЛУЧШЕННЫЙ координатор с планированием

Новые возможности:
- Planning Agent для разбивки задач
- Детальное логирование
- Умная детекция завершения
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
    error: Optional[str] = None
    stats: Optional[Dict] = None

    def __repr__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"<TaskResult {status} steps={self.steps_completed} time={self.total_time:.1f}s>"


class Orchestrator:
    """
    Улучшенный оркестратор с планированием и логированием.

    Основные улучшения:
    1. Planning Agent - создаёт план перед выполнением
    2. Следование плану - агент знает что делать дальше
    3. Детекция завершения шагов - не делает лишнего
    4. Полное логирование - все размышления в файл
    5. Адаптивное поведение - меняет стратегию при зацикливании
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        browser: BrowserManager,
        max_steps: int = 30,
        verbose: bool = True,
        use_planning: bool = True,
        logger: Optional[AgentLogger] = None
    ):
        """
        Args:
            llm_provider: LLM провайдер
            browser: Менеджер браузера
            max_steps: Максимум шагов
            verbose: Показывать логи в консоль
            use_planning: Использовать Planning Agent
            logger: Внешний логгер (если None - создаст свой)
        """
        self.browser = browser
        self.max_steps = max_steps
        self.verbose = verbose
        self.use_planning = use_planning

        # Логгер
        self.logger = logger if logger else create_session_logger()

        # Sub-агенты
        self.vision_agent = VisionAgent(llm_provider=llm_provider)
        self.action_agent = ActionAgent(llm_provider=llm_provider)

        if use_planning:
            self.planning_agent = PlanningAgent(llm_provider=llm_provider)
        else:
            self.planning_agent = None

        # Состояние
        self.current_plan: Optional[TaskPlan] = None
        self.consecutive_errors = 0
        self.loop_attempts = 0

    def _log(self, message: str):
        """Логирование если verbose"""
        if self.verbose:
            print(message)

    async def _check_step_completion(
        self,
        current_step_description: str,
        current_step_criteria: str
    ) -> bool:
        """
        Проверяет завершён ли текущий шаг плана.

        Args:
            current_step_description: Описание шага
            current_step_criteria: Критерии успеха

        Returns:
            True если шаг завершён
        """
        # Получаем текущее состояние
        page_state = await self.browser.get_page_state()

        # Формируем описание текущей ситуации
        situation = f"""
Page URL: {page_state['url']}
Page Title: {page_state['title']}
Elements visible: {len([e for e in page_state['elements'] if e.is_in_viewport])}
"""

        # Проверяем через Planning Agent
        if self.planning_agent:
            from ..agents.planning_agent import PlanStep
            step = PlanStep(
                step_number=0,
                description=current_step_description,
                success_criteria=current_step_criteria
            )

            is_complete = await self.planning_agent.should_step_be_complete(
                step,
                situation
            )

            return is_complete

        return False

    async def _execute_action_safe(self, action: Action) -> Dict[str, Any]:
        """Выполняет действие с логированием"""

        action_type = action.type
        params = action.params

        self.logger.log_action_execution(action_type, True, retry_attempt=0)

        try:
            # Навигация
            if action_type == "navigate":
                url = params.get("url")
                result = await self.browser.navigate(url)

            # Клик
            elif action_type == "click":
                element_id = params.get("element_id")
                result = await self.browser.click(element_id)

            # Ввод
            elif action_type == "type":
                element_id = params.get("element_id")
                text = params.get("text")
                result = await self.browser.type_text(element_id, text)

            # Нажатие клавиши
            elif action_type == "press":
                key = params.get("key", "Enter")
                result = await self.browser.press_key(key)

            # Прокрутка
            elif action_type == "scroll":
                direction = params.get("direction", "down")
                result = await self.browser.scroll(direction)

            # Ожидание
            elif action_type == "wait":
                seconds = params.get("seconds", 2)
                result = await self.browser.wait(seconds)

            # Завершение
            elif action_type == "complete":
                result = {
                    "success": True,
                    "completed": True,
                    "result": params.get("result", "Task completed")
                }

            else:
                result = {
                    "success": False,
                    "error": f"Unknown action: {action_type}"
                }

            # Логируем результат
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
        Выполняет задачу с планированием.

        Args:
            goal: Цель
            start_url: Начальный URL

        Returns:
            TaskResult
        """
        start_time = time.time()

        # Логируем цель
        self.logger.log_goal(goal)

        # Сбрасываем состояние
        self.action_agent.reset_history()
        self.consecutive_errors = 0
        self.loop_attempts = 0

        # Шаг 1: Создаём план (если включено)
        if self.use_planning and self.planning_agent:
            self.current_plan = await self.planning_agent.create_plan(goal)

            if self.current_plan:
                self.logger.log_plan({
                    "steps": [s.to_dict() for s in self.current_plan.steps]
                })
            else:
                self.logger.log_warning("Failed to create plan, proceeding without it")

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
        for step in range(1, self.max_steps + 1):
            self.logger.log_step_start(step, self.max_steps)

            # Emergency stop
            if self.consecutive_errors >= 5:
                elapsed = time.time() - start_time
                error_msg = f"Emergency stop: {self.consecutive_errors} errors"
                self.logger.log_error(error_msg)

                return TaskResult(
                    success=False,
                    result="",
                    steps_completed=step - 1,
                    total_time=elapsed,
                    error=error_msg
                )

            try:
                # 1. Получаем состояние
                page_state = await self.browser.get_page_state()
                self.logger.log_page_state(
                    page_state['url'],
                    page_state['title'],
                    len(page_state['elements'])
                )

                # 2. Проверяем завершение шага плана (если есть план)
                if self.current_plan and not self.current_plan.is_completed():
                    current_step = self.current_plan.get_current_step()

                    if current_step and current_step.status == StepStatus.IN_PROGRESS:
                        # Проверяем завершён ли шаг
                        is_step_done = await self._check_step_completion(
                            current_step.description,
                            current_step.success_criteria
                        )

                        if is_step_done:
                            self.current_plan.mark_step_completed()
                            self.logger.log_step_completion(current_step.description)

                            # Если план завершён
                            if self.current_plan.is_completed():
                                elapsed = time.time() - start_time
                                result_msg = "All plan steps completed successfully"

                                self.logger.log_task_completion(
                                    True,
                                    result_msg,
                                    step,
                                    elapsed,
                                    self.action_agent.get_stats()
                                )

                                return TaskResult(
                                    success=True,
                                    result=result_msg,
                                    steps_completed=step,
                                    total_time=elapsed,
                                    stats=self.action_agent.get_stats()
                                )

                    # Помечаем текущий шаг как in_progress
                    if current_step and current_step.status == StepStatus.PENDING:
                        current_step.status = StepStatus.IN_PROGRESS

                # 3. Vision Agent анализирует
                self._log("\n👁️  Vision Agent analyzing...")

                vision_analysis = await self.vision_agent.analyze_page(
                    goal=goal,
                    url=page_state['url'],
                    title=page_state['title'],
                    elements=page_state['elements']
                )

                self.logger.log_vision_analysis(
                    vision_analysis.page_type,
                    vision_analysis.confidence,
                    vision_analysis.observations,
                    len(vision_analysis.relevant_elements)
                )

                # Логируем размышления (если есть в raw_response)
                if hasattr(vision_analysis, 'raw_response'):
                    self.logger.log_thinking(
                        "vision_agent",
                        vision_analysis.raw_response,
                        {
                            "page_type": vision_analysis.page_type,
                            "confidence": vision_analysis.confidence
                        }
                    )

                # 4. Фильтруем элементы
                relevant_elements = self.vision_agent.filter_elements(
                    page_state['elements'],
                    vision_analysis.relevant_elements
                )

                # 5. Action Agent решает
                self._log("\n🤖 Action Agent deciding...")

                # Добавляем контекст плана если есть
                planning_context = ""
                if self.current_plan and not self.current_plan.is_completed():
                    current_step = self.current_plan.get_current_step()
                    if current_step:
                        planning_context = f"\nCURRENT PLAN STEP: {current_step.description}\nSUCCESS CRITERIA: {current_step.success_criteria}\n"

                action = await self.action_agent.decide_action(
                    goal=goal + planning_context,
                    vision_analysis=vision_analysis,
                    relevant_elements=relevant_elements,
                    step_number=step,
                    max_steps=self.max_steps
                )

                if not action:
                    self.logger.log_error("Action Agent failed to decide")
                    self.consecutive_errors += 1
                    continue

                # Логируем решение
                self.logger.log_action_decision(
                    action.type,
                    action.params,
                    action.reasoning,
                    action.confidence
                )

                # 6. Детекция зацикливания
                if self.action_agent._detect_loop():
                    self.logger.log_loop_detected()
                    self.loop_attempts += 1

                    if self.loop_attempts >= 2:
                        # Пробуем scroll или skip
                        self._log("Trying to break loop with scroll...")
                        await self.browser.scroll("down")
                        self.loop_attempts = 0

                # 7. Выполняем действие
                result = await self._execute_action_safe(action)

                # 8. Обрабатываем результат
                if not result.get('success'):
                    self.action_agent.mark_action_failed(action)
                    self.consecutive_errors += 1

                    # Помечаем шаг плана как failed
                    if self.current_plan:
                        current_step = self.current_plan.get_current_step()
                        if current_step:
                            self.current_plan.mark_step_failed()

                    continue

                # Успех
                self.consecutive_errors = 0

                # Проверяем завершение
                if result.get('completed'):
                    elapsed = time.time() - start_time
                    result_msg = result.get('result', 'Task completed')

                    self.logger.log_task_completion(
                        True,
                        result_msg,
                        step,
                        elapsed,
                        self.action_agent.get_stats()
                    )

                    return TaskResult(
                        success=True,
                        result=result_msg,
                        steps_completed=step,
                        total_time=elapsed,
                        stats=self.action_agent.get_stats()
                    )

                await asyncio.sleep(0.5)

            except Exception as e:
                self.logger.log_error(str(e), f"Step {step}")
                self.consecutive_errors += 1

        # Достигли максимума
        elapsed = time.time() - start_time

        self.logger.log_task_completion(
            False,
            "Maximum steps reached",
            self.max_steps,
            elapsed,
            self.action_agent.get_stats()
        )

        return TaskResult(
            success=False,
            result="",
            steps_completed=self.max_steps,
            total_time=elapsed,
            error="Maximum steps reached",
            stats=self.action_agent.get_stats()
        )