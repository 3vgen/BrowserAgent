"""
Orchestrator - оптимизированный координатор агентов

Улучшения:
- Умная обработка ошибок
- Адаптивное поведение
- Детальная статистика
- Graceful degradation
"""

import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..llm.base import BaseLLMProvider
from ..browser.manager import BrowserManager
from .vision_agent import VisionAgent
from .action_agent import ActionAgent, Action


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
    Оптимизированный оркестратор с умной обработкой ошибок.

    Улучшения:
    1. Retry логика для failed actions
    2. Адаптивные таймауты
    3. Детальная статистика
    4. Graceful degradation
    5. Emergency stop при критических ошибках
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        browser: BrowserManager,
        max_steps: int = 25,
        verbose: bool = True,
        retry_failed_actions: int = 1,
        emergency_stop_threshold: int = 5
    ):
        """
        Args:
            llm_provider: LLM провайдер
            browser: Менеджер браузера
            max_steps: Максимум шагов
            verbose: Детальные логи
            retry_failed_actions: Сколько раз повторять failed actions
            emergency_stop_threshold: Остановка после N ошибок подряд
        """
        self.browser = browser
        self.max_steps = max_steps
        self.verbose = verbose
        self.retry_failed_actions = retry_failed_actions
        self.emergency_stop_threshold = emergency_stop_threshold

        # Создаём sub-агентов с оптимизациями
        self.vision_agent = VisionAgent(
            llm_provider=llm_provider,
            cache_size=100
        )
        self.action_agent = ActionAgent(
            llm_provider=llm_provider,
            max_history=10,
            loop_detection_window=3
        )

        # Статистика
        self.consecutive_errors = 0
        self.total_errors = 0
        self.action_timings = []

    def _log(self, message: str, level: str = "info"):
        """
        Логирование с уровнями.

        Args:
            message: Сообщение
            level: info, warning, error, success
        """
        if not self.verbose:
            return

        prefixes = {
            "info": "",
            "warning": "⚠️  ",
            "error": "❌ ",
            "success": "✅ "
        }

        prefix = prefixes.get(level, "")
        print(f"{prefix}{message}")

    async def _execute_action_with_retry(
        self,
        action: Action
    ) -> Dict[str, Any]:
        """
        Выполняет действие с retry логикой.

        Args:
            action: Действие

        Returns:
            Результат выполнения
        """
        action_type = action.type
        params = action.params

        for attempt in range(self.retry_failed_actions + 1):
            try:
                if attempt > 0:
                    self._log(f"Retry attempt {attempt}/{self.retry_failed_actions}", "warning")
                    await asyncio.sleep(1)  # Пауза перед retry

                # Навигация
                if action_type == "navigate":
                    url = params.get("url")
                    result = await self.browser.navigate(url)

                # Клик
                elif action_type == "click":
                    element_id = params.get("element_id")
                    result = await self.browser.click(element_id)

                # Ввод текста
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
                    amount = params.get("amount", 500)
                    result = await self.browser.scroll(direction, amount)

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
                        "error": f"Unknown action type: {action_type}"
                    }

                # Если успешно - выходим
                if result.get('success'):
                    return result

                # Иначе пробуем ещё раз
                if attempt < self.retry_failed_actions:
                    self._log(f"Action failed: {result.get('error')}, retrying...", "warning")
                    continue

                return result

            except Exception as e:
                if attempt < self.retry_failed_actions:
                    self._log(f"Exception: {e}, retrying...", "warning")
                    continue

                return {
                    "success": False,
                    "error": f"Exception: {str(e)}"
                }

        return {
            "success": False,
            "error": "Max retries exceeded"
        }

    async def execute_task(
        self,
        goal: str,
        start_url: Optional[str] = None
    ) -> TaskResult:
        """
        Выполняет задачу с оптимизациями.

        Args:
            goal: Описание задачи
            start_url: Начальный URL

        Returns:
            TaskResult
        """
        start_time = time.time()

        self._log(f"\n{'='*80}")
        self._log(f"🎯 GOAL: {goal}")
        self._log(f"{'='*80}\n")

        # Сбрасываем состояние агентов
        self.action_agent.reset_history()
        self.consecutive_errors = 0
        self.total_errors = 0

        # Стартовая навигация
        if start_url:
            self._log(f"🌐 Starting at: {start_url}")
            result = await self.browser.navigate(start_url)

            if not result['success']:
                elapsed = time.time() - start_time
                return TaskResult(
                    success=False,
                    result="",
                    steps_completed=0,
                    total_time=elapsed,
                    error=f"Failed to navigate to start URL: {result.get('error')}"
                )

            self._log(f"Loaded: {result['title']}\n", "success")

        # Основной цикл
        for step in range(1, self.max_steps + 1):
            step_start_time = time.time()

            self._log(f"\n{'─'*80}")
            self._log(f"📍 STEP {step}/{self.max_steps}")
            self._log(f"{'─'*80}")

            # Emergency stop при слишком многих ошибках
            if self.consecutive_errors >= self.emergency_stop_threshold:
                self._log(
                    f"Emergency stop: {self.consecutive_errors} consecutive errors",
                    "error"
                )
                elapsed = time.time() - start_time

                return TaskResult(
                    success=False,
                    result="",
                    steps_completed=step - 1,
                    total_time=elapsed,
                    error=f"Too many consecutive errors ({self.consecutive_errors})",
                    stats=self.action_agent.get_stats()
                )

            try:
                # 1. Получаем состояние
                page_state = await self.browser.get_page_state()
                self._log(f"📄 {page_state['title']}")
                self._log(f"🔗 {page_state['url']}")
                self._log(f"🔢 Elements: {len(page_state['elements'])}")

                # 2. Vision Agent анализирует
                self._log(f"\n👁️  Vision Agent analyzing...")

                vision_analysis = await self.vision_agent.analyze_page(
                    goal=goal,
                    url=page_state['url'],
                    title=page_state['title'],
                    elements=page_state['elements'],
                    use_cache=True
                )

                self._log(f"   Type: {vision_analysis.page_type} (conf: {vision_analysis.confidence:.2f})")

                if vision_analysis.observations:
                    for obs in vision_analysis.observations[:2]:
                        self._log(f"   • {obs}")

                # 3. Фильтруем элементы
                relevant_elements = self.vision_agent.filter_elements(
                    page_state['elements'],
                    vision_analysis.relevant_elements,
                    max_elements=20
                )

                self._log(f"   Relevant: {len(relevant_elements)} elements")

                # 4. Action Agent решает
                self._log(f"\n🤖 Action Agent deciding...")

                action = await self.action_agent.decide_action(
                    goal=goal,
                    vision_analysis=vision_analysis,
                    relevant_elements=relevant_elements,
                    step_number=step,
                    max_steps=self.max_steps
                )

                if not action:
                    self._log("Failed to decide action", "error")
                    self.consecutive_errors += 1
                    self.total_errors += 1
                    continue

                # 5. Выполняем действие
                self._log(f"\n⚡ {action.type} (conf: {action.confidence:.2f})")
                if action.reasoning:
                    self._log(f"   {action.reasoning[:80]}")

                result = await self._execute_action_with_retry(action)

                # Записываем время выполнения
                step_time = time.time() - step_start_time
                self.action_timings.append(step_time)

                # Проверяем результат
                if not result.get('success'):
                    error_msg = result.get('error', 'Unknown error')
                    self._log(f"Action failed: {error_msg}", "error")

                    # Помечаем action как failed
                    self.action_agent.mark_action_failed(action)

                    self.consecutive_errors += 1
                    self.total_errors += 1
                    continue

                # Успех - сбрасываем счётчик ошибок
                self.consecutive_errors = 0
                self._log("Success", "success")

                # Проверяем завершение
                if result.get('completed'):
                    elapsed = time.time() - start_time

                    self._log(f"\n{'='*80}")
                    self._log("TASK COMPLETED!", "success")
                    self._log(f"📋 Result: {result.get('result')}")
                    self._log(f"📊 Steps: {step}/{self.max_steps}")
                    self._log(f"⏱️  Time: {elapsed:.1f}s")

                    stats = self.action_agent.get_stats()
                    if stats:
                        self._log(f"📈 Success rate: {stats.get('success_rate', 0)*100:.1f}%")

                    self._log(f"{'='*80}\n")

                    return TaskResult(
                        success=True,
                        result=result.get('result', ''),
                        steps_completed=step,
                        total_time=elapsed,
                        stats=stats
                    )

                # Пауза между шагами
                await asyncio.sleep(0.5)

            except Exception as e:
                self._log(f"Critical error in step {step}: {e}", "error")
                self.consecutive_errors += 1
                self.total_errors += 1

                if self.consecutive_errors >= self.emergency_stop_threshold:
                    break

        # Достигли максимума шагов
        elapsed = time.time() - start_time

        self._log(f"\n⚠️  Maximum steps ({self.max_steps}) reached", "warning")

        stats = self.action_agent.get_stats()

        return TaskResult(
            success=False,
            result="",
            steps_completed=self.max_steps,
            total_time=elapsed,
            error="Maximum steps reached",
            stats=stats
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Возвращает статистику производительности"""
        if not self.action_timings:
            return {}

        avg_time = sum(self.action_timings) / len(self.action_timings)
        max_time = max(self.action_timings)
        min_time = min(self.action_timings)

        return {
            "total_actions": len(self.action_timings),
            "avg_action_time": avg_time,
            "max_action_time": max_time,
            "min_action_time": min_time,
            "total_errors": self.total_errors
        }