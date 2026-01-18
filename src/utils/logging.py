"""
Agent Logger - детальное логирование работы агентов

Логирует:
- Все размышления агентов (thinking)
- Принятые решения
- Выполненные действия
- Ошибки и retry
- Статистику
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum


class LogLevel(Enum):
    """Уровни логирования"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AgentLogger:
    """
    Централизованный логгер для всех агентов.

    Логирует в файл + консоль с разными уровнями детализации.
    """

    def __init__(
            self,
            log_dir: str = "./data/logs",
            session_name: Optional[str] = None,
            console_level: str = "INFO",
            file_level: str = "DEBUG"
    ):
        """
        Args:
            log_dir: Директория для логов
            session_name: Имя сессии (для имени файла)
            console_level: Уровень для консоли
            file_level: Уровень для файла
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Имя сессии
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.session_name = session_name
        self.log_file = self.log_dir / f"agent_{session_name}.log"
        self.thinking_file = self.log_dir / f"thinking_{session_name}.jsonl"

        # Настройка логгера
        self.logger = logging.getLogger(f"AgentLogger_{session_name}")
        self.logger.setLevel(logging.DEBUG)

        # Очищаем существующие handlers
        self.logger.handlers = []

        # File handler (все детали)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, file_level))
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        # Console handler (важное)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level))
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # Счётчики
        self.step_counter = 0
        self.action_counter = 0

        self.log_session_start()

    def log_session_start(self):
        """Логирует начало сессии"""
        self.logger.info("=" * 80)
        self.logger.info(f"SESSION STARTED: {self.session_name}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Thinking log: {self.thinking_file}")
        self.logger.info("=" * 80)

    def log_goal(self, goal: str):
        """Логирует цель задачи"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"🎯 GOAL: {goal}")
        self.logger.info("=" * 80)

    def log_plan(self, plan_data: Dict):
        """Логирует созданный план"""
        self.logger.info("")
        self.logger.info("📋 PLAN CREATED:")

        if 'steps' in plan_data:
            for step in plan_data['steps']:
                self.logger.info(f"  {step.get('step', '?')}. {step.get('description', '')}")

        self.logger.info("")

    def log_step_start(self, step_number: int, max_steps: int):
        """Логирует начало шага"""
        self.step_counter = step_number
        self.logger.info("")
        self.logger.info("─" * 80)
        self.logger.info(f"📍 STEP {step_number}/{max_steps}")
        self.logger.info("─" * 80)

    def log_page_state(self, url: str, title: str, elements_count: int):
        """Логирует состояние страницы"""
        self.logger.info(f"📄 Page: {title}")
        self.logger.info(f"🔗 URL: {url}")
        self.logger.info(f"🔢 Elements: {elements_count}")

    def log_vision_analysis(
            self,
            page_type: str,
            confidence: float,
            observations: list,
            relevant_count: int
    ):
        """Логирует анализ Vision Agent"""
        self.logger.info("")
        self.logger.info("👁️  VISION AGENT ANALYSIS:")
        self.logger.info(f"   Page type: {page_type} (confidence: {confidence:.2f})")
        self.logger.info(f"   Relevant elements: {relevant_count}")

        if observations:
            self.logger.info("   Observations:")
            for obs in observations[:3]:
                self.logger.info(f"     • {obs}")

    def log_thinking(
            self,
            agent_name: str,
            thinking: str,
            context: Optional[Dict] = None
    ):
        """
        Логирует размышления агента (в отдельный файл).

        Args:
            agent_name: Имя агента (vision, action, planning)
            thinking: Текст размышлений
            context: Дополнительный контекст
        """
        # В консоль - кратко
        short_thinking = thinking[:120] + "..." if len(thinking) > 120 else thinking
        self.logger.info(f"💭 {agent_name}: {short_thinking}")

        # В файл - полностью (JSONL)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": self.step_counter,
            "agent": agent_name,
            "thinking": thinking,
            "context": context or {}
        }

        with open(self.thinking_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # В основной лог - тоже полностью но на уровне DEBUG
        self.logger.debug(f"[{agent_name}] FULL THINKING: {thinking}")

    def log_action_decision(
            self,
            action_type: str,
            params: Dict,
            reasoning: str,
            confidence: float
    ):
        """Логирует решение Action Agent"""
        self.action_counter += 1

        self.logger.info("")
        self.logger.info("🤖 ACTION AGENT DECISION:")
        self.logger.info(f"   Action: {action_type}")
        self.logger.info(f"   Confidence: {confidence:.2f}")
        self.logger.info(f"   Reasoning: {reasoning}")

        # Параметры на DEBUG уровне
        self.logger.debug(f"   Params: {json.dumps(params, ensure_ascii=False)}")

    def log_action_execution(
            self,
            action_type: str,
            success: bool,
            error: Optional[str] = None,
            retry_attempt: int = 0
    ):
        """Логирует выполнение действия"""
        if retry_attempt > 0:
            self.logger.warning(f"⚡ Executing {action_type} (retry {retry_attempt})")
        else:
            self.logger.info(f"⚡ Executing {action_type}")

        if success:
            self.logger.info("✅ Success")
        else:
            self.logger.error(f"❌ Failed: {error}")

    def log_loop_detected(self):
        """Логирует обнаружение зацикливания"""
        self.logger.warning("")
        self.logger.warning("⚠️  LOOP DETECTED!")
        self.logger.warning("Agent is repeating same actions. Trying different approach.")

    def log_step_completion(self, step_description: str):
        """Логирует завершение шага плана"""
        self.logger.info(f"✅ Step completed: {step_description}")

    def log_task_completion(
            self,
            success: bool,
            result: str,
            steps_completed: int,
            total_time: float,
            stats: Optional[Dict] = None
    ):
        """Логирует завершение задачи"""
        self.logger.info("")
        self.logger.info("=" * 80)

        if success:
            self.logger.info("✅ TASK COMPLETED SUCCESSFULLY")
            self.logger.info(f"📋 Result: {result}")
        else:
            self.logger.error("❌ TASK FAILED")
            self.logger.error(f"⚠️  Error: {result}")

        self.logger.info(f"📊 Steps: {steps_completed}")
        self.logger.info(f"⏱️  Time: {total_time:.1f}s")

        if stats:
            self.logger.info("")
            self.logger.info("📈 STATISTICS:")
            for key, value in stats.items():
                self.logger.info(f"   {key}: {value}")

        self.logger.info("=" * 80)

    def log_error(self, error: str, context: Optional[str] = None):
        """Логирует ошибку"""
        self.logger.error(f"❌ ERROR: {error}")
        if context:
            self.logger.error(f"   Context: {context}")

    def log_warning(self, warning: str):
        """Логирует предупреждение"""
        self.logger.warning(f"⚠️  {warning}")

    def log_debug(self, message: str):
        """Логирует отладочную информацию"""
        self.logger.debug(message)

    def get_log_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по логам"""
        return {
            "session": self.session_name,
            "log_file": str(self.log_file),
            "thinking_file": str(self.thinking_file),
            "steps_executed": self.step_counter,
            "actions_executed": self.action_counter
        }

    def close(self):
        """Закрывает логгер"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"SESSION ENDED: {self.session_name}")
        self.logger.info(f"Total steps: {self.step_counter}")
        self.logger.info(f"Total actions: {self.action_counter}")
        self.logger.info("=" * 80)

        # Закрываем handlers
        for handler in self.logger.handlers:
            handler.close()


# Глобальный экземпляр логгера
_global_logger: Optional[AgentLogger] = None


def get_logger() -> AgentLogger:
    """Возвращает глобальный логгер"""
    global _global_logger

    if _global_logger is None:
        _global_logger = AgentLogger()

    return _global_logger


def create_session_logger(session_name: Optional[str] = None) -> AgentLogger:
    """Создаёт новый логгер для сессии"""
    global _global_logger

    # Закрываем старый если есть
    if _global_logger is not None:
        _global_logger.close()

    _global_logger = AgentLogger(session_name=session_name)
    return _global_logger


# Пример использования
if __name__ == "__main__":
    # Создаём логгер
    logger = AgentLogger(session_name="test_session")

    # Логируем цель
    logger.log_goal("Search for Python on Google")

    # Логируем план
    logger.log_plan({
        "steps": [
            {"step": 1, "description": "Navigate to Google"},
            {"step": 2, "description": "Type query"},
            {"step": 3, "description": "Click search"}
        ]
    })

    # Логируем шаг
    logger.log_step_start(1, 10)
    logger.log_page_state("https://google.com", "Google", 50)

    # Логируем размышления
    logger.log_thinking(
        "vision_agent",
        "I can see this is Google homepage. There's a search box in the center of the page."
    )

    logger.log_thinking(
        "action_agent",
        "Based on Vision Agent analysis, I should type the query into the search box first."
    )

    # Логируем действие
    logger.log_action_decision(
        "type",
        {"element_id": "elem_5", "text": "Python"},
        "Need to enter search query",
        0.92
    )

    logger.log_action_execution("type", True)

    # Завершение
    logger.log_task_completion(
        True,
        "Search completed successfully",
        3,
        15.5,
        {"success_rate": 1.0, "avg_confidence": 0.88}
    )

    # Показываем где логи
    summary = logger.get_log_summary()
    print("\n" + "=" * 80)
    print("LOG FILES:")
    print(f"  Main log: {summary['log_file']}")
    print(f"  Thinking log: {summary['thinking_file']}")
    print("=" * 80)

    logger.close()