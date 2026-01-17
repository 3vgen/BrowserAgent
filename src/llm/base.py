"""
Base LLM Provider - абстрактный класс для всех AI провайдеров
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """Сообщение в диалоге"""
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "content": self.content
        }


@dataclass
class LLMResponse:
    """Ответ от LLM"""
    content: str
    raw_response: Any = None  # Полный ответ от API

    def __repr__(self) -> str:
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"<LLMResponse: {preview}>"


class BaseLLMProvider(ABC):
    """
    Базовый класс для всех LLM провайдеров.
    Позволяет легко переключаться между Ollama, Claude, OpenAI и т.д.
    """

    def __init__(
            self,
            model: str,
            temperature: float = 0.4,
            max_tokens: int = 2000,
            **kwargs
    ):
        """
        Args:
            model: Название модели
            temperature: Температура (0.0 - детерминированно, 1.0 - креативно)
            max_tokens: Максимум токенов в ответе
            **kwargs: Дополнительные параметры для конкретного провайдера
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

    @abstractmethod
    async def generate(
            self,
            messages: List[Message],
            system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Генерирует ответ на основе истории сообщений.

        Args:
            messages: История диалога
            system_prompt: Системный промпт (опционально)

        Returns:
            LLMResponse с ответом модели
        """
        pass

    async def generate_simple(
            self,
            user_message: str,
            system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Упрощённый метод для одного сообщения.

        Args:
            user_message: Сообщение пользователя
            system_prompt: Системный промпт

        Returns:
            LLMResponse
        """
        messages = [Message(role="user", content=user_message)]
        return await self.generate(messages, system_prompt)

    @abstractmethod
    def is_available(self) -> bool:
        """
        Проверяет доступность провайдера.

        Returns:
            True если провайдер доступен
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model}>"


class LLMProviderError(Exception):
    """Ошибка при работе с LLM провайдером"""
    pass