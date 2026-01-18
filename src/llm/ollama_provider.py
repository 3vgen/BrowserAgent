"""
Ollama Provider - работа с локальными моделями через Ollama
"""

import httpx
from typing import List, Optional, Dict, Any
import json

from src.llm.base import BaseLLMProvider, Message, LLMResponse, LLMProviderError


class OllamaProvider(BaseLLMProvider):
    """
    Провайдер для Ollama (локальные LLM модели).

    Поддерживает модели:
    - qwen2.5:7b (рекомендуется)
    - qwen2.5:14b (умнее, медленнее)
    - llama3.1
    - mistral
    - и другие
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        temperature: float = 0.4,
        max_tokens: int = 2000,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        **kwargs
    ):
        """
        Args:
            model: Название модели в Ollama
            temperature: Температура генерации
            max_tokens: Максимум токенов (параметр num_predict в Ollama)
            base_url: URL Ollama сервера
            timeout: Таймаут запроса в секундах
        """
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        transport = httpx.AsyncHTTPTransport(http1=True, http2=False)
        self.client = httpx.AsyncClient(transport=transport, timeout=timeout)

    async def generate(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Генерирует ответ через Ollama API.

        Ollama API использует формат:
        POST /api/chat
        {
          "model": "qwen2.5:7b",
          "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
          ],
          "stream": false
        }
        """
        # Формируем список сообщений
        ollama_messages = []

        # Добавляем системный промпт если есть
        if system_prompt:
            ollama_messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Добавляем историю
        for msg in messages:
            ollama_messages.append(msg.to_dict())

        # Параметры запроса
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,  # Не стримим пока
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        try:
            # Отправляем запрос
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()

            # Парсим ответ
            data = response.json()

            # Извлекаем текст ответа
            content = data.get("message", {}).get("content", "")

            if not content:
                raise LLMProviderError("Empty response from Ollama")

            return LLMResponse(
                content=content,
                raw_response=data
            )

        except httpx.ConnectError:
            raise LLMProviderError(
                "Cannot connect to Ollama. Is it running? "
                "Start with: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise LLMProviderError(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                )
            raise LLMProviderError(f"Ollama HTTP error: {e}")
        except Exception as e:
            raise LLMProviderError(f"Ollama error: {e}")

    def is_available(self) -> bool:
        """
        Проверяет доступен ли Ollama сервер.

        Returns:
            True если Ollama запущен и модель доступна
        """
        try:
            # Синхронный запрос для простоты
            with httpx.Client(timeout=5) as client:
                # Проверяем что сервер запущен
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()

                # Проверяем что модель установлена
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]

                return self.model in models
        except:
            return False

    async def list_models(self) -> List[str]:
        """
        Получает список установленных моделей.

        Returns:
            Список названий моделей
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            raise LLMProviderError(f"Cannot list models: {e}")

    async def pull_model(self, model: str) -> None:
        """
        Скачивает модель (если не установлена).

        Args:
            model: Название модели
        """
        print(f"📥 Downloading model: {model}")
        print("This may take a few minutes...")

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=None  # Скачивание может быть долгим
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")

                            # Показываем прогресс
                            if "total" in data and "completed" in data:
                                total = data["total"]
                                completed = data["completed"]
                                percent = (completed / total * 100) if total > 0 else 0
                                print(f"\r{status}: {percent:.1f}%", end="", flush=True)
                            else:
                                print(f"\r{status}", end="", flush=True)
                        except json.JSONDecodeError:
                            pass

            print(f"\n✅ Model {model} ready!")

        except Exception as e:
            raise LLMProviderError(f"Cannot pull model: {e}")

    async def close(self):
        """Закрывает HTTP клиент"""
        await self.client.aclose()


# Вспомогательная функция для удобного создания
async def create_ollama_provider(
    model: str = "qwen2.5:7b",
    auto_pull: bool = True,
    **kwargs
) -> OllamaProvider:
    """
    Создаёт и настраивает Ollama провайдер.

    Args:
        model: Название модели
        auto_pull: Автоматически скачать модель если не установлена
        **kwargs: Дополнительные параметры

    Returns:
        Настроенный OllamaProvider
    """
    provider = OllamaProvider(model=model, **kwargs)

    # Проверяем доступность
    if not provider.is_available():
        print(f"⚠️  Model {model} not found locally")

        if auto_pull:
            await provider.pull_model(model)
        else:
            raise LLMProviderError(
                f"Model {model} not found. "
                f"Install with: ollama pull {model}"
            )

    return provider


# Пример использования
if __name__ == "__main__":
    import asyncio

    async def test_ollama():
        """Тестовая функция"""

        print("="*80)
        print("OLLAMA PROVIDER TEST")
        print("="*80)

        # Создаём провайдер
        print("\n📍 Creating provider...")
        try:
            provider = await create_ollama_provider(
                model="qwen2.5:7b",
                auto_pull=True,
                temperature=0.7
            )
            print(f"✅ Provider ready: {provider}")
        except LLMProviderError as e:
            print(f"❌ Error: {e}")
            print("\nMake sure Ollama is running:")
            print("  brew install ollama")
            print("  ollama serve")
            return

        # Тест 1: Простой запрос
        print("\n" + "─"*80)
        print("TEST 1: Simple question")
        print("─"*80)

        response = await provider.generate_simple(
            user_message="What is 2+2? Answer in one sentence.",
            system_prompt="You are a helpful AI assistant."
        )

        print(f"\n🤖 Response: {response.content}")

        # Тест 2: JSON генерация (важно для агента!)
        print("\n" + "─"*80)
        print("TEST 2: JSON generation")
        print("─"*80)

        response = await provider.generate_simple(
            user_message="""Сгенерируй JSON объект с этими полями:
- имя: случайное имя человека
- возраст: случайный возраст 20-50
- хобби: случайное хобби

Верни ТОЛЬКО JSON, без другого текста.""",
            system_prompt="Ты json генератор. Верни только json."
        )

        print(f"\n🤖 Response:\n{response.content}")

        # Пробуем распарсить JSON
        try:
            # Ищем JSON в ответе (на случай если модель добавила текст)
            content = response.content.strip()
            start = content.find('{')
            end = content.rfind('}') + 1

            if 0 <= start < end:
                json_str = content[start:end]
                data = json.loads(json_str)
                print(f"✅ Valid JSON parsed: {data}")
            else:
                print("⚠️  No JSON found in response")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")

        # Тест 3: Диалог с историей
        print("\n" + "─"*80)
        print("TEST 3: Conversation with history")
        print("─"*80)

        messages = [
            Message(role="user", content="My name is Alex"),
            Message(role="assistant", content="Nice to meet you, Alex!"),
            Message(role="user", content="What's my name?")
        ]

        response = await provider.generate(
            messages=messages,
            system_prompt="You are a helpful assistant with good memory."
        )

        print(f"\n🤖 Response: {response.content}")

        # Закрываем
        await provider.close()
        print("\n✅ All tests completed!")

    asyncio.run(test_ollama())