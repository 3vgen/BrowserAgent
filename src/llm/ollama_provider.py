"""
Ollama Provider - работа с локальными моделями через Ollama
Использует raw socket для обхода проблемы с 503
"""

import socket
import json
import asyncio
from typing import List, Optional

from src.llm.base import BaseLLMProvider, Message, LLMResponse, LLMProviderError


class OllamaProvider(BaseLLMProvider):
    """
    Провайдер для Ollama (локальные LLM модели).
    """

    def __init__(
            self,
            model: str = "qwen2.5:7b",
            temperature: float = 0.4,
            max_tokens: int = 2000,
            host: str = "127.0.0.1",
            port: int = 11434,
            timeout: int = 120,
            **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.host = host
        self.port = port
        self.timeout = timeout

    def _raw_request(self, method: str, path: str, body: dict = None) -> dict:
        """
        Делает HTTP запрос через raw socket.
        Обходит проблему 503 с httpx/requests.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        try:
            sock.connect((self.host, self.port))

            # Формируем запрос
            if body:
                body_str = json.dumps(body)
                request = (
                    f"{method} {path} HTTP/1.1\r\n"
                    f"Host: {self.host}:{self.port}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(body_str)}\r\n"
                    f"Connection: close\r\n"
                    f"\r\n"
                    f"{body_str}"
                )
            else:
                request = (
                    f"{method} {path} HTTP/1.1\r\n"
                    f"Host: {self.host}:{self.port}\r\n"
                    f"Connection: close\r\n"
                    f"\r\n"
                )

            sock.send(request.encode())

            # Читаем ответ
            response = b""
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                response += data

            # Парсим HTTP ответ
            response_str = response.decode('utf-8', errors='replace')

            # Разделяем заголовки и тело
            header_end = response_str.find("\r\n\r\n")
            if header_end == -1:
                raise LLMProviderError("Invalid HTTP response")

            headers = response_str[:header_end]
            body_text = response_str[header_end + 4:]

            # Проверяем статус
            first_line = headers.split("\r\n")[0]
            status_code = int(first_line.split()[1])

            if status_code != 200:
                raise LLMProviderError(f"HTTP {status_code}: {body_text[:200]}")

            # Парсим JSON
            return json.loads(body_text)

        finally:
            sock.close()

    async def generate(
            self,
            messages: List[Message],
            system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Генерирует ответ через Ollama API."""

        ollama_messages = []

        if system_prompt:
            ollama_messages.append({
                "role": "system",
                "content": system_prompt
            })

        for msg in messages:
            ollama_messages.append(msg.to_dict())

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        # Выполняем в thread pool чтобы не блокировать event loop
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: self._raw_request("POST", "/api/chat", payload)
        )

        content = data.get("message", {}).get("content", "")

        if not content:
            raise LLMProviderError("Empty response from Ollama")

        return LLMResponse(content=content, raw_response=data)

    def is_available(self) -> bool:
        """Проверяет доступен ли Ollama и модель."""
        try:
            data = self._raw_request("GET", "/api/tags")
            models = [m["name"] for m in data.get("models", [])]
            return self.model in models
        except:
            return False

    async def list_models(self) -> List[str]:
        """Получает список установленных моделей."""
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: self._raw_request("GET", "/api/tags")
        )
        return [m["name"] for m in data.get("models", [])]

    async def close(self):
        """Ничего закрывать не нужно."""
        pass


async def create_ollama_provider(
        model: str = "qwen2.5:7b",
        **kwargs
) -> OllamaProvider:
    """Создаёт Ollama провайдер."""
    provider = OllamaProvider(model=model, **kwargs)

    if not provider.is_available():
        raise LLMProviderError(
            f"Model {model} not found. Install with: ollama pull {model}"
        )

    return provider


# Тест
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