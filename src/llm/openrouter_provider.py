import httpx
from typing import List, Optional, Dict, Any
import json

from src.llm.base import BaseLLMProvider, Message, LLMResponse, LLMProviderError


class OpenRouterProvider(BaseLLMProvider):
    """
    Провайдер для OpenRouter (доступ к различным LLM через единый API).

    Поддерживает модели:
    - openai/gpt-oss-20b:free
    - anthropic/claude-3-opus
    - google/gemini-pro
    - meta-llama/llama-3-70b
    - и многие другие
    """

    def __init__(
            self,
            model: str = "openai/gpt-oss-20b:free",
            temperature: float = 0.4,
            max_tokens: int = 2000,
            api_key: Optional[str] = None,
            base_url: str = "https://openrouter.ai/api/v1",
            timeout: int = 120,
            enable_reasoning: bool = False,
            **kwargs
    ):
        """
        Args:
            model: Название модели в OpenRouter
            temperature: Температура генерации
            max_tokens: Максимум токенов
            api_key: API ключ OpenRouter (обязательно!)
            base_url: URL OpenRouter API
            timeout: Таймаут запроса в секундах
            enable_reasoning: Включить режим reasoning (для поддерживаемых моделей)
        """
        super().__init__(model, temperature, max_tokens, **kwargs)

        if not api_key:
            raise LLMProviderError(
                "OpenRouter API key is required. "
                "Get it at: https://openrouter.ai/keys"
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.enable_reasoning = enable_reasoning

        # Создаём HTTP клиент с заголовками
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/your-project",  # Опционально
                "X-Title": "Your App Name",  # Опционально
            }
        )

    async def generate(
            self,
            messages: List[Message],
            system_prompt: Optional[str] = None,
            preserve_reasoning: bool = False
    ) -> LLMResponse:
        """
        Генерирует ответ через OpenRouter API.

        Args:
            messages: История диалога
            system_prompt: Системный промпт
            preserve_reasoning: Сохранить reasoning_details из предыдущих сообщений

        Returns:
            LLMResponse с ответом и reasoning_details (если включено)
        """
        # Формируем список сообщений
        api_messages = []

        # Добавляем системный промпт если есть
        if system_prompt:
            api_messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Добавляем историю
        for msg in messages:
            message_dict = msg.to_dict()

            # Если это Message объект с дополнительными полями
            if hasattr(msg, 'reasoning_details') and preserve_reasoning:
                message_dict['reasoning_details'] = msg.reasoning_details

            api_messages.append(message_dict)

        # Базовые параметры запроса
        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Добавляем reasoning если включено
        if self.enable_reasoning:
            payload["extra_body"] = {
                "reasoning": {"enabled": True}
            }

        try:
            # Отправляем запрос
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()

            # Парсим ответ
            data = response.json()

            # Извлекаем сообщение
            if not data.get("choices") or len(data["choices"]) == 0:
                raise LLMProviderError("Empty response from OpenRouter")

            message = data["choices"][0].get("message", {})
            content = message.get("content", "")

            if not content:
                raise LLMProviderError("Empty content in response")

            # Создаём расширенный ответ
            llm_response = LLMResponse(
                content=content,
                raw_response=data
            )

            # Добавляем reasoning_details если есть
            if "reasoning_details" in message:
                llm_response.reasoning_details = message["reasoning_details"]

            return llm_response

        except httpx.ConnectError:
            raise LLMProviderError(
                "Cannot connect to OpenRouter API. "
                "Check your internet connection."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise LLMProviderError(
                    "Invalid API key. Get one at: https://openrouter.ai/keys"
                )
            elif e.response.status_code == 404:
                raise LLMProviderError(
                    f"Model '{self.model}' not found. "
                    f"Check available models at: https://openrouter.ai/models"
                )
            elif e.response.status_code == 429:
                raise LLMProviderError(
                    "Rate limit exceeded. Wait a moment and try again."
                )
            else:
                error_detail = e.response.text
                raise LLMProviderError(f"OpenRouter HTTP error: {e.response.status_code} - {error_detail}")
        except Exception as e:
            raise LLMProviderError(f"OpenRouter error: {e}")

    def is_available(self) -> bool:
        """
        Проверяет доступен ли OpenRouter API.

        Returns:
            True если API доступен и ключ валиден
        """
        try:
            # Синхронный запрос для простоты
            with httpx.Client(timeout=5) as client:
                response = client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code == 200
        except:
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Получает список доступных моделей.

        Returns:
            Список моделей с информацией
        """
        try:
            response = await self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            data = response.json()

            return data.get("data", [])
        except Exception as e:
            raise LLMProviderError(f"Cannot list models: {e}")

    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Получает информацию о конкретной модели.

        Args:
            model: Название модели

        Returns:
            Информация о модели
        """
        models = await self.list_models()

        for m in models:
            if m.get("id") == model:
                return m

        raise LLMProviderError(f"Model {model} not found")

    async def close(self):
        """Закрывает HTTP клиент"""
        await self.client.aclose()


# Расширенный Message класс с поддержкой reasoning
class ReasoningMessage(Message):
    """Сообщение с поддержкой reasoning_details"""

    def __init__(self, role: str, content: str, reasoning_details: Optional[Dict] = None):
        super().__init__(role, content)
        self.reasoning_details = reasoning_details

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.reasoning_details:
            result["reasoning_details"] = self.reasoning_details
        return result


# Вспомогательная функция для удобного создания
async def create_openrouter_provider(
        model: str = "openai/gpt-oss-20b:free",
        api_key: Optional[str] = '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',
        enable_reasoning: bool = False,
        **kwargs
) -> OpenRouterProvider:
    """
    Создаёт и настраивает OpenRouter провайдер.

    Args:
        model: Название модели
        api_key: API ключ
        enable_reasoning: Включить reasoning
        **kwargs: Дополнительные параметры

    Returns:
        Настроенный OpenRouterProvider
    """
    return OpenRouterProvider(
        model=model,
        api_key=api_key,
        enable_reasoning=enable_reasoning,
        **kwargs
    )


# Пример использования
if __name__ == "__main__":
    import asyncio


    async def test_openrouter():
        """Тестовая функция"""

        print("=" * 80)
        print("OPENROUTER PROVIDER TEST")
        print("=" * 80)

        # API ключ (замените на свой!)
        API_KEY = "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"

        # Создаём провайдер
        print("\n📍 Creating provider...")
        try:
            provider = create_openrouter_provider(
                model="openai/gpt-oss-20b:free",
                api_key=API_KEY,
                temperature=0.7,
                enable_reasoning=True
            )
            print(f"✅ Provider ready: {provider}")
        except LLMProviderError as e:
            print(f"❌ Error: {e}")
            return

        # Проверяем доступность
        if provider.is_available():
            print("✅ API is available")
        else:
            print("⚠️  API check failed")

        # Тест 1: Простой запрос
        print("\n" + "─" * 80)
        print("TEST 1: Simple question")
        print("─" * 80)

        response = await provider.generate_simple(
            user_message="What is 2+2? Answer in one sentence.",
            system_prompt="You are a helpful AI assistant."
        )

        print(f"\n🤖 Response: {response.content}")

        # Тест 2: Reasoning (если поддерживается)
        print("\n" + "─" * 80)
        print("TEST 2: Reasoning test")
        print("─" * 80)

        response = await provider.generate_simple(
            user_message="How many r's are in the word 'strawberry'?"
        )

        print(f"\n🤖 Response: {response.content}")

        if hasattr(response, 'reasoning_details') and response.reasoning_details:
            print(f"\n🧠 Reasoning details: {response.reasoning_details}")

        # Тест 3: Продолжение reasoning
        print("\n" + "─" * 80)
        print("TEST 3: Continue reasoning")
        print("─" * 80)

        # Создаём сообщение с reasoning_details
        messages = [
            Message(role="user", content="How many r's are in the word 'strawberry'?"),
            ReasoningMessage(
                role="assistant",
                content=response.content,
                reasoning_details=getattr(response, 'reasoning_details', None)
            ),
            Message(role="user", content="Are you sure? Think carefully.")
        ]

        response2 = await provider.generate(messages, preserve_reasoning=True)
        print(f"\n🤖 Response: {response2.content}")

        # Тест 4: JSON генерация
        print("\n" + "─" * 80)
        print("TEST 4: JSON generation")
        print("─" * 80)

        response = await provider.generate_simple(
            user_message="""Generate a JSON object with these fields:
- name: random person name
- age: random age 20-50
- hobby: random hobby

Return ONLY JSON, no other text.""",
            system_prompt="You are a JSON generator. Return only valid JSON."
        )

        print(f"\n🤖 Response:\n{response.content}")

        # Пробуем распарсить JSON
        try:
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

        # Закрываем
        await provider.close()
        print("\n✅ All tests completed!")


    asyncio.run(test_openrouter())