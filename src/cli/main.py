"""
CLI - командный интерфейс для AI Browser Agent
"""

import asyncio
import sys
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm.ollama_provider import create_ollama_provider
from src.browser.manager import BrowserManager
from src.agents.orchestrator import Orchestrator


async def main():
    """Главная функция CLI"""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                    🤖 AI BROWSER AGENT v0.1                                ║
║                                                                            ║
║                    Powered by Ollama + Qwen 2.5                            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Настройки
    MODEL = "qwen2.5:7b"
    MAX_STEPS = 20

    print(f"⚙️  Configuration:")
    print(f"   Model: {MODEL}")
    print(f"   Max steps: {MAX_STEPS}")
    print()

    # Инициализация
    print("🚀 Starting up...\n")

    try:
        # LLM
        print("📍 Setting up LLM provider...")
        llm = await create_ollama_provider(model=MODEL)
        print("✅ LLM ready")

        # Browser
        print("📍 Starting browser...")
        browser = BrowserManager(
            headless=False,
            slow_mo=300,
            profile_dir="./data/browser_profile"
        )
        await browser.start()
        print("✅ Browser ready")

        # Agent (now using Orchestrator with sub-agents)
        print("📍 Creating orchestrator with sub-agents...")
        orchestrator = Orchestrator(
            llm_provider=llm,
            browser=browser,
            max_steps=MAX_STEPS,
            verbose=True
        )
        print("✅ Orchestrator ready")
        print("   👁️  Vision Agent - analyzes pages")
        print("   🤖 Action Agent - decides actions\n")

    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
        return

    # Главный цикл
    print("=" * 80)
    print("🎯 READY! Enter your tasks below.")
    print("=" * 80)
    print("\n💡 Examples:")
    print("   • Search for 'AI news' on Google")
    print("   • Go to Wikipedia and find article about Python")
    print("   • Open Hacker News and find top story")
    print("\n💬 Commands:")
    print("   • Type your task and press Enter")
    print("   • 'exit' or 'quit' to stop")
    print("   • 'url <address>' to navigate somewhere first")
    print()

    try:
        while True:
            # Получаем задачу от пользователя
            print("─" * 80)
            user_input = input("🎯 Your task: ").strip()

            if not user_input:
                continue

            # Команды выхода
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break

            # Команда навигации
            if user_input.lower().startswith('url '):
                url = user_input[4:].strip()
                print(f"\n🌐 Navigating to {url}...")
                result = await browser.navigate(url)
                if result['success']:
                    print(f"✅ Loaded: {result['title']}")
                else:
                    print(f"❌ Failed: {result.get('error')}")
                continue

            # Выполняем задачу через оркестратор
            result = await orchestrator.execute_task(
                goal=user_input,
                start_url=None  # Продолжаем с текущей страницы
            )

            # Показываем результат
            print("\n" + "=" * 80)
            if result['success']:
                print(f"✅ SUCCESS")
                print(f"📋 Result: {result.get('result')}")
            else:
                print(f"❌ FAILED")
                print(f"⚠️  Error: {result.get('error')}")
            print(f"📊 Steps completed: {result.get('steps_completed')}")
            print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\n⏸️  Interrupted by user")

    finally:
        # Cleanup
        print("\n🔒 Shutting down...")
        await browser.close()
        await llm.close()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())

    # url https://www.google.com
    # go to wikipedia and find artical about 'Король и Шут'
