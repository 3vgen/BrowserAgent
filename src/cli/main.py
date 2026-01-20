"""
CLI - командный интерфейс для AI Browser Agent
"""

import asyncio
import sys
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm.ollama_provider_request import create_ollama_provider
from src.llm.openrouter_provider import create_openrouter_provider

from src.browser.manager import BrowserManager
from src.agents.orchestrator import Orchestrator


async def main():
    """Главная функция CLI"""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                    🤖 AI BROWSER AGENT v0.2                                ║
║                                                                            ║
║                    Powered by Planning + Vision + Action                   ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Настройки
    MODEL = "qwen2.5:7b"
    MAX_STEPS_PER_PLAN_STEP = 10
    MAX_TOTAL_STEPS = 50
    USE_PLANNING = True

    print(f"⚙️  Configuration:")
    print(f"   Model: {MODEL}")
    print(f"   Max actions per plan step: {MAX_STEPS_PER_PLAN_STEP}")
    print(f"   Max total actions: {MAX_TOTAL_STEPS}")
    print(f"   Planning enabled: {USE_PLANNING}")
    print()

    # Инициализация
    print("🚀 Starting up...\n")

    try:
        # LLM
        print("📍 Setting up LLM provider...")
        # llm = await create_ollama_provider(model=MODEL)
        llm = await create_openrouter_provider(model='mistralai/devstral-2512:free')
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

        # Orchestrator
        print("📍 Creating orchestrator with agents...")
        orchestrator = Orchestrator(
            llm_provider=llm,
            browser=browser,
            max_steps_per_plan_step=MAX_STEPS_PER_PLAN_STEP,
            max_total_steps=MAX_TOTAL_STEPS,
            use_planning=USE_PLANNING,
            verbose=True
        )
        print("✅ Orchestrator ready")

        if USE_PLANNING:
            print("   📋 Planning Agent - decomposes tasks into atomic steps")
        print("   👁️  Vision Agent - analyzes pages in context")
        print("   🤖 Action Agent - executes focused actions\n")

    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        print("\nMake sure your LLM provider is running:")
        print("  For Ollama: ollama serve")
        print("  For OpenRouter: check your API key")
        return

    # Главный цикл
    print("=" * 80)
    print("🎯 READY! Enter your tasks below.")
    print("=" * 80)
    print("\n💡 Examples:")
    print("   • Search for 'AI news' on Google")
    print("   • Go to Wikipedia and find article about Python")
    print("   • Order BBQ burger and fries on Yandex Lavka (just add to cart)")
    print("   • Find 3 AI engineer vacancies on hh.ru")
    print("\n💬 Commands:")
    print("   • Type your task and press Enter")
    print("   • 'exit' or 'quit' to stop")
    print("   • 'url <address>' to navigate somewhere first")
    print("   • 'stats' to see current orchestrator stats")
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

            # Команда статистики
            if user_input.lower() == 'stats':
                stats = orchestrator.get_stats()
                print("\n📊 Orchestrator Statistics:")
                print(f"   Total actions executed: {stats.get('total_steps', 0)}")

                if 'plan' in stats:
                    plan_stats = stats['plan']
                    progress = plan_stats.get('progress', {})
                    print(f"   Plan progress: {progress.get('completed', 0)}/{progress.get('total', 0)} steps")

                if 'action_agent' in stats:
                    action_stats = stats['action_agent']
                    print(f"   Success rate: {action_stats.get('success_rate', 0):.1%}")
                    print(f"   Actions by type: {action_stats.get('action_types', {})}")

                print()
                continue

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
            print(f"\n{'='*80}")
            print(f"🚀 Starting task execution...")
            print(f"{'='*80}\n")

            result = await orchestrator.execute_task(
                goal=user_input,
                start_url=None  # Продолжаем с текущей страницы
            )

            # Показываем детальный результат
            print("\n" + "=" * 80)
            print("📊 TASK RESULT")
            print("=" * 80)

            if result.success:
                print(f"✅ STATUS: SUCCESS")
                print(f"📋 Result: {result.result}")
            else:
                print(f"❌ STATUS: FAILED")
                if result.error:
                    print(f"⚠️  Error: {result.error}")

            print(f"\n📈 Execution Metrics:")
            print(f"   Total actions: {result.steps_completed}")
            print(f"   Time elapsed: {result.total_time:.1f}s")

            if result.plan_steps_total > 0:
                print(f"   Plan steps: {result.plan_steps_completed}/{result.plan_steps_total}")
                completion_rate = (result.plan_steps_completed / result.plan_steps_total) * 100
                print(f"   Completion rate: {completion_rate:.0f}%")

            # Статистика агентов
            if result.stats:
                action_stats = result.stats.get('action_agent', {})
                if action_stats:
                    print(f"\n🤖 Action Agent:")
                    print(f"   Actions executed: {action_stats.get('total_actions', 0)}")
                    print(f"   Success rate: {action_stats.get('success_rate', 0):.1%}")

                    action_types = action_stats.get('action_types', {})
                    if action_types:
                        print(f"   Actions breakdown:")
                        for action_type, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True):
                            print(f"      • {action_type}: {count}")

                # План (если есть)
                if 'plan' in result.stats:
                    plan_data = result.stats['plan']
                    steps = plan_data.get('steps', [])
                    if steps:
                        print(f"\n📋 Plan Execution:")
                        for step in steps:
                            status_emoji = {
                                'completed': '✅',
                                'failed': '❌',
                                'in_progress': '▶️',
                                'pending': '⏸️',
                                'skipped': '⏭️'
                            }
                            emoji = status_emoji.get(step.get('status', 'pending'), '?')
                            desc = step.get('description', 'Unknown')
                            result_text = step.get('result', '')
                            result_info = f" → {result_text}" if result_text else ""
                            print(f"      {emoji} Step {step.get('step', '?')}: {desc}{result_info}")

            print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\n⏸️  Interrupted by user")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n🔒 Shutting down...")
        try:
            await browser.close()
            print("✅ Browser closed")
        except:
            pass

        try:
            await llm.close()
            print("✅ LLM provider closed")
        except:
            pass

        print("✅ Cleanup complete")


if __name__ == "__main__":
    """
    Example tasks to try:
    
    1. Simple search:
       url https://www.google.com
       Search for 'AI agents'
    
    2. Wikipedia article:
       Go to Wikipedia and find article about 'Король и Шут'
    
    3. Job search (complex multi-step):
       Найди 3 подходящие вакансии AI-инженера на hh.ru
    
    4. E-commerce (CRITICAL TEST - should separate items!):
       url https://lavka.yandex.ru
       Закажи мне BBQ-бургер и картошку фри, просто добавь в корзину
       # Добавь в корзину семнадцатый айфон на сайте big geek
       Expected plan:
       1. Search for BBQ burger
       2. Add BBQ burger to cart
       3. Search for fries
       4. Add fries to cart
       
       Should NOT search for "BBQ-бургер и картошку фри" together!
    """
    asyncio.run(main())