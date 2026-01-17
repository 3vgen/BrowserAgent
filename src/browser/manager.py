"""
Browser Manager - управление браузером и выполнение действий
"""

import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from src.browser.dom_extractor import DOMExtractor, Element


class BrowserManager:
    """Управляет браузером и выполняет примитивные действия"""

    def __init__(
            self,
            headless: bool = False,
            slow_mo: int = 300,
            profile_dir: str = "./data/browser_profile",
            viewport: Dict[str, int] = None
    ):
        """
        Args:
            headless: Запускать браузер без GUI
            slow_mo: Замедление в миллисекундах (для наблюдения)
            profile_dir: Директория для сохранения профиля браузера
            viewport: Размер окна браузера
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.profile_dir = Path(profile_dir)
        self.viewport = viewport or {"width": 1280, "height": 720}

        self.playwright = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

        # Кэш последних извлечённых элементов
        self._last_elements: List[Element] = []

    async def start(self) -> None:
        """Запускает браузер с persistent context"""
        # Создаём директорию для профиля если не существует
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        self.playwright = await async_playwright().start()

        # Persistent context сохраняет cookies, localStorage и т.д.
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            viewport=self.viewport,
            slow_mo=self.slow_mo,
            args=[
                '--disable-blink-features=AutomationControlled',  # Скрываем автоматизацию
            ]
        )

        # Берём первую страницу или создаём новую
        if self.context.pages:
            self.page = self.context.pages[0]
        else:
            self.page = await self.context.new_page()

        print(f"🌐 Browser started (profile: {self.profile_dir})")

    async def close(self) -> None:
        """Закрывает браузер"""
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()
        print("🔒 Browser closed")

    async def navigate(self, url: str, timeout: int = 30000) -> Dict[str, Any]:
        """
        Переходит на URL.

        Args:
            url: URL для перехода
            timeout: Таймаут в миллисекундах

        Returns:
            {"success": bool, "url": str, "title": str, "error": str}
        """
        try:
            await self.page.goto(url, wait_until='domcontentloaded', timeout=timeout)
            await self.page.wait_for_load_state('networkidle', timeout=10000)

            return {
                "success": True,
                "url": self.page.url,
                "title": await self.page.title(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def get_page_state(self) -> Dict[str, Any]:
        """
        Получает текущее состояние страницы.

        Returns:
            {
                "url": str,
                "title": str,
                "elements": List[Element],
                "elements_formatted": str  # Для отправки в LLM
            }
        """
        # Извлекаем элементы
        elements = await DOMExtractor.extract(self.page)

        # Приоритизируем (топ 100)
        prioritized = DOMExtractor.prioritize_elements(elements, limit=100)

        # Сохраняем в кэш для последующего использования
        self._last_elements = prioritized

        # Форматируем для LLM
        formatted = DOMExtractor.format_for_llm(prioritized)

        return {
            "url": self.page.url,
            "title": await self.page.title(),
            "elements": prioritized,
            "elements_formatted": formatted
        }

    async def click(self, element_id: str, timeout: int = 5000) -> Dict[str, Any]:
        """
        Кликает по элементу.

        Args:
            element_id: ID элемента (elem_X)
            timeout: Таймаут в миллисекундах

        Returns:
            {"success": bool, "element_id": str, "error": str}
        """
        try:
            selector = f'[data-agent-id="{element_id}"]'

            # Скроллим к элементу если нужно
            await self.page.locator(selector).scroll_into_view_if_needed()

            # Кликаем
            await self.page.click(selector, timeout=timeout)

            # Даём время на реакцию страницы
            await asyncio.sleep(1)

            return {
                "success": True,
                "element_id": element_id
            }
        except Exception as e:
            return {
                "success": False,
                "element_id": element_id,
                "error": str(e)
            }

    async def type_text(
            self,
            element_id: str,
            text: str,
            clear_first: bool = True,
            timeout: int = 5000
    ) -> Dict[str, Any]:
        """
        Вводит текст в элемент.

        Args:
            element_id: ID элемента
            text: Текст для ввода
            clear_first: Очистить поле перед вводом
            timeout: Таймаут в миллисекундах

        Returns:
            {"success": bool, "element_id": str, "text": str, "error": str}
        """
        try:
            selector = f'[data-agent-id="{element_id}"]'

            # Скроллим к элементу
            await self.page.locator(selector).scroll_into_view_if_needed()

            if clear_first:
                # Очищаем поле
                await self.page.fill(selector, "", timeout=timeout)

            # Вводим текст (медленно, как человек)
            await self.page.type(selector, text, delay=50, timeout=timeout)

            return {
                "success": True,
                "element_id": element_id,
                "text": text
            }
        except Exception as e:
            return {
                "success": False,
                "element_id": element_id,
                "error": str(e)
            }

    async def press_key(self, key: str) -> Dict[str, Any]:
        """
        Нажимает клавишу (Enter, Escape, etc).

        Args:
            key: Название клавиши

        Returns:
            {"success": bool, "key": str}
        """
        try:
            await self.page.keyboard.press(key)
            await asyncio.sleep(0.5)

            return {
                "success": True,
                "key": key
            }
        except Exception as e:
            return {
                "success": False,
                "key": key,
                "error": str(e)
            }

    async def scroll(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        """
        Прокручивает страницу.

        Args:
            direction: "down" или "up"
            amount: Количество пикселей

        Returns:
            {"success": bool, "direction": str}
        """
        try:
            delta = amount if direction == "down" else -amount
            await self.page.evaluate(f"window.scrollBy(0, {delta})")
            await asyncio.sleep(0.5)

            return {
                "success": True,
                "direction": direction,
                "amount": amount
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def wait(self, seconds: float) -> Dict[str, Any]:
        """
        Ждёт указанное время.

        Args:
            seconds: Время ожидания в секундах

        Returns:
            {"success": bool, "seconds": float}
        """
        await asyncio.sleep(seconds)
        return {
            "success": True,
            "seconds": seconds
        }

    async def screenshot(self, path: str = "screenshot.png") -> Dict[str, Any]:
        """
        Делает скриншот страницы.

        Args:
            path: Путь для сохранения

        Returns:
            {"success": bool, "path": str}
        """
        try:
            await self.page.screenshot(path=path, full_page=True)
            return {
                "success": True,
                "path": path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_element_by_id(self, element_id: str) -> Optional[Element]:
        """
        Находит элемент по ID в кэше последних извлечённых элементов.

        Args:
            element_id: ID элемента

        Returns:
            Element или None
        """
        for elem in self._last_elements:
            if elem.id == element_id:
                return elem
        return None


# Пример использования
if __name__ == "__main__":
    async def test_browser():
        """Тестовая функция"""
        browser = BrowserManager(headless=False, slow_mo=500)

        try:
            # Запускаем браузер
            await browser.start()

            # Переходим на сайт
            print("\n📍 Navigating to Wikipedia...")
            result = await browser.navigate("https://en.wikipedia.org")
            print(f"✅ Loaded: {result['title']}")

            # Получаем состояние страницы
            print("\n📍 Extracting page state...")
            state = await browser.get_page_state()
            print(f"✅ Found {len(state['elements'])} elements")
            print("\n" + state['elements_formatted'])

            # Интерактивный режим
            print("\n" + "=" * 80)
            print("INTERACTIVE MODE")
            print("=" * 80)
            print("\nCommands:")
            print("  list          - show elements again")
            print("  click <id>    - click element")
            print("  type <id> <text> - type text")
            print("  scroll [up|down] - scroll page")
            print("  screenshot    - take screenshot")
            print("  url <url>     - navigate to URL")
            print("  exit          - quit")
            print()

            while True:
                cmd = input("💻 > ").strip()

                if not cmd:
                    continue

                parts = cmd.split(maxsplit=2)
                action = parts[0].lower()

                if action == "exit":
                    break

                elif action == "list":
                    state = await browser.get_page_state()
                    print(state['elements_formatted'])

                elif action == "click" and len(parts) > 1:
                    element_id = parts[1]
                    result = await browser.click(element_id)
                    if result['success']:
                        print(f"✅ Clicked {element_id}")
                        # Обновляем состояние
                        await asyncio.sleep(1)
                        state = await browser.get_page_state()
                        print(state['elements_formatted'])
                    else:
                        print(f"❌ Error: {result.get('error')}")

                elif action == "type" and len(parts) > 2:
                    element_id = parts[1]
                    text = parts[2]
                    result = await browser.type_text(element_id, text)
                    if result['success']:
                        print(f"✅ Typed '{text}' into {element_id}")
                    else:
                        print(f"❌ Error: {result.get('error')}")

                elif action == "scroll":
                    direction = parts[1] if len(parts) > 1 else "down"
                    result = await browser.scroll(direction)
                    if result['success']:
                        print(f"✅ Scrolled {direction}")
                        state = await browser.get_page_state()
                        print(state['elements_formatted'])
                    else:
                        print(f"❌ Error: {result.get('error')}")

                elif action == "screenshot":
                    result = await browser.screenshot()
                    if result['success']:
                        print(f"✅ Screenshot saved: {result['path']}")
                    else:
                        print(f"❌ Error: {result.get('error')}")

                elif action == "url" and len(parts) > 1:
                    url = parts[1]
                    result = await browser.navigate(url)
                    if result['success']:
                        print(f"✅ Navigated to: {result['title']}")
                        state = await browser.get_page_state()
                        print(state['elements_formatted'])
                    else:
                        print(f"❌ Error: {result.get('error')}")

                else:
                    print("❌ Unknown command")

        finally:
            await browser.close()


    asyncio.run(test_browser())