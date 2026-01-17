"""
ШАГ 1: Базовое управление браузером и извлечение DOM
Цель: Научиться видеть страницу глазами агента
"""

import asyncio
from playwright.async_api import async_playwright, Page
from typing import List, Dict
import json


class DOMExtractor:
    """Извлекает интерактивные элементы со страницы"""

    @staticmethod
    async def get_interactive_elements(page: Page) -> List[Dict]:
        """
        Выполняет JavaScript на странице для извлечения всех интерактивных элементов.
        Возвращает список элементов с их характеристиками.
        """

        # Этот JavaScript код будет выполнен прямо в браузере
        extraction_script = """
        () => {
            const elements = [];
            let elementCounter = 0;

            // Проверка видимости элемента
            function isElementVisible(elem) {
                const style = window.getComputedStyle(elem);
                const rect = elem.getBoundingClientRect();

                return (
                    style.display !== 'none' &&
                    style.visibility !== 'hidden' &&
                    style.opacity !== '0' &&
                    rect.width > 0 &&
                    rect.height > 0
                );
            }

            // Генерация простого селектора
            function generateSelector(elem) {
                // Если есть ID - используем его
                if (elem.id) {
                    return `#${elem.id}`;
                }

                // Иначе строим путь через теги и классы
                let path = [];
                let current = elem;

                for (let i = 0; i < 3 && current && current.nodeType === Node.ELEMENT_NODE; i++) {
                    let selector = current.tagName.toLowerCase();

                    // Добавляем первые 2 класса если есть
                    if (current.className && typeof current.className === 'string') {
                        const classes = current.className.trim().split(/\\s+/).slice(0, 2);
                        if (classes.length > 0 && classes[0]) {
                            selector += '.' + classes.join('.');
                        }
                    }

                    path.unshift(selector);
                    current = current.parentElement;
                }

                return path.join(' > ');
            }

            // Находим все интерактивные элементы
            const selectors = [
                'a[href]',           // Ссылки
                'button',            // Кнопки
                'input',             // Поля ввода
                'textarea',          // Текстовые области
                'select',            // Выпадающие списки
                '[role="button"]',   // Элементы с ролью кнопки
                '[onclick]',         // Элементы с onclick
                'h1', 'h2', 'h3'     // Заголовки (для понимания структуры)
            ];

            const foundElements = document.querySelectorAll(selectors.join(','));

            foundElements.forEach(elem => {
                if (!isElementVisible(elem)) {
                    return; // Пропускаем невидимые
                }

                const rect = elem.getBoundingClientRect();
                const elementId = `elem_${elementCounter++}`;

                // Добавляем специальный атрибут для идентификации
                elem.setAttribute('data-agent-id', elementId);

                // Собираем информацию об элементе
                const info = {
                    id: elementId,
                    tag: elem.tagName.toLowerCase(),
                    type: elem.type || null,
                    text: (elem.innerText || elem.textContent || '').trim().substring(0, 100),
                    placeholder: elem.placeholder || null,
                    href: elem.href || null,
                    ariaLabel: elem.getAttribute('aria-label'),
                    value: elem.value || null,
                    selector: generateSelector(elem),
                    position: {
                        x: Math.round(rect.x),
                        y: Math.round(rect.y),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    },
                    isInViewport: (
                        rect.top >= 0 && 
                        rect.top <= window.innerHeight &&
                        rect.left >= 0 &&
                        rect.left <= window.innerWidth
                    )
                };

                elements.push(info);
            });

            return elements;
        }
        """

        # Выполняем скрипт и получаем результат
        elements = await page.evaluate(extraction_script)
        return elements

    @staticmethod
    def print_elements(elements: List[Dict], limit: int = 30):
        """Красиво печатает элементы для человека"""

        print(f"\n{'=' * 80}")
        print(f"НАЙДЕНО ИНТЕРАКТИВНЫХ ЭЛЕМЕНТОВ: {len(elements)}")
        print(f"{'=' * 80}\n")

        # Группируем по типам
        by_type = {}
        for elem in elements:
            tag = elem['tag']
            if tag not in by_type:
                by_type[tag] = []
            by_type[tag].append(elem)

        print("📊 Статистика по типам:")
        for tag, items in sorted(by_type.items()):
            print(f"   {tag}: {len(items)}")

        print(f"\n{'─' * 80}")
        print(f"ПЕРВЫЕ {min(limit, len(elements))} ЭЛЕМЕНТОВ:")
        print(f"{'─' * 80}\n")

        # Показываем первые N элементов
        for i, elem in enumerate(elements[:limit]):
            viewport_marker = "👁️ " if elem['isInViewport'] else "   "

            print(f"{viewport_marker}[{elem['id']}] {elem['tag'].upper()}", end='')

            if elem.get('type'):
                print(f" (type={elem['type']})", end='')

            print()

            if elem.get('text'):
                print(f"     Text: \"{elem['text'][:60]}...\"" if len(
                    elem['text']) > 60 else f"     Text: \"{elem['text']}\"")

            if elem.get('placeholder'):
                print(f"     Placeholder: \"{elem['placeholder']}\"")

            if elem.get('href'):
                href_display = elem['href'][:50] + "..." if len(elem['href']) > 50 else elem['href']
                print(f"     Link: {href_display}")

            if elem.get('ariaLabel'):
                print(f"     Label: \"{elem['ariaLabel']}\"")

            print(f"     Position: ({elem['position']['x']}, {elem['position']['y']})")
            print()

        if len(elements) > limit:
            print(f"... и ещё {len(elements) - limit} элементов\n")


class SimpleBrowser:
    """Простой менеджер браузера"""

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    async def start(self):
        """Запускает браузер"""
        print("🚀 Запуск браузера...")

        self.playwright = await async_playwright().start()

        # Запускаем браузер с сохранением профиля
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir="./browser_profile",
            headless=False,  # Видимый браузер!
            viewport={'width': 1280, 'height': 720},
            slow_mo=300,  # Замедляем на 300мс для наблюдения
        )

        # Берём первую страницу или создаём новую
        if self.context.pages:
            self.page = self.context.pages[0]
        else:
            self.page = await self.context.new_page()

        print("✅ Браузер запущен!\n")

    async def navigate(self, url: str):
        """Переходит на URL"""
        print(f"🌐 Переход на: {url}")
        await self.page.goto(url, wait_until='domcontentloaded')
        await self.page.wait_for_load_state('networkidle', timeout=10000)
        print("✅ Страница загружена\n")

    async def analyze_page(self):
        """Анализирует текущую страницу"""
        print(f"📄 Анализ страницы...")
        print(f"   URL: {self.page.url}")
        print(f"   Заголовок: {await self.page.title()}")

        # Извлекаем элементы
        elements = await DOMExtractor.get_interactive_elements(self.page)

        # Показываем что нашли
        DOMExtractor.print_elements(elements)

        return elements

    async def click_element(self, element_id: str):
        """Кликает по элементу по его ID"""
        print(f"🖱️  Клик по элементу: {element_id}")

        try:
            # Используем data-agent-id который мы установили в JavaScript
            await self.page.click(f'[data-agent-id="{element_id}"]', timeout=5000)
            await asyncio.sleep(1)  # Ждём реакции страницы
            print("✅ Клик выполнен\n")
            return True
        except Exception as e:
            print(f"❌ Ошибка клика: {e}\n")
            return False

    async def type_text(self, element_id: str, text: str):
        """Вводит текст в элемент"""
        print(f"⌨️  Ввод текста в {element_id}: \"{text}\"")

        try:
            await self.page.fill(f'[data-agent-id="{element_id}"]', text, timeout=5000)
            print("✅ Текст введён\n")
            return True
        except Exception as e:
            print(f"❌ Ошибка ввода: {e}\n")
            return False

    async def scroll_down(self):
        """Прокручивает страницу вниз"""
        print("📜 Прокрутка вниз...")
        await self.page.evaluate('window.scrollBy(0, 500)')
        await asyncio.sleep(0.5)
        print("✅ Прокручено\n")

    async def close(self):
        """Закрывает браузер"""
        print("\n🔒 Закрытие браузера...")
        if self.playwright:
            await self.context.close()
            await self.playwright.stop()
        print("👋 Готово!")


async def demo():
    """Демонстрация работы"""

    browser = SimpleBrowser()

    try:
        await browser.start()

        print("=" * 80)
        print("ДЕМОНСТРАЦИЯ: Извлечение элементов со страницы")
        print("=" * 80)
        print()

        # # 1. Открываем простую страницу
        # print("📍 ТЕСТ 1: Простая страница (Example.com)")
        # print("─" * 80)
        # await browser.navigate("https://example.com")
        # elements = await browser.analyze_page()
        #
        # input("\n⏸️  Нажмите Enter для продолжения...")

        # 2. Открываем более сложную страницу
        print("\n" + "=" * 80)
        print("📍 ТЕСТ 2: Более сложная страница (Wikipedia)")
        print("─" * 80)
        await browser.navigate("https://en.wikipedia.org")
        elements = await browser.analyze_page()

        input("\n⏸️  Нажмите Enter для продолжения...")

        # 3. Интерактивный режим
        print("\n" + "=" * 80)
        print("📍 ИНТЕРАКТИВНЫЙ РЕЖИМ")
        print("=" * 80)
        print("\nТеперь вы можете протестировать действия вручную!")
        print("\nКоманды:")
        print("  url <адрес>     - открыть страницу")
        print("  analyze         - проанализировать текущую страницу")
        print("  click <elem_id> - кликнуть по элементу")
        print("  type <elem_id> <text> - ввести текст")
        print("  scroll          - прокрутить вниз")
        print("  exit            - выход")
        print()

        current_elements = elements

        while True:
            command = input("💻 Команда: ").strip()

            if not command:
                continue

            parts = command.split(maxsplit=2)
            cmd = parts[0].lower()

            if cmd == 'exit':
                break

            elif cmd == 'url' and len(parts) > 1:
                await browser.navigate(parts[1])
                current_elements = await browser.analyze_page()

            elif cmd == 'analyze':
                current_elements = await browser.analyze_page()

            elif cmd == 'click' and len(parts) > 1:
                await browser.click_element(parts[1])
                # После клика переанализируем страницу
                await asyncio.sleep(1)
                current_elements = await browser.analyze_page()

            elif cmd == 'type' and len(parts) > 2:
                element_id = parts[1]
                text = parts[2]
                await browser.type_text(element_id, text)

            elif cmd == 'scroll':
                await browser.scroll_down()
                current_elements = await browser.analyze_page()

            else:
                print("❌ Неизвестная команда или не хватает параметров\n")

    finally:
        await browser.close()


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                   ШАГ 1: БАЗОВОЕ УПРАВЛЕНИЕ БРАУЗЕРОМ                      ║
║                                                                            ║
║  Это первый шаг в создании AI-агента.                                      ║
║  Мы учимся извлекать элементы со страницы и взаимодействовать с ними.      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    asyncio.run(demo())