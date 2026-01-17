"""
DOM Extractor - извлекает интерактивные элементы со страницы
Это "глаза" агента - так он видит веб-страницу
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from playwright.async_api import Page


@dataclass
class Element:
    """Представление интерактивного элемента на странице"""

    id: str  # Уникальный ID (elem_0, elem_1, ...)
    tag: str  # HTML тег (button, input, a, ...)
    type: Optional[str] = None  # Тип для input (text, submit, ...)
    text: str = ""  # Видимый текст элемента
    placeholder: Optional[str] = None
    href: Optional[str] = None  # Для ссылок
    aria_label: Optional[str] = None
    value: Optional[str] = None  # Текущее значение для input
    selector: str = ""  # CSS селектор (для отладки)
    position: Dict[str, int] = None  # {x, y, width, height}
    is_in_viewport: bool = False  # Виден ли элемент на экране

    def __post_init__(self):
        if self.position is None:
            self.position = {"x": 0, "y": 0, "width": 0, "height": 0}

    def to_dict(self) -> Dict:
        """Конвертация в словарь для JSON"""
        return {
            "id": self.id,
            "tag": self.tag,
            "type": self.type,
            "text": self.text[:100] if self.text else "",  # Ограничиваем длину
            "placeholder": self.placeholder,
            "href": self.href[:100] if self.href else None,
            "aria_label": self.aria_label,
            "value": self.value,
            "is_in_viewport": self.is_in_viewport,
        }

    def __repr__(self) -> str:
        parts = [f"<Element {self.id} {self.tag}"]
        if self.text:
            parts.append(f'"{self.text[:30]}..."' if len(self.text) > 30 else f'"{self.text}"')
        if self.is_in_viewport:
            parts.append("visible")
        return " ".join(parts) + ">"


class DOMExtractor:
    """Извлекает интерактивные элементы со страницы"""

    # JavaScript код для выполнения в браузере
    EXTRACTION_SCRIPT = """
    () => {
        const elements = [];
        let elementCounter = 0;

        // Проверка видимости элемента
        function isElementVisible(elem) {
            if (!elem.offsetParent && elem.tagName !== 'BODY') {
                return false;
            }

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

        // Генерация простого CSS селектора
        function generateSelector(elem) {
            // Если есть ID - используем его
            if (elem.id) {
                return `#${elem.id}`;
            }

            // Строим путь через теги и классы
            const path = [];
            let current = elem;

            for (let i = 0; i < 3 && current && current.nodeType === Node.ELEMENT_NODE; i++) {
                let selector = current.tagName.toLowerCase();

                // Добавляем первые 2 класса если есть
                if (current.className && typeof current.className === 'string') {
                    const classes = current.className
                        .trim()
                        .split(/\\s+/)
                        .filter(c => c && !c.match(/^[0-9]/)) // Исключаем классы начинающиеся с цифр
                        .slice(0, 2);

                    if (classes.length > 0) {
                        selector += '.' + classes.join('.');
                    }
                }

                path.unshift(selector);
                current = current.parentElement;
            }

            return path.join(' > ');
        }

        // Получение текста элемента
        function getElementText(elem) {
            // Для input/textarea берём placeholder или value
            if (elem.tagName === 'INPUT' || elem.tagName === 'TEXTAREA') {
                return elem.placeholder || elem.value || '';
            }

            // Для других элементов - innerText
            let text = elem.innerText || elem.textContent || '';

            // Очищаем от лишних пробелов и переносов
            text = text.replace(/\\s+/g, ' ').trim();

            return text;
        }

        // Селекторы интерактивных элементов
        const interactiveSelectors = [
            'a[href]',              // Ссылки
            'button',               // Кнопки
            'input:not([type="hidden"])', // Поля ввода (кроме скрытых)
            'textarea',             // Текстовые области
            'select',               // Выпадающие списки
            '[role="button"]',      // Элементы с ролью кнопки
            '[role="link"]',        // Элементы с ролью ссылки
            '[role="textbox"]',     // Элементы с ролью текстового поля
            '[onclick]',            // Элементы с onclick
            '[contenteditable="true"]', // Редактируемый контент
            'h1', 'h2', 'h3',       // Заголовки (для контекста)
        ];

        // Находим все интерактивные элементы
        const foundElements = document.querySelectorAll(interactiveSelectors.join(','));

        foundElements.forEach(elem => {
            // Пропускаем невидимые элементы
            if (!isElementVisible(elem)) {
                return;
            }

            const rect = elem.getBoundingClientRect();
            const elementId = `elem_${elementCounter++}`;

            // Добавляем data-agent-id для последующего взаимодействия
            elem.setAttribute('data-agent-id', elementId);

            // Собираем информацию об элементе
            const elementInfo = {
                id: elementId,
                tag: elem.tagName.toLowerCase(),
                type: elem.type || null,
                text: getElementText(elem),
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

            elements.push(elementInfo);
        });

        return elements;
    }
    """

    @staticmethod
    async def extract(page: Page) -> List[Element]:
        """
        Извлекает все интерактивные элементы со страницы.

        Args:
            page: Playwright Page объект

        Returns:
            Список Element объектов
        """
        # Выполняем JavaScript в браузере
        raw_elements = await page.evaluate(DOMExtractor.EXTRACTION_SCRIPT)

        # Конвертируем в Element объекты
        elements = []
        for raw in raw_elements:
            element = Element(
                id=raw['id'],
                tag=raw['tag'],
                type=raw.get('type'),
                text=raw.get('text', ''),
                placeholder=raw.get('placeholder'),
                href=raw.get('href'),
                aria_label=raw.get('ariaLabel'),
                value=raw.get('value'),
                selector=raw.get('selector', ''),
                position=raw.get('position', {}),
                is_in_viewport=raw.get('isInViewport', False)
            )
            elements.append(element)

        return elements

    @staticmethod
    def prioritize_elements(elements: List[Element], limit: int = 100) -> List[Element]:
        """
        Приоритизирует элементы для отправки в LLM.

        Стратегия:
        1. Сначала элементы в viewport
        2. Интерактивные элементы (кнопки, ссылки, поля ввода) важнее заголовков
        3. Ограничиваем общее количество

        Args:
            elements: Список всех элементов
            limit: Максимальное количество элементов

        Returns:
            Отфильтрованный и отсортированный список
        """
        # Разделяем на категории
        in_viewport = []
        out_viewport = []

        for elem in elements:
            if elem.is_in_viewport:
                in_viewport.append(elem)
            else:
                out_viewport.append(elem)

        # Сортируем по важности (интерактивные элементы важнее)
        def importance_score(elem: Element) -> int:
            score = 0

            # Интерактивные элементы важнее
            if elem.tag in ['button', 'input', 'textarea', 'select', 'a']:
                score += 10

            # Элементы с текстом важнее
            if elem.text and len(elem.text) > 3:
                score += 5

            # Элементы с aria-label важны
            if elem.aria_label:
                score += 3

            return score

        in_viewport.sort(key=importance_score, reverse=True)
        out_viewport.sort(key=importance_score, reverse=True)

        # Берём сначала из viewport, потом остальные
        result = in_viewport + out_viewport

        return result[:limit]

    @staticmethod
    def format_for_llm(elements: List[Element]) -> str:
        """
        Форматирует элементы в текст для отправки в LLM.

        Цель: минимизировать токены, но сохранить всю важную информацию.

        Args:
            elements: Список элементов

        Returns:
            Отформатированная строка
        """
        lines = ["=== INTERACTIVE ELEMENTS ON PAGE ===\n"]

        # Группируем по типам для статистики
        by_tag = {}
        for elem in elements:
            if elem.tag not in by_tag:
                by_tag[elem.tag] = 0
            by_tag[elem.tag] += 1

        # Статистика
        lines.append("Page statistics:")
        for tag, count in sorted(by_tag.items()):
            lines.append(f"  {tag}: {count}")

        lines.append("\nElements (visible first):\n")

        # Форматируем каждый элемент компактно
        for elem in elements:
            parts = []

            # ID и тег обязательны
            parts.append(f"[{elem.id}]")
            parts.append(elem.tag.upper())

            # Тип для input
            if elem.type:
                parts.append(f"type={elem.type}")

            # Текст (самое важное!)
            if elem.text:
                text = elem.text[:50]
                if len(elem.text) > 50:
                    text += "..."
                parts.append(f'text="{text}"')

            # Placeholder для полей ввода
            if elem.placeholder:
                parts.append(f'placeholder="{elem.placeholder[:30]}"')

            # Ссылка
            if elem.href:
                href = elem.href[:60]
                if len(elem.href) > 60:
                    href += "..."
                parts.append(f"href={href}")

            # Aria label
            if elem.aria_label:
                parts.append(f'label="{elem.aria_label[:30]}"')

            # Маркер видимости
            if elem.is_in_viewport:
                parts.append("✓visible")
            else:
                parts.append("below-fold")

            lines.append(" | ".join(parts))

        lines.append(f"\nTotal: {len(elements)} interactive elements")

        return "\n".join(lines)


# Пример использования (для тестирования модуля отдельно)
if __name__ == "__main__":
    import asyncio
    from playwright.async_api import async_playwright


    async def test_extraction():
        """Тестовая функция"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()

            # Тестируем на Wikipedia
            await page.goto("https://en.wikipedia.org")
            await page.wait_for_load_state('networkidle')

            # Извлекаем элементы
            elements = await DOMExtractor.extract(page)
            print(f"\n✅ Found {len(elements)} elements\n")

            # Приоритизируем
            prioritized = DOMExtractor.prioritize_elements(elements, limit=30)
            print(f"✅ Prioritized to {len(prioritized)} elements\n")

            # Форматируем для LLM
            formatted = DOMExtractor.format_for_llm(prioritized)
            print(formatted)

            # Показываем первые 5 элементов как объекты
            print("\n" + "=" * 80)
            print("First 5 elements as objects:")
            print("=" * 80 + "\n")
            for elem in prioritized[:5]:
                print(elem)

            await browser.close()


    asyncio.run(test_extraction())