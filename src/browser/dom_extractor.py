"""
DOM Extractor - УЛУЧШЕННЫЙ экстрактор элементов

Проблемы которые решает:
1. Пропускает важные элементы (кнопки поиска, submit)
2. Берёт слишком много мусора (скрытые элементы)
3. Плохо определяет search inputs
4. Не видит динамически загруженные элементы

Решения:
1. Более умная проверка видимости
2. Специальная обработка search элементов
3. Ожидание динамического контента
4. Лучшие селекторы
"""

from typing import List, Dict
from dataclasses import dataclass
from playwright.async_api import Page
import asyncio


@dataclass
class Element:
    """Представление интерактивного элемента"""

    id: str
    tag: str
    type: str | None = None
    text: str = ""
    placeholder: str | None = None
    href: str | None = None
    aria_label: str | None = None
    value: str | None = None
    selector: str = ""
    position: Dict[str, int] | None = None
    is_in_viewport: bool = False
    role: str | None = None  # ARIA role
    name: str | None = None  # name attribute

    def __post_init__(self):
        if self.position is None:
            self.position = {"x": 0, "y": 0, "width": 0, "height": 0}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "tag": self.tag,
            "type": self.type,
            "text": self.text[:100] if self.text else "",
            "placeholder": self.placeholder,
            "href": self.href[:100] if self.href else None,
            "aria_label": self.aria_label,
            "role": self.role,
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
    """Улучшенный экстрактор DOM элементов"""

    # Более детальный JavaScript для извлечения
    EXTRACTION_SCRIPT = """
    async () => {
        const elements = [];
        let elementCounter = 0;
        
        // Ждём немного для динамического контента
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // УЛУЧШЕННАЯ проверка видимости
        function isElementVisible(elem) {
            // Базовые проверки
            if (!elem.offsetParent && elem.tagName !== 'BODY') {
                // Исключение для position: fixed элементов
                const style = window.getComputedStyle(elem);
                if (style.position !== 'fixed') {
                    return false;
                }
            }
            
            const style = window.getComputedStyle(elem);
            const rect = elem.getBoundingClientRect();
            
            // Проверяем CSS свойства
            if (style.display === 'none' || 
                style.visibility === 'hidden' || 
                parseFloat(style.opacity) === 0) {
                return false;
            }
            
            // Проверяем размеры (но разрешаем маленькие элементы для иконок)
            if (rect.width === 0 && rect.height === 0) {
                return false;
            }
            
            // Элемент должен иметь хоть какую-то площадь
            if (rect.width < 1 && rect.height < 1) {
                return false;
            }
            
            return true;
        }
        
        // Проверка что элемент действительно интерактивный
        function isInteractive(elem) {
            const tag = elem.tagName.toLowerCase();
            
            // Явно интерактивные теги
            if (['button', 'a', 'input', 'textarea', 'select'].includes(tag)) {
                return true;
            }
            
            // Элементы с ролями
            const role = elem.getAttribute('role');
            if (role && ['button', 'link', 'textbox', 'searchbox', 'combobox'].includes(role)) {
                return true;
            }
            
            // Элементы с обработчиками
            if (elem.onclick || elem.getAttribute('onclick')) {
                return true;
            }
            
            // Элементы с tabindex (focusable)
            if (elem.hasAttribute('tabindex')) {
                return true;
            }
            
            // Contenteditable
            if (elem.getAttribute('contenteditable') === 'true') {
                return true;
            }
            
            return false;
        }
        
        // Генерация улучшенного селектора
        function generateSelector(elem) {
            // Приоритет 1: ID
            if (elem.id && /^[a-zA-Z]/.test(elem.id)) {
                return `#${elem.id}`;
            }
            
            // Приоритет 2: name attribute
            if (elem.name) {
                return `${elem.tagName.toLowerCase()}[name="${elem.name}"]`;
            }
            
            // Приоритет 3: уникальные атрибуты
            const uniqueAttrs = ['data-testid', 'data-id', 'aria-label'];
            for (const attr of uniqueAttrs) {
                const value = elem.getAttribute(attr);
                if (value) {
                    return `${elem.tagName.toLowerCase()}[${attr}="${value}"]`;
                }
            }
            
            // Приоритет 4: путь через классы
            const path = [];
            let current = elem;
            
            for (let i = 0; i < 3 && current && current.nodeType === Node.ELEMENT_NODE; i++) {
                let selector = current.tagName.toLowerCase();
                
                // Добавляем полезные классы
                if (current.className && typeof current.className === 'string') {
                    const classes = current.className
                        .trim()
                        .split(/\\s+/)
                        .filter(c => c && !/^[0-9]/.test(c) && c.length < 30);
                    
                    if (classes.length > 0) {
                        // Берём самый специфичный класс
                        const bestClass = classes.find(c => 
                            c.includes('search') || 
                            c.includes('btn') || 
                            c.includes('button') ||
                            c.includes('input') ||
                            c.includes('link')
                        ) || classes[0];
                        
                        selector += '.' + bestClass;
                    }
                }
                
                path.unshift(selector);
                current = current.parentElement;
            }
            
            return path.join(' > ');
        }
        
        // Получение полного текста (включая псевдо-элементы)
        function getFullText(elem) {
            // Для input/textarea - placeholder или value
            if (elem.tagName === 'INPUT' || elem.tagName === 'TEXTAREA') {
                return elem.placeholder || elem.value || '';
            }
            
            // Для кнопок - innerText или value
            if (elem.tagName === 'BUTTON') {
                return elem.innerText || elem.textContent || elem.value || '';
            }
            
            // Для остальных - innerText
            let text = elem.innerText || elem.textContent || '';
            
            // Очищаем
            text = text.replace(/\\s+/g, ' ').trim();
            
            return text;
        }
        
        // РАСШИРЕННЫЙ список селекторов
        const selectors = [
            // Основные интерактивные
            'input:not([type="hidden"])',
            'button',
            'a[href]',
            'textarea',
            'select',
            
            // ARIA роли (важно для современных SPA!)
            '[role="button"]',
            '[role="link"]',
            '[role="textbox"]',
            '[role="searchbox"]',
            '[role="combobox"]',
            '[role="menuitem"]',
            
            // Специальные для поиска (Google, etc)
            '[name="q"]',              // Google search
            '[name="search"]',
            '[type="search"]',
            '[aria-label*="search" i]',
            '[aria-label*="поиск" i]',
            '[placeholder*="search" i]',
            '[placeholder*="поиск" i]',
            
            // Submit кнопки
            '[type="submit"]',
            'button[type="submit"]',
            
            // Clickable элементы
            '[onclick]',
            '[tabindex]',
            '[contenteditable="true"]',
            
            // Заголовки для контекста
            'h1', 'h2', 'h3'
        ];
        
        // Собираем элементы
        const foundElements = new Set();
        
        for (const selector of selectors) {
            try {
                const elems = document.querySelectorAll(selector);
                elems.forEach(elem => {
                    if (isElementVisible(elem) && isInteractive(elem)) {
                        foundElements.add(elem);
                    }
                });
            } catch (e) {
                // Игнорируем ошибки в селекторах
            }
        }
        
        // Обрабатываем найденные элементы
        foundElements.forEach(elem => {
            const rect = elem.getBoundingClientRect();
            const elementId = `elem_${elementCounter++}`;
            
            // Устанавливаем data-agent-id
            elem.setAttribute('data-agent-id', elementId);
            
            // Собираем информацию
            const elementInfo = {
                id: elementId,
                tag: elem.tagName.toLowerCase(),
                type: elem.type || null,
                text: getFullText(elem),
                placeholder: elem.placeholder || null,
                href: elem.href || null,
                ariaLabel: elem.getAttribute('aria-label'),
                value: elem.value || null,
                selector: generateSelector(elem),
                role: elem.getAttribute('role') || null,
                name: elem.name || null,
                position: {
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height)
                },
                isInViewport: (
                    rect.top >= -100 &&  // Немного за пределами тоже считаем
                    rect.top <= window.innerHeight + 100 &&
                    rect.left >= -100 &&
                    rect.left <= window.innerWidth + 100
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
        Извлекает элементы со страницы с улучшенной логикой.

        Args:
            page: Playwright Page

        Returns:
            Список Element объектов
        """
        # Ждём стабилизации DOM
        try:
            await page.wait_for_load_state('domcontentloaded', timeout=5000)
        except:
            pass  # Продолжаем даже если таймаут

        # Выполняем улучшенный скрипт
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
                is_in_viewport=raw.get('isInViewport', False),
                role=raw.get('role'),
                name=raw.get('name')
            )
            elements.append(element)

        return elements

    @staticmethod
    def prioritize_elements(elements: List[Element], limit: int = 100) -> List[Element]:
        """
        УЛУЧШЕННАЯ приоритизация элементов.

        Стратегия:
        1. Search inputs - наивысший приоритет
        2. Submit buttons
        3. Видимые интерактивные
        4. Заголовки для контекста
        5. Остальное
        """
        # Категоризация
        search_inputs = []
        submit_buttons = []
        interactive = []
        headers = []
        other = []

        for elem in elements:
            # Search inputs (критически важны!)
            if (elem.tag == 'input' and
                (elem.type == 'search' or
                 elem.name in ['q', 'search'] or
                 'search' in (elem.placeholder or '').lower() or
                 'search' in (elem.aria_label or '').lower() or
                 elem.role == 'searchbox')):
                search_inputs.append(elem)

            # Submit кнопки
            elif (elem.tag == 'button' and elem.type == 'submit') or \
                 (elem.tag == 'input' and elem.type == 'submit'):
                submit_buttons.append(elem)

            # Заголовки
            elif elem.tag in ['h1', 'h2', 'h3']:
                headers.append(elem)

            # Интерактивные
            elif elem.tag in ['button', 'a', 'input', 'textarea', 'select']:
                interactive.append(elem)

            else:
                other.append(elem)

        # Внутри каждой категории: видимые первыми
        def sort_by_visibility(elems):
            return sorted(elems, key=lambda e: (
                not e.is_in_viewport,
                e.position.get('y', 0)
            ))

        search_inputs = sort_by_visibility(search_inputs)
        submit_buttons = sort_by_visibility(submit_buttons)
        interactive = sort_by_visibility(interactive)
        headers = sort_by_visibility(headers)

        # Собираем в правильном порядке
        result = (
            search_inputs +
            submit_buttons +
            interactive[:30] +  # Топ 30 интерактивных
            headers[:5] +       # Топ 5 заголовков для контекста
            other[:10]          # Немного остального
        )

        return result[:limit]

    @staticmethod
    def format_for_llm(elements: List[Element]) -> str:
        """Улучшенное форматирование для LLM"""

        lines = ["=== INTERACTIVE ELEMENTS ===\n"]

        # Статистика
        by_tag = {}
        for elem in elements:
            by_tag[elem.tag] = by_tag.get(elem.tag, 0) + 1

        stats = ", ".join([f"{tag}:{count}" for tag, count in sorted(by_tag.items())])
        lines.append(f"Total: {len(elements)} ({stats})\n")

        # Специальные элементы (search, submit)
        search_elems = [e for e in elements if
                       'search' in (e.placeholder or '').lower() or
                       'search' in (e.aria_label or '').lower() or
                       e.role == 'searchbox' or
                       e.name in ['q', 'search']]

        if search_elems:
            lines.append("🔍 SEARCH ELEMENTS (IMPORTANT):")
            for elem in search_elems:
                lines.append(f"   {elem.id} | {elem.tag} | ph:\"{elem.placeholder}\" | ✓VISIBLE" if elem.is_in_viewport else f"   {elem.id} | {elem.tag}")
            lines.append("")

        # Элементы
        lines.append("ALL ELEMENTS:")
        for elem in elements:
            parts = [elem.id, elem.tag.upper()]

            if elem.type:
                parts.append(f"type={elem.type}")

            if elem.text:
                text = elem.text[:40].replace('\n', ' ')
                parts.append(f'"{text}"')

            if elem.placeholder:
                parts.append(f'ph:"{elem.placeholder[:25]}"')

            if elem.href:
                parts.append('link')

            if elem.aria_label:
                parts.append(f'aria:"{elem.aria_label[:20]}"')

            if elem.role:
                parts.append(f'role={elem.role}')

            if elem.is_in_viewport:
                parts.append("✓visible")

            lines.append(" | ".join(parts))

        return "\n".join(lines)