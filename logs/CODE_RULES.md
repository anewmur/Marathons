# Правила кода для проекта MarathonModel

## 1. Процесс разработки

### Согласование до написания кода
Прежде чем писать класс или модуль:
1. Описать функционал словами
2. Определить входные данные
3. Определить выходные данные
4. Нарисовать карту потока данных
5. Получить явное ОК от заказчика

Код пишется только после согласования. Никаких сюрпризов!

### Инициатива
Разработчик действует по согласованию. Только явное указание "пиши как знаешь" даёт инициативу — и только на одно конкретное решение, не на весь проект.

### Динамические документы
CODE_RULES.md и REPORT.md — живые документы. Обновляются по ходу проекта. Изменения согласовываются.

### Структура проекта
Структура файлов динамическая — можно менять при согласовании если станет понятно зачем.

## 2. Структура файлов

### Один файл = один класс
- Каждый класс в отдельном файле
- Имя файла = имя класса в snake_case
- Пример: класс `DataLoader` → файл `data_loader.py`
- Исключение: утилиты в `utils.py`

### Текущая структура
```
marathon_model/
├── CODE_RULES.md          # Правила кода
├── REPORT.md              # Отчёт о проекте
├── config.yaml            # Параметры модели
├── main.py                # Класс MarathonModel (управляющий пайплайн)
├── data_loader.py         # [ГОТОВ] Класс DataLoader (загрузка с кэшированием)
├── preprocessor.py        # [ГОТОВ] Класс Preprocessor
├── reference_builder.py   # [ГОТОВ] TraceReferenceBuilder
├── age_reference_builder.py  # [ГОТОВ] AgeReferenceBuilder
├── age_model.py           # [TODO] Возрастная модель
├── predictor.py           # [TODO] Прогнозирование
├── utils.py               # Вспомогательные функции
├── dictionaries.py        # Словари нормализации
├── logging_setup.py       # Настройка логирования
└── visualizer.py          # [TODO] Визуализация (понадобится)
```

## 3. Конфигурация

### config.yaml
Числовые параметры модели хранятся в config.yaml:
```yaml
preprocessing:
  age_center: 35
  age_scale: 10

references:
  top_fraction: 0.25
  trim_fraction: 0.10
  min_group_size: 40
  min_used_runners: 8
  bootstrap_samples: 1000
  random_seed: 123

age_references:
  min_group_size: 30
  bootstrap_samples: 1000
  random_seed: 42

validation:
  year: 2025
```

### KNOWN_RACES
Словарь известных трасс находится в `dictionaries.py`.
При нераспознанной трассе — явная ошибка с предложением добавить трассу.

## 4. Типизация

### Встроенные типы Python 3.10+
Не импортировать из typing. Использовать:
- `dict` вместо `Dict`
- `list` вместо `List`
- `int | None` вместо `Optional[int]`

```python
# Правильно:
def process(data: dict[str, object]) -> list[str] | None:
    pass

# Неправильно:
from typing import Dict, Optional
def process(data: Dict[str, Any]) -> Optional[List[str]]:
    pass
```

## 5. Документация

### Docstrings
Только описание функционала. Без примеров использования.

```python
def calculate_reference(times: list[float]) -> float:
    """
    Рассчитать эталонное время.
    
    Args:
        times: Список времён финиша в секундах
        
    Returns:
        Эталонное время в секундах
    """
    pass
```

### Примеры использования
После кода, после разделителя `# ------`:

```python
# Весь код класса выше...

# ------
# Пример использования:
#
# loader = DataLoader()
# data = loader.load_file('results.xlsx')
# print(data.head())
```

## 6. Заглушки

### Заглушки без ошибок
Вместо `raise NotImplementedError` использовать print и возврат пустого результата:

```python
# Правильно:
def build_references(self):
    print("TODO: build_references — построение эталонов")
    return None

# Неправильно:
def build_references(self):
    raise NotImplementedError("Не реализовано")
```

Код не должен падать. Должно быть понятно что сработало, а что нет.

## 7. Обработка ошибок

### Явные ошибки при проблемах с данными
```python
if race_name not in self.KNOWN_RACES:
    raise ValueError(f"Неизвестная трасса: '{race_name}'. Добавьте в KNOWN_RACES.")
```

### Валидация в начале метода
```python
def fit(self, data: pd.DataFrame) -> None:
    if data.empty:
        raise ValueError("DataFrame пуст")
    
    required = ['age', 'time_seconds']
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    
    # Основная логика после всех проверок
```

## 8. Стиль кода

### Понятность важнее скорости
Код должен быть понятным. Простые конструкции предпочтительнее "умных" однострочников.

### Константы
Константы класса ЗАГЛАВНЫМИ_БУКВАМИ:
```python
class DataLoader:
    MIN_AGE = 10
    MAX_AGE = 100
```

### Комментарии
На русском, код на английском:
```python
# Извлекаем год из строки с датой
year = self._extract_year(date_str)
```

## 9. Кэширование данных

### DataLoader: механизм кэширования
DataLoader поддерживает автоматическое кэширование загруженных данных:

**Принцип работы:**
1. При вызове `load_directory_cached()` создаётся манифест файлов с метаданными (имя, размер, mtime)
2. Данные сохраняются в бинарном кэше (`data_cache.pkl`)
3. При следующем запуске сравниваются манифесты
4. Если файлы не изменились — данные загружаются из кэша (в разы быстрее)
5. Если файлы изменились — кэш перестраивается

**Файлы кэша:**
- `list_excel.yaml` (или `.json`) — манифест с метаданными Excel-файлов
- `data_cache.pkl` — бинарный кэш DataFrame в формате pickle

**Формат кэша:**
Используется `pandas.to_pickle()` / `read_pickle()`:
- ✅ Не требует дополнительных библиотек (встроено в pandas)
- ✅ Сохраняет все типы данных pandas
- ⚠️ Только для Python (не универсальный формат)

**Альтернатива:** Можно использовать parquet для более компактного хранения, но требуется установка `pyarrow`:
```bash
pip install pyarrow
```

## 10. Контрольный чеклист

Перед коммитом:
- [ ] Функционал согласован
- [ ] Класс в отдельном файле
- [ ] Нет импортов из `typing`
- [ ] Type hints для публичных методов
- [ ] Примеры после `# ------`
- [ ] Заглушки через print (не raise)
- [ ] Валидация в начале методов
- [ ] CODE_RULES.md актуален
- [ ] REPORT.md актуален
