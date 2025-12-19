# Правила для Python-MVP

Эти правила предназначены для поддержания контроля над кодом в проектах на этапе MVP: минимум инфраструктуры, максимум наблюдаемости и отладки.

## 0. Карта проекта как отправная точка

Перед любым изменением функционала обновляйте и сверяйте с картой проекта/класса: простой markdown-файл (`PROJECT_MAP.md`) с основными сущностями, потоками данных и границами ответственности. Карта может появляться постепенно — начинайте с заголовков и заполняйте по мере роста. Новое поведение сначала вписывается в карту.

**Пример `PROJECT_MAP.md`**

```markdown
# Карта проекта MarathonModel

## Основные сущности
- MarathonModel: оркестратор пайплайна
- df: сырые данные после загрузки
- df_clean: предобработанные данные с row_id (индекс)
- trace_references: эталоны трасс по (race_id, gender)
- age_references: dict[str, pd.DataFrame] — возрастные таблицы по race_id
- config: параметры из config.yaml

## Поток данных
1. load_data() → df
2. validate_raw() → проверка колонок и null
3. add_row_id() → df с индексом row_id
4. preprocess() → df_clean
5. build_trace_references() → trace_references (на LOY-выборке)
6. build_age_references() → age_references (на LOY-выборке)
7. fit_age_model() → (заглушка)
8. predict() → прогноз (заглушка)

## Границы ответственности
- MarathonModel: оркестрация, контроль шагов (_check_step), логирование (_log_step), debug_slice
- DataLoader: загрузка и кэширование файлов/директорий
- Preprocessor: очистка и нормализация данных
- TraceReferenceBuilder: построение эталонов трасс
- AgeReferenceBuilder: построение возрастных таблиц
- Внешние шаги (preprocess, builders): чистая логика, возврат DataFrame, без I/O
```

## 1. Обязательные аннотации и краткие docstring

Каждая функция/метод имеет аннотации типов и короткий docstring: что принимает, что возвращает, что делает. Инварианты указывать только если они неочевидны или критичны.

```python
def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Принимает: DataFrame с сырыми колонками name/surname.
    Возвращает: DataFrame с очищенными именами.
    Делает: приводит к верхнему регистру, удаляет лишние пробелы.
    Инварианты: row_id остаётся уникальным, нет null в name/surname.
    """
    ...
    return df
```

## 2. Наблюдаемость через единый шаблон логов

Каждый шаг логирует минимум: `rows_in`, `rows_out`, ключевую метрику (groups, clusters, dropped) и `elapsed_sec`.

```python
import logging, time

def stage_assign_gender(df: pd.DataFrame) -> pd.DataFrame:
    start = time.time()
    rows_in = len(df)
    # ... логика
    rows_out = len(df)
    dropped = rows_in - rows_out
    elapsed = time.time() - start
    logging.info(f"assign_gender: rows {rows_in} → {rows_out}, dropped {dropped}, elapsed {elapsed:.2f}s")
    return df
```

## 3. Проверки входа в одном месте

На входе пайплайна/компонента — одна функция валидации, возвращающая `df` (или `raise ValueError`). Дальше только локальные проверки результата текущего шага.

```python
def stage_validate(df: pd.DataFrame) -> pd.DataFrame:
    required = ['name', 'surname', 'time']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")
        if df[col].isna().any():
            raise ValueError(f"Nulls in {col}")
    if len(df) == 0:
        raise ValueError("Empty dataframe")
    return df
```

## 4. Конвейер как цепочка шагов с одним смыслом

Пайплайн — последовательность стадий, каждая делает один смысловой блок. Вложенные функции запрещены. Смешивание оркестровки и бизнес-логики запрещено.

```python
df = (load_raw()
      .pipe(stage_validate)
      .pipe(add_row_id)
      .pipe(normalize_names)
      .pipe(assign_gender)
      .pipe(group_runners)
      .pipe(assign_runner_id))
```

## 5. Технический row_id с самого начала

Сразу после загрузки/валидации добавляем `row_id` как основной ключ. Бизнес-ID можно читать, но переписывать только в финальном шаге.

```python
def add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df['row_id'] = range(len(df))
    return df.set_index('row_id')

def assign_runner_id(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df = df.reset_index()
    df['runner_id'] = df['row_id'].map(mapping)
    return df
```

## 6. Чистая логика отдельно от I/O

Запись файлов/отчётов/артефактов (включая промежуточные JSON кластеров) — только в конце пайплайна, в оркестраторе или под флагом `--save`. Шаги обработки возвращают данные (например, `cluster_records`), но не пишут их сами. Основные функции по умолчанию ничего не пишут на диск.

## 7. Единый debug-интерфейс для срезов

Одна функция `debug_slice` с гибкими фильтрами.

```python
import pandas as pd
import numpy as np

def debug_slice(df: pd.DataFrame, filters: dict | None = None) -> None:
    if not filters:
        return
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if callable(val):
            mask &= val(df[col])
        elif isinstance(val, (list, tuple, set)):
            mask &= df[col].isin(val)
        elif isinstance(val, np.ndarray):
            mask &= df[col].isin(val)
        else:
            mask &= (df[col] == val)
    print(df[mask].to_string(index=True))
```

Пример: `debug_slice(df, {"surname": "Иванов", "gender": lambda s: s.isna()})`

## 8. Запрет на монолитные функции

Функция/метод длиннее 30 строк может быть только «дирижёром». Максимум 20–30 строк для функций с логикой. Исключение: линейная функция без ветвлений и циклов. Сложная логика разбивается на шаги для отдельного логирования и дебага.

## 9. Правки только через зафиксированный кейс

Перед изменением фиксируете 1–2 примера и проверяете их через `debug_slice` до/после (или отдельный debug-режим).

## 10. Стандартная статистика и минимальные инварианты после шага

Каждый шаг собирает и логирует одинаковый набор статистики. После шага — 2–3 инварианта с `raise ValueError`, но только для колонок и свойств, гарантированных контрактом текущего шага. Проверка `rows_out > 0` только для нефильтрующих шагов.

```python
stats = {"rows_in": rows_in, "rows_out": rows_out, "unique_groups": df['group'].nunique()}
logging.info(f"stage stats: {stats}")

if not df.index.is_unique:
    raise ValueError("Duplicate row_id")
# time_seconds проверяется только если колонка уже создана
if 'time_seconds' in df.columns and (df['time_seconds'] < 0).any():
    raise ValueError("Negative time")
```

## 11. Когда создавать отдельный класс или модуль

Новый класс/модуль только при хотя бы одном условии: отдельная ответственность, независимый жизненный цикл, 3–4 подстадии внутри метода, переиспользование или >10–12 методов в текущем классе. Не дробить преждевременно — функциональный стиль предпочтителен для прозрачных стадий.

## 12. Константы и пороги: что в YAML, что в коде

В YAML — всё, что крутите без правки кода (пороги, пути, параметры). В коде — только структурные константы с комментариями.

```yaml
# config.yaml
thresholds:
  max_age: 100
  min_time_seconds: 0
paths:
  input: "data/raw.parquet"
  output: "results/final.parquet"
```

```python
REQUIRED_COLUMNS = ['name', 'surname', 'time']  # структура данных — ок в коде
```

---

## 13. Тестирование

### 13.1 Структура тестов

Тесты располагаются в `spline_model/spline_tests/`:

```
spline_model/spline_tests/
├── __init__.py
├── tests_smoke.py      # точка входа, запускает все тесты
├── test_spline.py      # тесты базиса и решателя
├── test_centering.py   # тесты центрирования
├── test_fitter.py      # тесты AgeSplineFitter
├── test_predict.py     # тесты predict_h, predict_mean
└── test_real_data.py   # интеграционные тесты на реальных данных
```

### 13.2 Типы тестов

**Unit-тесты (на синтетических данных):**
- Быстрые, детерминированные
- Проверяют математические свойства (partition of unity, ограничения)
- Не требуют реальных данных
- Файлы: test_spline.py, test_centering.py, test_fitter.py, test_predict.py

**Интеграционные тесты (на реальных данных):**
- Требуют полный пайплайн MarathonModel
- Проверяют сквозную работу на реальных данных
- Файл: test_real_data.py

### 13.3 Соглашения по написанию тестов

**Имена функций:**
```python
def test_<что_тестируем>_<ожидаемое_поведение>() -> None:
    """Краткое описание что проверяется."""
    ...
```

**Структура теста:**
```python
def test_predict_h_at_age_center_is_near_zero() -> None:
    """
    Проверяет контракт центрирования: h(age_center) ≈ 0.
    """
    # 1. Подготовка данных
    config = {...}
    fitter = AgeSplineFitter(config=config)
    df = pd.DataFrame({...})
    
    # 2. Выполнение
    model = fitter.fit_gender(gender_df=df, gender="M", trace_references=None)
    h_at_center = model.predict_h(35.0)
    
    # 3. Проверка с raise RuntimeError (не assert!)
    if abs(h_at_center) > 1e-8:
        raise RuntimeError(f"h(35) should be ~0, got {h_at_center}")
```

**Использовать `raise RuntimeError`, а не `assert`:**
- assert может быть отключен флагом -O
- RuntimeError всегда работает и даёт понятное сообщение

### 13.4 Тесты на реальных данных

Тесты в test_real_data.py:
- Принимают опциональный параметр `model=None`
- Если model передан — используют его
- Если нет — строят через `_real_data_build_model()`

```python
def test_real_data_predict_h_at_age_center_is_near_zero(model=None) -> None:
    if model is None:
        model = _real_data_build_model()
    ...
```

**Точка входа `test_real_data()`:**
```python
def test_real_data() -> None:
    model = _real_data_build_model()  # один раз
    
    tests = [
        ("test_name_1", test_func_1),
        ("test_name_2", test_func_2),
    ]
    
    for test_name, test_fn in tests:
        print(f"== Running: {test_name}")
        test_fn(model=model)  # переиспользуем модель
        print(f"PASSED: {test_name}")
```

### 13.5 Запуск тестов

**Все smoke-тесты:**
```bash
python spline_model/spline_tests/tests_smoke.py
```

**Только unit-тесты (быстро):**
```bash
python -c "from spline_model.spline_tests.test_spline import *; test_spline()"
```

**Только интеграционные:**
```bash
python -c "from spline_model.spline_tests.test_real_data import test_real_data; test_real_data()"
```

### 13.6 Что тестировать

**Обязательно:**
- Математические инварианты (partition of unity, A·β=0)
- Контракты центрирования (h(0)=0, h'(0)=0)
- Конечность и корректность выходов (np.isfinite)
- Формы массивов и типы возвращаемых значений

**Sanity checks (с предупреждениями, не падают):**
- Разумность формы кривой (h(30) < h(50) < h(70))
- Диапазоны значений

---
