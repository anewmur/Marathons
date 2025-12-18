## Цель проекта
Построить статистическую модель для предсказания времени финиша по:
- трассе (идентификатор трассы хранится в колонке race_id)
- полу
- возрасту

Работа ведётся на шкале логарифма времени:
- Y = ln(T), где T — time_seconds

Возраст учитывается через сглаженную функцию (сплайн), общую для всех трасс внутри пола.

---

## Терминология и обозначения

- race_id: строковый идентификатор трассы в данных (колонка DataFrame)
- c: трасса как математический индекс, соответствующий значениям race_id
- year: год старта (колонка); в формулах обозначается j
- gender: пол (колонка); в формулах обозначается g
- age: возраст (колонка); в формулах обозначается a

В коде и таблицах используем race_id. В формулах используем c, но понимаем, что c ⇔ race_id.

---

## Модель (математически)

Для старта i на трассе c, пола g, года j и возраста a:

Y_i = ln(T_i)

Z_i = Y_i − ln R^{use}_{c,g}(j)

Z_i = h_g(a_i) + ε_i  
ε_i ~ N(0, σ_g^2)

Где:
- R^{use}_{c,g}(j) — трассово-годовой эталон на ln-шкале
- h_g(a) — возрастная поправка (сплайн), общая по всем трассам данного пола
- σ_g^2 — дисперсия остатка

Прогноз:
- Y_pred = ln R^{use}_{c,g}(j*) + h_g(a_new)
- T_pred = exp(Y_pred)
- интервалы и неопределённость строятся на Y-шкале, затем переводятся в секунды

---

## Поток данных

### 1. Загрузка данных (DataLoader)
- читает Excel
- приводит к единому контракту
- кэширует parquet

Выход:
- df_raw

---

### 2. Базовая фильтрация и валидация сырья
- filter_raw
- validate_raw
- add_row_id

Выход:
- df_raw_valid с индексом row_id

---

## 3. Предобработка (Preprocessor)

Цель: получить df_clean, где каждая строка — корректный старт, а идентификация людей детерминирована и воспроизводима.

### 3.1 Нормализация строк
- surname, name: trim, схлопывание пробелов, пустые → NA (недопустимы),
  транслитерация кириллицы в латиницу по PERSON_TOKEN_CYR_TO_LAT,
  затем upper
- Сейчас для формально trim=0
- city: нормализация (канонизация)

### 3.2 Базовая идентификация человека по имени
- person_ordered_key = SURNAME|NAME
- person_set_key = min(SURNAME|NAME, NAME|SURNAME)
  (инвариант к перестановке имени и фамилии)

### 3.3 Приближённый год рождения
- approx_birth_year = year − age

### 3.4 Стабилизация года рождения
Внутри каждого person_set_key:
- строятся частоты approx_birth_year
- годы разбиваются на кластеры с шагом ≤ 1 год (цепочкой)
- для каждого кластера выбирается birth_year_stable
- строкам примерживается:
  - cluster_id
  - birth_year_stable

### 3.5 Формирование runner_id (контракт)
runner_id — строковый идентификатор гипотезы "один физический человек".
На базовом шаге runner_id строится из person_set_key и стабилизированного года рождения.

Формат:
- birth_suffix = "_UNKNOWN", если отсутствуют birth_year_stable или birth_cluster_min_year
- иначе birth_suffix = f"_{birth_year_stable}_M{birth_cluster_min_year}"
- runner_id = f"{person_set_key}{birth_suffix}"

Этот формат является частью контракта и используется во всех последующих шагах.


### 3.6 Разрешение конфликтов города (политика проекта)

Город используется как жёсткий дискриминатор после стабилизации года рождения.
В базовый runner_id город не входит.

Шаги:
- если внутри runner_id город однозначен → заполняем пропуски этим значением
- если внутри runner_id город конфликтует (>=2 разных city) и city не NA → разукрупняем runner_id:
  runner_id = runner_id + "__CITY_" + city
- если внутри runner_id city == NA во всех строках → runner_id оставляем как есть
  (разделять по городу невозможно)

Интерпретация:
- разные города внутри одного runner_id считаются разными людьми,
  если это не противоречит уже выполненной стабилизации года рождения

Контракт после шага:
- внутри runner_id город либо однозначен, либо NA (если везде NA)
- разукрупнение по городу отражается в runner_id суффиксом "__CITY_{city}"

### 3.7 Исправление пола (контракт для модели)
Цель: обеспечить однозначность пола внутри runner_id.

Шаги:
- если внутри runner_id есть строгое большинство по gender → присваиваем этот gender всем строкам runner_id
- если строгого большинства нет → runner_id целиком удаляется

Контракт после шага:
- в df_clean внутри runner_id пол однозначен

### 3.8 Удаление неразрешимых конфликтов "один старт"
Ситуация:
- у одного runner_id более одной строки в одном и том же (race_id, year)

Политика:
- после шага разукрупнения runner_id по городу любые оставшиеся случаи,
  где в (runner_id, race_id, year) больше одной строки, удаляются целиком
  (редкие случаи, не стоящие отдельной логики)
- отдельный частный случай: если в (runner_id, race_id, year) > 1 строк,
  bib_number имеет > 1 уникального значения, а city однозначен, группа удаляется
  (это считается неразрешимым шумом)

### 3.9 Строгая дедупликация
Удаляются точные дубликаты по ключу:
- базовый ключ: (runner_id, race_id, year, time_seconds)
- если доля заполнения bib_number > 0.5, то bib_number добавляется в ключ:
  (runner_id, race_id, year, time_seconds, bib_number)

### 3.10 Добавление модельных признаков
- age_clamped = clip(age, age_min_global, age_max_global)
- Y = ln(time_seconds)
- x = (age_clamped − age_center) / age_scale

Выход:
- df_clean — финальный датасет для моделирования

---

## 4. Трассово-годовые эталоны R^{use}

### ReferenceBuilder

ReferenceBuilder (TraceReferenceBuilder) строит эталон на шкале времени T (секунды).
Логарифм эталона используется в формулах как ln R^{use} и вычисляется после построения эталона.

Протокол для каждой группы (race_id, gender) на входных данных build():
1) фильтруем time_seconds: не NA и > 0
2) сортируем time_seconds по возрастанию (быстрые впереди)
3) выбираем top_fraction лучших; если top_fraction даёт меньше min_used_runners наблюдений,
   то используем top_size = min(n_total, max(ceil(n_total * top_fraction), min_used_runners))
4) внутри top применяем симметричный trim_fraction
5) эталон reference_time = median(использованных time_seconds)
6) вычисляем reference_log = ln(reference_time)
7) дисперсию эталона оцениваем bootstrap по тому же протоколу

Индексация:
- в текущей реализации эталон строится по (race_id, gender) и агрегирует все годы,
  которые поданы на вход build()
- LOY vs production обеспечивается внешней фильтрацией по year до вызова build()

Выход:
- таблица эталонов reference_time и reference_log для (race_id, gender)
---

## 5. Возрастная модель (ключевая часть)

### 5.1 Архитектура

Модуль: `spline_model/`

Основные классы:
- `AgeSplineFitter` — обучение модели (age_spline_fit.py)
- `AgeSplineModel` — хранение параметров и предсказание (age_spline_model.py)

Вспомогательные модули:
- `build_centering_matrix.py` — построение матриц ограничений (A, C)
- `z_frame.py` — построение Z-фрейма

### 5.2 Математическая спецификация

Модель на шкале Z:
```
Z = h_g(a) + ε,  ε ~ N(0, σ²)
```

Сплайновая функция:
```
h_g(a) = Σ β_k · B_k(x),  где x = (age - age_center) / age_scale
```

Ограничения центрирования (в точке x₀ = 0, т.е. age = age_center):
- h(0) = 0  (значение)
- h'(0) = 0 (производная)

Параметризация:
- B_raw: исходный B-сплайновый базис размера (n, K_raw)
- A: матрица ограничений размера (2, K_raw)
- C: базис нуль-пространства A, размера (K_raw, K_cent), где K_cent = K_raw - 2
- β = C · γ — редуцированная параметризация, автоматически удовлетворяющая A·β = 0

Штраф P-spline:
```
penalty = λ · ||D·β||² = λ · β' · (D'D) · β
```
где D — матрица вторых разностей.

### 5.3 Поток данных возрастной модели

```
train_frame_loy (status=OK, year≠validation_year)
    ↓
build_z_frame() → z_frame с колонками (gender, age, Z)
    ↓
AgeSplineFitter.fit(z_frame) → dict[gender, AgeSplineModel]
    ↓
Для каждого пола:
    1. _compute_x_std_and_clamped_age() → x = (age - 35) / 10
    2. build_knots_x() → узлы по квантилям в шкале x
    3. build_raw_basis() → B_raw (n, K_raw)
    4. build_centering_matrix() → (A, C)
    5. build_second_difference_matrix() → D
    6. solve_penalized_lsq() → β = C·γ
    7. _assemble_model() → AgeSplineModel
```

### 5.4 Ключевые функции (реализованы ✅)

| Функция | Файл | Назначение |
|---------|------|------------|
| `build_knots_x` | age_spline_fit.py | Квантильные узлы с детерминированным слиянием |
| `build_raw_basis` | age_spline_fit.py | B-сплайновый базис через BSpline.design_matrix |
| `build_second_difference_matrix` | age_spline_fit.py | Матрица D для P-spline штрафа |
| `build_centering_matrix` | build_centering_matrix.py | Матрицы (A, C) через SVD |
| `solve_penalized_lsq` | age_spline_fit.py | Решение в редуцированной параметризации |
| `build_z_frame` | z_frame.py | Построение Z = Y - ln(R^use) |
| `AgeSplineFitter.fit_gender` | age_spline_fit.py | Обучение для одного пола |
| `AgeSplineFitter.fit` | age_spline_fit.py | Обучение для всех полов |
| `AgeSplineModel.predict_h` | age_spline_model.py | Предсказание h(age) |
| `AgeSplineModel.predict_mean` | age_spline_model.py | Предсказание μ + γx + h(x) |
| `AgeSplineModel.design_row` | age_spline_model.py | Строка дизайн-матрицы для диагностики |

### 5.5 Конфигурация (config.yaml)

```yaml
preprocessing:
  age_center: 35.0      # центр нормировки возраста
  age_scale: 10.0       # масштаб нормировки

age_spline_model:
  age_min_global: 18.0  # глобальный минимум возраста (clamp)
  age_max_global: 80.0  # глобальный максимум возраста (clamp)
  degree: 3             # степень сплайна (кубический)
  max_inner_knots: 6    # максимум внутренних узлов
  min_knot_gap: 0.2     # минимальный шаг между узлами в шкале x
  lambda_value: 1.0     # параметр гладкости (фиксированный)
  centering_tol: 1e-10  # допуск на проверку ограничений
```

### 5.6 Текущие метрики (реальные данные)

- Обучающая выборка: n_M = 29450, n_F = 6755
- Базис: K_raw = 12 (M), K_raw = 13 (F)
- Центрирование: h(35) ≈ 4e-17 (машинная точность)
- Partition of unity: max_err ≈ 2e-16
- Время обучения: ~0.08 сек на оба пола

---

## 6. Прогноз

Вход:
- race_id (⇔ c)
- gender (g)
- age (a)
- year (j*)

Шаги:
1) берём ln R^{use}_{c,g} для пары (race_id, gender) из таблицы эталонов
2) вычисляем h_g(a) через AgeSplineModel.predict_h(age)
3) Y_pred = ln R^{use} + h_g(a)
4) переводим в секунды: T_pred = exp(Y_pred)
5) интервалы: считаем на Y-шкале с учётом σ_g^2, затем exp(·)

---

## 7. Валидация

На validation_year:
- ошибки на Y-шкале и на шкале секунд
- анализ по трассам и полу
- проверка покрытия интервалов (если интервалы реализованы)

---

## Что уже реализовано ✅

**Предобработка:**
- полный пайплайн предобработки до df_clean
- person_set_key, approx_birth_year
- кластеризация годов рождения внутри person_set_key и birth_year_stable
- runner_id в фиксированном формате
- нормализация/заполнение города и разбиение runner_id по конфликтному city
- исправление пола по большинству
- удаление неразрешимых мульти-стартов
- построение трассовых эталонов

**Возрастная модель:**
- Полный пайплайн B-сплайна с P-spline штрафом
- Центрирование h(0)=0, h'(0)=0 через редуцированную параметризацию
- AgeSplineFitter.fit() для обоих полов
- AgeSplineModel.predict_h() для предсказания
- Тесты на синтетике и реальных данных

---

## Что осталось реализовать

1) **select_lambda_gcv** — автоматический выбор λ по GCV
2) **MarathonModel.predict_log_time()** — полный прогноз ln R^use + h(age)
3) **Линейная часть модели** — добавить μ и γ (coef_mu, coef_gamma)
4) **compute_tau2_bar** — средняя дисперсия эталона
5) **apply_winsor** — винзоризация выбросов
6) **determine_degradation** — деградация при малых данных
7) **Прогноз с неопределённостью** — интервалы через Монте-Карло
8) **save/load_age_spline_models** — сериализация моделей
9) **Блок валидации** — метрики, отчёты

---

## Принципиальные позиции проекта

- идентификация людей: только детерминированные правила, без "умных" кластеризаций
- возраст: только гладкая функция (сплайн), а не дискретные "возрастные эталоны"
- допускается удаление строк и целых runner_id при неразрешимых конфликтах
  (это сознательная политика качества данных, а не "ошибка пайплайна")
- единая шкала x_std = (age - age_center) / age_scale во всей модели
- центрирование в точке x=0 (age=35) для интерпретируемости параметров


### 13.7 Текущие тесты

| Файл | Тесты | Что проверяют |
|------|-------|---------------|
| test_spline.py | test_build_raw_basis_partition_of_unity | Σ B_k(x) = 1 |
| test_spline.py | test_solve_penalized_lsq_recovers_gamma_when_lambda_zero | λ=0 → OLS |
| test_centering.py | test_centering_matrix_shapes_and_null_properties | A·C=0, rank(C)=K-2 |
| test_centering.py | test_centering_constraints_hold_for_random_gamma | β=C·γ → A·β=0 |
| test_centering.py | test_solve_penalized_lsq_preserves_constraints | Решение удовлетворяет A·β≈0 |
| test_fitter.py | test_fit_gender_produces_model_and_beta_is_finite | fit_gender работает |
| test_predict.py | test_predict_h_at_age_center_is_near_zero | h(35) ≈ 0 |
| test_predict.py | test_predict_h_returns_finite_across_age_range | Конечность на [18,80] |
| test_predict.py | test_predict_h_scalar_vs_array_consistent | Скаляр = массив |
| test_predict.py | test_predict_h_clamps_age_outside_bounds | h(10)=h(18), h(100)=h(80) |
| test_predict.py | test_predict_mean_equals_predict_h_when_mu_gamma_zero | m=h при μ=γ=0 |
| test_predict.py | test_design_row_returns_correct_structure | Структура словаря |
| test_real_data.py | test_real_data_prepare_z_frame | x_std, partition of unity |
| test_real_data.py | test_real_data_train_frame_has_Z | Колонка Z существует |
| test_real_data.py | test_real_data_solve_penalized_lsq_runs_and_preserves_centering | A·β≈0 на реальных |
| test_real_data.py | test_real_data_fit_gender_runs_and_preserves_centering | fit_gender на реальных |
| test_real_data.py | test_real_data_predict_h_at_age_center_is_near_zero | h(35)≈0 на реальных |
| test_real_data.py | test_real_data_predict_h_is_finite_across_age_range | Конечность |
| test_real_data.py | test_real_data_predict_h_clamps_correctly | Clamp на реальных |
| test_real_data.py | test_real_data_predict_h_monotonic_after_peak | Sanity check формы |
| test_real_data.py | test_real_data_fit_both_genders | fit() для M и F |
| test_real_data.py | test_real_data_fit_report_contains_required_fields | Поля fit_report |
