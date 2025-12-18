# **СПЕЦИФИКАЦИЯ: Построение возрастных сплайнов h_g(a)**

## **1. Цель и назначение этапа**

### **1.1. Основная цель**
Реализовать оценку возрастной средней функции для каждого пола g:
```
m_g(a) = μ_g + γ_g * x(a) + h_g(a)
```
где:
- `μ_g` — интерсепт (константа по полу)
- `γ_g * x(a)` — линейный тренд по нормированному возрасту
- `h_g(a)` — центрированная сплайновая поправка (кубический B-сплайн)

### **1.2. Обязательное центрирование сплайна**
Сплайновая часть h_g(a) обязана быть центрирована так, чтобы **не содержать ни константы, ни линейного тренда**:
- `h_g(x0) = 0` (значение в опорной точке)
- `h'_g(x0) = 0` (производная в опорной точке)

где `x0 = 0` — нормированное значение age_center.

**Это достигается через матрицу C как базис нулевого пространства ограничений A.**

### **1.3. Экстраполяция**
Вне обучающего диапазона возраста **не экстраполировать и не запрещать**. 
- При обучении: применять clamp возраста к [age_min_global, age_max_global]
- При predict: clamp НЕ применяется автоматически (делается снаружи если нужно)

### **1.4. LOY и мир данных**
**LOY разделение выполняется уровнем выше.** 

Фиттер:
- **В `__init__`**: читает параметры из `config["age_spline_model"]` и `config["preprocessing"]`, фиксирует их
- **В `fit`**: НЕ обращается к config, использует только зафиксированные параметры
- Получает готовый df своего мира (с колонками gender, age, z) без логики "какой сейчас год"

---

## **2. Шкала возраста и нормировка**

### **2.1. ЕДИНАЯ шкала x для всего проекта**

**В проекте используется ОДНА нормировка**:
```python
x = (age - age_center) / age_scale
```

где из config.yaml:
```yaml
preprocessing:
  age_center: 35
  age_scale: 10
```

**Эта же x используется**:
1. В Preprocessor.add_features (уже есть в df_clean как колонка "x")
2. В возрастном сплайне (вычисляется заново внутри fit_gender и predict_mean)

**Примеры**:
- age = 18 → x = (18-35)/10 = -1.7
- age = 35 → x = 0.0 (опорная точка)
- age = 45 → x = 1.0
- age = 80 → x = 4.5

### **2.2. Опорная точка центрирования**

```python
x0 = (age_center - age_center) / age_scale = 0.0
```

**ВСЕГДА x0 = 0.0** при использовании проектной нормировки.

Ограничения центрирования:
- `h(0) = 0`
- `h'(0) = 0`

### **2.3. Функция нормировки**

```python
def normalize_age(age: float, age_center: float, age_scale: float) -> float:
    """
    Нормировка возраста по ЕДИНОЙ проектной формуле.
    x = (age - age_center) / age_scale
    
    Входы:
    - age: возраст в годах (может быть вещественным)
    - age_center: центр нормировки (из config.preprocessing.age_center)
    - age_scale: масштаб нормировки (из config.preprocessing.age_scale)
    
    Выход:
    - x: нормированный возраст (безразмерный)
    
    Примечание: clamp здесь НЕ применяется — это делается отдельно при необходимости.
    """
    return (age - age_center) / age_scale
```

### **2.4. Global clamp при обучении**

При обучении (в fit_gender) возраст обрезается к глобальным границам:

```python
# 1. Global clamp возраста
age_min_global = self.age_min_global  # из config, например 18
age_max_global = self.age_max_global  # из config, например 80

df_gender["age"] = df_gender["age"].clip(age_min_global, age_max_global)

# 2. Нормировка (та же, что в Preprocessor)
df_gender["x"] = (df_gender["age"] - self.age_center) / self.age_scale

# 3. Фактические границы для статистики
a_min_g = df_gender["age"].min()  # фактический минимум для пола g
a_max_g = df_gender["age"].max()  # фактический максимум для пола g
age_range_years = a_max_g - a_min_g
```

### **2.5. Что сохраняет AgeSplineModel**

```python
@dataclass
class AgeSplineModel:
    gender: str
    
    # Параметры нормировки (из preprocessing в config)
    age_center: float  # 35 (единственный источник: config.preprocessing.age_center)
    age_scale: float   # 10 (из config.preprocessing.age_scale)
    
    # Опорная точка для центрирования (всегда 0.0)
    x0: float = 0.0
    
    # Глобальные границы возраста (для справки, использовались при обучении)
    age_min_global: float  # 18
    age_max_global: float  # 80
    
    # Фактический диапазон данных пола (для справки и логов)
    age_range_actual: tuple[float, float]  # (a_min_g, a_max_g)
    
    # B-сплайн параметры
    degree: int  # 3 (кубический)
    knots_x: list[float]  # узлы в шкале x (НЕ в шкале age!)
    
    # Центрирование базиса
    basis_centering: dict[str, Any]  # содержит A, C, centering_method, K_raw, K_cent
    
    # Оцененные коэффициенты
    coef_mu: float        # μ_g (интерсепт)
    coef_gamma: float     # γ_g (линейный тренд)
    coef_beta: pd.Series  # β_g (коэффициенты центрированного сплайна)
                          # индекс = имена базисных функций
    
    # Параметр гладкости и дисперсии
    lambda_value: float   # выбранный λ
    sigma2_reml: float    # остаточная дисперсия по REML (или GCV)
    tau2_bar: float       # средняя дисперсия эталона для этого пола
    sigma2_use: float     # = max(floor, sigma2_reml - tau2_bar)
    nu: float             # степени свободы для t-распределения
    
    # Winsorization
    winsor_params: dict[str, float]  # медиана, MAD, порог, доля зажатых
    
    # Отчет о подгонке
    fit_report: dict[str, Any]  # n, age_range_years, edf, degrade, warnings, timings
```

---

## **3. Архитектура классов и контракты**

### **3.1. AgeSplineModel — модель для одного пола**

См. раздел 2.5 выше.

**Методы**:

```python
def predict_mean(self, age: float) -> float:
    """
    Вычисляет m_g(a) = μ + γ*x + h(x).
    
    Входы:
    - age: возраст в годах (может быть вне обучающего диапазона)
    
    Выход:
    - m: предсказанное значение на нормированной шкале Z
    
    ВАЖНО: clamp НЕ применяется автоматически. 
    Если нужен clamp, он делается снаружи перед вызовом.
    """
    # Нормировка по ЕДИНОЙ проектной формуле
    x = (age - self.age_center) / self.age_scale
    
    # Базис
    B_raw_x = build_raw_basis(np.array([x]), self.knots_x, self.degree)
    B_cent_x = B_raw_x @ self.basis_centering["C"]
    
    # Сплайновая часть
    h_x = (B_cent_x @ self.coef_beta.values)[0]
    
    # Полная модель
    m = self.coef_mu + self.coef_gamma * x + h_x
    
    return m


def design_row(self, age: float) -> dict[str, float]:
    """
    Возвращает столбцы центрированного дизайна ( матрицы объясняющих переменных).
    
    Выход:
    {
        "intercept": 1.0,
        "x": float,              # нормированный возраст
        "spline_0": float,       # значения центрированных базисных функций
        "spline_1": float,
        ...
    }
    
    Ключи spline_* соответствуют индексу coef_beta.
    Используется для диагностики и тестирования.
    """
    x = (age - self.age_center) / self.age_scale
    
    B_raw_x = build_raw_basis(np.array([x]), self.knots_x, self.degree)
    B_cent_x = B_raw_x @ self.basis_centering["C"]
    
    result = {
        "intercept": 1.0,
        "x": x
    }
    
    for j, coef_name in enumerate(self.coef_beta.index):
        result[coef_name] = B_cent_x[0, j]
    
    return result
```

### **3.2. AgeSplineFitter — построение моделей**

```python
class AgeSplineFitter:
    """
    Построение возрастных сплайнов для всех полов из одного мира данных.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Инициализация параметров из config.
        ВСЕ параметры считываются ЗДЕСЬ и фиксируются.
        В методе fit обращения к config НЕТ.
        
        Читает:
        - config["age_spline_model"] — параметры сплайна
        - config["preprocessing"] — age_center, age_scale
        """
        spline_config = config.get("age_spline_model", {})
        preproc_config = config.get("preprocessing", {})
        
        # Нормировка возраста (из preprocessing)
        self.age_center = float(preproc_config.get("age_center", 35))
        self.age_scale = float(preproc_config.get("age_scale", 10))
        
        # Глобальные границы для clamp при обучении
        self.age_min_global = float(spline_config.get("age_min_global", 18))
        self.age_max_global = float(spline_config.get("age_max_global", 80))
        
        # B-сплайн параметры
        self.degree = int(spline_config.get("degree", 3))
        self.max_inner_knots = int(spline_config.get("max_inner_knots", 10))
        self.min_knot_gap = float(spline_config.get("min_knot_gap", 0.2))
        
        # λ выбор
        self.lambda_grid_log_min = float(spline_config.get("lambda_grid_log_min", -6))
        self.lambda_grid_log_max = float(spline_config.get("lambda_grid_log_max", 2))
        self.lambda_grid_n = int(spline_config.get("lambda_grid_n", 50))
        self.lambda_method = str(spline_config.get("lambda_method", "GCV"))
        
        # Пороги деградации
        self.n_min_spline = int(spline_config.get("n_min_spline", 300))
        self.range_min_years = float(spline_config.get("range_min_years", 15))
        self.n_min_linear = int(spline_config.get("n_min_linear", 80))
        self.range_min_linear = float(spline_config.get("range_min_linear", 8))
        
        # Дисперсии
        self.sigma2_floor = float(spline_config.get("sigma2_floor", 1e-8))
        self.nu_floor = float(spline_config.get("nu_floor", 3.0))
        
        # Winsor
        self.winsor_enabled = bool(spline_config.get("winsor_enabled", True))
        self.winsor_k = float(spline_config.get("winsor_k", 3.0))
        
        # Диагностика
        self.centering_tol = float(spline_config.get("centering_tol", 1e-10))
        self.verbose_fit = bool(spline_config.get("verbose_fit", True))
    
    
    def fit(
        self,
        df: pd.DataFrame
    ) -> dict[str, AgeSplineModel]:
        """
        Построение моделей для всех полов в df.
        
        Входы:
        - df: обучающий датафрейм с колонками:
            * gender: str ("M" или "F")
            * age: float|int (возраст в годах)
            * z: float (нормированная величина Z* = Y - ln(R_use), УЖЕ вычислена)
        
        Выход:
        - dict[gender, AgeSplineModel]
        
        Шаги:
        1. Для каждого уникального gender:
           a. Фильтрация: gender_df = df[df["gender"] == gender]
           b. fit_gender(gender_df, gender) → AgeSplineModel
        2. Логирование агрегированных метрик
        3. Возврат dict моделей
        
        НЕ читает config (все параметры уже в self из __init__).
        """
        start_time = time.time()
        
        # Проверка обязательных колонок
        required_columns = ["gender", "age", "z"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"AgeSplineFitter.fit: missing columns: {missing_columns}")
        
        # Цикл по полам
        models = {}
        for gender in df["gender"].unique():
            gender_df = df[df["gender"] == gender].copy()
            
            logger.info("fit_gender: gender=%s, n=%d", gender, len(gender_df))
            models[gender] = self.fit_gender(gender_df, gender)
        
        elapsed = time.time() - start_time
        logger.info("AgeSplineFitter.fit: completed %d genders in %.2fs", 
                    len(models), elapsed)
        
        return models
    
    
    def fit_gender(
        self,
        gender_df: pd.DataFrame,
        gender: str
    ) -> AgeSplineModel:
        """
        Построение модели для одного пола.
        
        gender_df уже содержит колонки: age, z
        
        Шаги:
        1. Global clamp возраста к [age_min_global, age_max_global]
        2. Нормировка: x = (age - age_center) / age_scale
        3. Вычисление a_min_g, a_max_g, age_range_years (для статистики)
        4. Проверка условий деградации (n, age_range_years)
        5. Построение узлов knots_x (в шкале x!)
        6. Построение центрированного базиса через C
        7. Выбор λ по REML/GCV
        8. Оценка коэффициентов μ, γ, β
        9. Winsorization (если включен)
        10. Вычисление дисперсий: sigma2_reml, tau2_bar, sigma2_use, nu
        11. Заполнение fit_report
        12. Возврат AgeSplineModel
        """
        # Реализуется по шагам (см. раздел 16 — порядок реализации)
        pass
```

---

## **4. Узлы и "сырой" B-сплайновый базис**

### **4.1. Функция build_knots_x**

```python
def build_knots_x(
    x_values: pd.Series,
    degree: int,
    max_inner_knots: int,
    min_knot_gap: float
) -> list[float]:
    """
    Построение полного списка узлов для B-сплайна.
    
    Входы:
    - x_values: обучающие значения x (в шкале x, НЕ age!)
    - degree: степень сплайна (обычно 3)
    - max_inner_knots: макс. число внутренних узлов
    - min_knot_gap: минимальный разрыв между узлами (в единицах x)
    
    Выход:
    - knots_x: полный список узлов, включая граничные с кратностью degree+1
    
    Правило внутренних узлов:
    1. Базово по квантилям x_values (равномерная сетка квантилей)
    2. Слияние узлов при нарушении min_knot_gap:
       - детерминированно удалять ближайший правый, повторять
    3. Ограничение на max_inner_knots
    
    Граничные узлы:
    - На x_min: degree+1 повторений
    - На x_max: degree+1 повторений
    
    Гарантия: одинаковые x_values → идентичный knots_x
    
    Пример:
    x_values в диапазоне [-1.7, 4.5]
    degree = 3, max_inner_knots = 5, min_knot_gap = 0.2
    
    Внутренние узлы по квантилям: [-0.8, 0.5, 1.8, 3.0, 4.0]
    (после проверки min_knot_gap и слияния если нужно)
    
    Полный knots_x:
    [-1.7, -1.7, -1.7, -1.7,  # граничные (degree+1)
     -0.8, 0.5, 1.8, 3.0, 4.0,  # внутренние
     4.5, 4.5, 4.5, 4.5]  # граничные (degree+1)
    """
    pass
```

### **4.2. Построение "сырого" базиса B_raw**

```python
from scipy.interpolate import BSpline

def build_raw_basis(
    x_values: np.ndarray,
    knots_x: list[float],
    degree: int
) -> np.ndarray:
    """
    Построение матрицы "сырого" B-сплайнового базиса.
    
    Входы:
    - x_values: точки вычисления (n,) в шкале x
    - knots_x: полный список узлов с граничными кратностями
    - degree: степень сплайна
    
    Выход:
    - B_raw: матрица (n, K_raw), где K_raw — число базисных функций
    
    Каждый столбец = значения одной B-сплайновой базисной функции.
    
    ОПТИМИЗАЦИЯ: Цикл по K_raw может быть узким местом.
    Для MVP оставляем как есть, замеряем время в fit_report["timings"].
    """
    K_raw = len(knots_x) - degree - 1
    n = len(x_values)
    B_raw = np.zeros((n, K_raw))
    
    knots_array = np.array(knots_x)
    for j in range(K_raw):
        # Единичный вектор коэффициентов для j-й базисной функции
        coef = np.zeros(K_raw)
        coef[j] = 1.0
        
        # Создаем B-сплайн и вычисляем в точках x_values
        spline_j = BSpline(knots_array, coef, degree, extrapolate=False)
        B_raw[:, j] = spline_j(x_values)
    
    return B_raw


def build_raw_basis_derivative(
    x_values: np.ndarray,
    knots_x: list[float],
    degree: int
) -> np.ndarray:
    """
    Производная "сырого" базиса.
    
    Выход:
    - B_raw_prime: матрица (n, K_raw) производных базисных функций
    
    Используется для центрирования: b0_prime = B_raw'(x0)
    """
    K_raw = len(knots_x) - degree - 1
    n = len(x_values)
    B_raw_prime = np.zeros((n, K_raw))
    
    knots_array = np.array(knots_x)
    for j in range(K_raw):
        coef = np.zeros(K_raw)
        coef[j] = 1.0
        
        spline_j = BSpline(knots_array, coef, degree, extrapolate=False)
        # Производная B-сплайна
        spline_j_prime = spline_j.derivative(nu=1)
        B_raw_prime[:, j] = spline_j_prime(x_values)
    
    return B_raw_prime
```

---

## **5. Центрирование базиса через матрицу C**

### **5.1. Требование центрирования**

Сплайновая часть задаётся **линейными ограничениями в опорной точке x0 = 0**:
- `h(0) = 0` (значение)
- `h'(0) = 0` (производная)

Это эквивалентно исключению компонент {1, x} из сплайновой части и **не зависит от состава обучающей выборки**.

### **5.2. Схема центрирования**

**Метод: явная матрица C как базис нулевого пространства ограничений A.**

#### Шаги в fit_gender:

1. **Строится B_raw** размера (n, K_raw) по knots_x и degree на обучающих x

2. **Опорная точка** `x0 = 0.0` (так как x0 = (age_center - age_center) / age_scale)

3. **Вычисляются в точке x0 = 0**:
   - `b0 = B_raw(0)` — вектор значений базиса, размер K_raw
   - `b0_prime = B_raw'(0)` — вектор производных базиса, размер K_raw

4. **Строится матрица ограничений A** размера (2, K_raw):
   ```
   A = [[b0],
        [b0_prime]]
   ```

5. **Строится матрица C** размера (K_raw, K_cent) через **ФИКСИРОВАННЫЙ SVD**:
   - `A @ C = 0`
   - `rank(C) = K_cent = K_raw - 2`
   - Детерминированная ориентация столбцов
   
6. **Центрированный базис**:
   ```
   B_cent(x) = B_raw(x) @ C
   ```

7. **В дизайн модели входят столбцы**:
   - `intercept = 1`
   - `x` (нормированный возраст)
   - `B_cent(x)` (K_cent столбцов)

8. **Сплайновая часть**:
   ```
   h_g(x) = B_cent(x) @ β
   ```
   По построению `h_g(0) = 0` и `h'_g(0) = 0` для любых β.

### **5.3. Функция построения C — ДЕТЕРМИНИРОВАННАЯ**

```python
def build_centering_matrix(
    knots_x: list[float],
    degree: int,
    x0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Построение матриц A и C для центрирования базиса.
    
    Входы:
    - knots_x: полный список узлов
    - degree: степень сплайна
    - x0: опорная точка центрирования (обычно 0.0)
    
    Выход:
    - A: матрица ограничений (2, K_raw)
    - C: базис нулевого пространства (K_raw, K_cent)
    
    ДЕТЕРМИНИРОВАННЫЙ МЕТОД:
    1. Вычисление b0 = B_raw(x0) и b0_prime = B_raw'(x0)
    2. Формирование A = [b0; b0_prime]
    3. Нахождение C через SVD (ФИКСИРОВАННЫЙ):
       U, s, Vt = np.linalg.svd(A.T, full_matrices=True)
       Берём столбцы U, соответствующие нулевым сингулярным значениям
    4. ДЕТЕРМИНИРОВАННАЯ ОРИЕНТАЦИЯ:
       Для каждого столбца C[:,j]:
       - найти элемент с максимальным |C[i,j]|
       - если C[i,j] < 0, умножить весь столбец на -1
    5. Порядок столбцов: как выдаёт SVD, без перестановок
    
    Проверки:
    - |A @ C| < tol (например 1e-10)
    - rank(C) = K_raw - 2
    """
    K_raw = len(knots_x) - degree - 1
    
    # 1. Вычисление A
    b0 = build_raw_basis(np.array([x0]), knots_x, degree)[0, :]  # (K_raw,)
    b0_prime = build_raw_basis_derivative(np.array([x0]), knots_x, degree)[0, :]  # (K_raw,)
    A = np.vstack([b0, b0_prime])  # (2, K_raw)
    
    # 2. SVD для нулевого пространства
    U, s, Vt = np.linalg.svd(A.T, full_matrices=True)
    tol = 1e-12
    null_mask = (s < tol)
    C = U[:, null_mask]  # (K_raw, K_cent)
    
    # 3. Детерминированная ориентация
    for j in range(C.shape[1]):
        i_max = np.argmax(np.abs(C[:, j]))
        if C[i_max, j] < 0:
            C[:, j] *= -1
    
    # 4. Проверки
    assert np.allclose(A @ C, 0.0, atol=1e-10), "Centering constraint violated"
    K_cent_expected = K_raw - 2
    assert C.shape[1] == K_cent_expected, \
        f"Expected K_cent={K_cent_expected}, got {C.shape[1]}"
    
    return A, C
```

### **5.4. Что хранить в basis_centering**

```python
basis_centering = {
    "centering_method": "constraints_value_slope_at_x0",
    "x0": 0.0,              # опорная точка (всегда 0 для проектной нормировки)
    "A": np.ndarray,        # форма (2, K_raw)
    "C": np.ndarray,        # форма (K_raw, K_cent)
    "K_raw": int,           # число базисных функций "сырого" сплайна
    "K_cent": int,          # число центрированных функций = K_raw - 2
}
```

**НЕ хранить**:
- Обучающие данные (X_lin, x_values)
- Проекции (P)
- Любые параметры "вычитания по выборке"

### **5.5. Самопроверки при fit (ОБЯЗАТЕЛЬНЫЕ)**

```python
# 1. ОБЯЗАТЕЛЬНАЯ проверка: A @ C близко к нулю
A = basis_centering["A"]
C = basis_centering["C"]
assert np.allclose(A @ C, 0.0, atol=1e-10), "Centering constraint violated"

# 2. ОБЯЗАТЕЛЬНАЯ численная проверка центрирования на случайном β
np.random.seed(42)  # для воспроизводимости
beta_test = np.random.randn(K_cent)

# Вычисляем h(x0) и h'(x0)
x0 = 0.0
B_raw_x0 = build_raw_basis(np.array([x0]), knots_x, degree)  # (1, K_raw)
B_cent_x0 = B_raw_x0 @ C  # (1, K_cent)
h_x0 = (B_cent_x0 @ beta_test)[0]

B_raw_x0_prime = build_raw_basis_derivative(np.array([x0]), knots_x, degree)
B_cent_x0_prime = B_raw_x0_prime @ C
h_prime_x0 = (B_cent_x0_prime @ beta_test)[0]

assert np.abs(h_x0) < 1e-8, f"h(x0) = {h_x0}, expected ≈ 0"
assert np.abs(h_prime_x0) < 1e-8, f"h'(x0) = {h_prime_x0}, expected ≈ 0"

logger.info("Centering check passed: |h(x0)| < 1e-8, |h'(x0)| < 1e-8")

# 3. ОПЦИОНАЛЬНАЯ диагностика (не обязательное условие):
# Регрессия B_cent на {1, x} по обучающим точкам
# Это НЕ часть определения центрирования, только информация
X_lin = np.column_stack([np.ones(n), x_values])
for j in range(K_cent):
    coef, _, _, _ = np.linalg.lstsq(X_lin, B_cent[:, j], rcond=None)
    if not np.allclose(coef, 0.0, atol=1e-3):
        logger.warning(f"Column {j} has non-zero projection on {{1,x}}: {coef}")
```

**ВАЖНО**: Регрессия на {1, x} — только диагностика, не требование.

---

## **6. Оценивание параметров и выбор λ**

### **6.1. Модель с штрафом**

Для пола g оцениваются параметры:
- `μ_g` — интерсепт
- `γ_g` — коэффициент при x
- `β_g` — коэффициенты центрированного сплайна (размер K_cent)

**Штрафуемая МНК**:
```
Minimize: ||z - W @ θ||² + λ * ||D @ β||²
```
где:
- `W = [1, x, B_cent]` — дизайн-матрица размера (n, 2 + K_cent)
- `θ = [μ, γ, β]` — все параметры
- Штраф применяется только к β (интерсепт и линейный тренд не штрафуются)

**Матрица штрафа D**:
- Штраф на вторую производную (D2) по порядку базисных функций в coef_beta
- Порядок фиксируется детерминированно при построении C

### **6.2. Выбор λ по REML или GCV**

λ выбирается детерминированно по сетке `lambda_grid` в лог-шкале.

**Критерий**:
- **REML** (желательно): маргинальное правдоподобие в линейной смешанной формулировке
- **GCV** (временная замена): Generalized Cross-Validation

**На выходе сохраняются**:
- `lambda_value` — выбранный параметр гладкости
- `sigma2_reml` — остаточная дисперсия
- `edf` — effective degrees of freedom (в fit_report)
- `fit_report["lambda_method"]` — "REML" или "GCV"

### **6.3. Решение штрафуемой МНК**

```python
def solve_penalized_lsq(
    W: np.ndarray,        # (n, p) где p = 2 + K_cent
    z: np.ndarray,        # (n,)
    lambda_val: float,    # штраф
    penalty_idx: list[int]  # индексы столбцов для штрафа (обычно [2, 3, ..., p-1])
) -> tuple[np.ndarray, dict]:
    """
    Решение штрафуемой МНК через QR-разложение.
    
    Minimize: ||z - W @ θ||² + λ * ||D @ θ_penalty||²
    
    где θ_penalty — подвектор θ соответствующий penalty_idx.
    
    Augmented system:
    [ W           ]       [ z ]
    [ sqrt(λ)*D ] @ θ ≈  [ 0 ]
    
    Решается через QR.
    
    Выход:
    - theta: вектор коэффициентов (p,)
    - info: {"edf": float, "R_diag": list, ...}
    """
    pass
```

---

## **7. Winsorization и устойчивость**

### **7.1. Протокол Winsor**

Перед финальной оценкой σ² допускается winsor по остаткам:

1. **Первоначальная подгонка**: получить `z_hat`, остатки `r = z - z_hat`

2. **Оценка медианы и MAD**:
   ```python
   median_r = np.median(r)
   mad_r = np.median(np.abs(r - median_r))
   ```

3. **Зажим остатков**:
   ```python
   k = self.winsor_k  # из config, например 3.0
   lower = median_r - k * mad_r
   upper = median_r + k * mad_r
   
   r_clipped = np.clip(r, lower, upper)
   z_winsor = z_hat + r_clipped
   ```

4. **Повторная подгонка** на зажатой целевой переменной `z_winsor`

5. **Сохранение параметров**:
   ```python
   winsor_params = {
       "enabled": True,
       "k": k,
       "median": median_r,
       "mad": mad_r,
       "lower": lower,
       "upper": upper,
       "fraction_clamped": float(((r < lower) | (r > upper)).mean())
   }
   ```

### **7.2. Управление через config**

```yaml
age_spline_model:
  winsor_enabled: true
  winsor_k: 3.0  # порог = k * MAD
```

Winsor включается одинаково в LOY и production мирах.

---

## **8. Дисперсии: tau2_bar, sigma2_reml, sigma2_use, nu**

### **8.1. tau2_bar — средняя дисперсия эталона**

Определяется как средняя величина τ², соответствующая дисперсии эталона.

**Вычисление**:
```python
def compute_tau2_bar(
    gender_df: pd.DataFrame,
    trace_references: pd.DataFrame,
    gender: str
) -> float:
    """
    Вычисление средней дисперсии эталона для пола.
    
    Входы:
    - gender_df: данные для пола с колонкой race_id
    - trace_references: таблица с reference_variance по (race_id, gender)
    - gender: "M" или "F"
    
    Выход:
    - tau2_bar: среднее по строкам gender_df
    
    Если reference_variance недоступна, возвращает 0.0 с предупреждением.
    """
    if "reference_variance" not in trace_references.columns:
        logger.warning("reference_variance not in trace_references, using tau2_bar=0")
        return 0.0
    
    merged = gender_df.merge(
        trace_references[["race_id", "gender", "reference_variance"]],
        on=["race_id", "gender"],
        how="left"
    )
    
    tau2_bar = merged["reference_variance"].mean()
    
    if pd.isna(tau2_bar):
        logger.warning("tau2_bar is NA, using 0.0")
        return 0.0
    
    return float(tau2_bar)
```

### **8.2. sigma2_reml — остаточная дисперсия**

Остаточная дисперсия на шкале z по результату REML (или GCV).

### **8.3. sigma2_use — используемая дисперсия**

```python
sigma2_use = max(sigma2_floor, sigma2_reml - tau2_bar)
```

где:
- `sigma2_floor` — из config (например, 1e-8)
- Обязателен для численной стабильности

### **8.4. nu — эффективные степени свободы**

```python
nu = max(nu_floor, n - edf)
```

где:
- `edf` — из итоговой подгонки (effective degrees of freedom)
- `nu_floor` — из config (например, 3.0)

---

## **9. Деградация при малых данных**

### **9.1. Режимы деградации**

В `fit_gender` обязательны **явные режимы деградации**:

```python
n = len(gender_df)
age_range_years = gender_df["age"].max() - gender_df["age"].min()

if n < self.n_min_linear or age_range_years < self.range_min_linear:
    # Режим: constant_only
    fit_report["degrade"] = "constant_only"
    # Модель: m = μ
    # coef_gamma = 0.0
    # coef_beta = pd.Series([], dtype=float)
    
elif n < self.n_min_spline or age_range_years < self.range_min_years:
    # Режим: linear_only
    fit_report["degrade"] = "linear_only"
    # Модель: m = μ + γ*x
    # coef_beta = pd.Series([], dtype=float)
    # Сплайновых столбцов нет
    
else:
    # Нормальный режим: spline
    fit_report["degrade"] = "none"
    # Модель: m = μ + γ*x + h(x)
    # Все компоненты присутствуют
```

### **9.2. Сохранение артефакта**

Любая деградация сохраняет артефакт в том же формате:
- `coef_beta` может быть пустым pd.Series
- `knots_x` может быть минимальным
- `basis_centering["C"]` может быть пустой матрицей

**Контракт остаётся неизменным.**

---

## **10. Изменения в config.yaml**

```yaml
preprocessing:
  age_center: 35      # ЕДИНСТВЕННЫЙ источник центра возраста в проекте
  age_scale: 10       # масштаб для x = (age - age_center) / age_scale

age_spline_model:
  # B-сплайн параметры
  degree: 3                    # кубический сплайн
  max_inner_knots: 10          # максимум внутренних узлов
  min_knot_gap: 0.2            # минимальный разрыв в x-шкале (= 2 года)
  
  # Глобальные границы возраста (для clamp при обучении)
  age_min_global: 18
  age_max_global: 80
  
  # Выбор λ
  lambda_grid_log_min: -6      # log10(lambda_min) = 1e-6
  lambda_grid_log_max: 2       # log10(lambda_max) = 100
  lambda_grid_n: 50            # число точек в сетке
  lambda_method: "GCV"         # "REML" или "GCV"
  
  # Пороги деградации
  n_min_spline: 300            # минимум наблюдений для сплайна
  range_min_years: 15          # минимальный диапазон возрастов для сплайна
  n_min_linear: 80             # минимум для линейной модели
  range_min_linear: 8          # минимальный диапазон для линейной
  
  # Дисперсии
  sigma2_floor: 1.0e-8         # минимальная дисперсия
  nu_floor: 3.0                # минимальные степени свободы
  
  # Winsorization
  winsor_enabled: true
  winsor_k: 3.0                # порог = k * MAD
  
  # Диагностика
  centering_tol: 1.0e-10       # допуск для проверки A @ C ≈ 0
  verbose_fit: true            # детальное логирование
```

---

## **11. Изменения в MarathonModel**

### **11.1. Атрибуты**

```python
class MarathonModel:
    ...
    age_spline_models: dict[str, AgeSplineModel] | None  # {"M": model_m, "F": model_f}
    _age_spline_fitter: AgeSplineFitter | None
```

### **11.2. Метод __init__**

```python
def __init__(self, ...) -> None:
    ...
    self.age_spline_models = None
    self._age_spline_fitter = AgeSplineFitter(self.config)
```

### **11.3. Метод fit_age_model**

```python
def fit_age_model(self) -> "MarathonModel":
    """
    Обучает возрастную модель на сплайнах.
    
    Шаги:
    1. Проверка зависимостей: preprocess, build_trace_references
    2. Фильтрация train_frame:
       - status == "OK"
       - year != validation_year (если validation_year > 0)
    3. Проверка обязательных колонок: race_id, gender, age, Y
    4. ВЫЧИСЛЕНИЕ z = Y - ln(R_use):
       - Джойн с trace_references
       - z = Y - reference_log
    5. Вызов fitter.fit(train_frame[["gender", "age", "z"]])
    6. Сохранение результата: self.age_spline_models = models_dict
    7. Сохранение в .pkl файл (опционально)
    8. Логирование
    
    Returns: self
    """
    self._check_step("fit_age_model", ["preprocess", "build_trace_references"])
    
    start_time = time.time()
    rows_in = len(self.df_clean)
    
    # 1-2. Подготовка обучающего фрейма
    train_frame = self.df_clean.copy()
    
    if "status" in train_frame.columns:
        train_frame = train_frame[train_frame["status"] == "OK"].copy()
    
    rows_before_year_drop = len(train_frame)
    if self.validation_year > 0 and "year" in train_frame.columns:
        train_frame = train_frame[train_frame["year"] != self.validation_year].copy()
    dropped_by_year = rows_before_year_drop - len(train_frame)
    
    # 3. Проверка обязательных колонок
    required_columns = ["race_id", "gender", "age", "Y"]
    missing_columns = [col for col in required_columns if col not in train_frame.columns]
    if missing_columns:
        raise ValueError(f"fit_age_model: missing columns: {missing_columns}")
    
    rows_before_na_drop = len(train_frame)
    train_frame = train_frame.dropna(subset=required_columns).copy()
    dropped_by_na = rows_before_na_drop - len(train_frame)
    
    # 4. ВЫЧИСЛЕНИЕ z = Y - ln(R_use)
    train_frame = train_frame.merge(
        self.trace_references[["race_id", "gender", "reference_log"]],
        on=["race_id", "gender"],
        how="left"
    )
    
    # Проверка, что все эталоны найдены
    if train_frame["reference_log"].isna().any():
        n_missing = train_frame["reference_log"].isna().sum()
        raise ValueError(f"fit_age_model: {n_missing} rows missing trace references")
    
    train_frame["z"] = train_frame["Y"] - train_frame["reference_log"]
    
    rows_out = len(train_frame)
    
    # 5. Фитирование возрастной модели
    # Фиттер получает только нужные колонки: gender, age, z
    fit_df = train_frame[["gender", "age", "z"]].copy()
    
    self.age_spline_models = self._age_spline_fitter.fit(df=fit_df)
    
    # 6. Сохранение в файл (если включено)
    if self.dumping_info.get("age_spline_model", False):
        output_path = Path("artifacts/age_spline_models.pkl")
        save_age_spline_models(
            self.age_spline_models, 
            output_path,
            metadata={
                "validation_year": self.validation_year,
                "n_total": rows_out
            }
        )
        logger.info("Age spline models saved: %s", output_path)
    
    # 7. Логирование
    self._steps_completed["fit_age_model"] = True
    self._log_step(
        "fit_age_model",
        rows_in=rows_in,
        rows_out=rows_out,
        elapsed_sec=(time.time() - start_time),
        extra=f"validation_year={self.validation_year}, dropped_by_year={dropped_by_year}, dropped_by_na={dropped_by_na}"
    )
    
    return self
```

---

## **12. Логи и отчёты**

### **12.1. На уровне каждого пола (в fit_report)**

```python
fit_report = {
    # Данные
    "n": int,                      # число наблюдений после фильтрации
    "age_range_actual": tuple,     # (a_min_g, a_max_g)
    "age_range_years": float,      # a_max_g - a_min_g
    
    # Сплайн
    "knots_count_inner": int,
    "degree": int,
    "K_raw": int,                  # число "сырых" базисных функций
    "K_cent": int,                 # число центрированных функций
    
    # Гладкость
    "lambda_value": float,
    "lambda_method": str,          # "REML" или "GCV"
    
    # Степени свободы
    "edf": float,                  # effective degrees of freedom
    "nu": float,                   # для t-распределения
    
    # Дисперсии
    "sigma2_reml": float,
    "tau2_bar": float,
    "sigma2_use": float,
    
    # Качество на обучении
    "rmse_z": float,               # RMSE по z после финальной подгонки
    "mae_z": float,                # MAE по z
    
    # Деградация
    "degrade": str,                # "none", "linear_only", "constant_only"
    
    # Winsor
    "winsor": dict,                # enabled, k, median, mad, fraction_clamped
    
    # Тайминги
    "timings": {
        "build_basis_sec": float,
        "solve_penalized_sec": float,
        "total_sec": float
    },
    
    # Предупреждения
    "warnings": list[str]          # ["Too few unique ages", "Knots collapsed", ...]
}
```

### **12.2. Логи при обучении**

```
INFO: fit_age_model: start rows=8234
INFO: fit_gender: gender=M, n=5234
INFO: fit_gender: gender=M, age_range=[18.0, 79.0], age_range_years=61.0
INFO: fit_gender: gender=M, knots_inner=7, K_raw=10, K_cent=8
INFO: fit_gender: gender=M, lambda=0.0123 (method=GCV), edf=6.42, nu=5227.58
INFO: fit_gender: gender=M, sigma2_reml=0.0202, tau2_bar=0.0015, sigma2_use=0.0187
INFO: fit_gender: gender=M, rmse_z=0.136, mae_z=0.098, elapsed=2.34s
INFO: fit_gender: gender=F, n=2891
INFO: fit_gender: gender=F, age_range=[19.0, 75.0], age_range_years=56.0
INFO: fit_gender: gender=F, knots_inner=7, K_raw=10, K_cent=8
INFO: fit_gender: gender=F, lambda=0.0187 (method=GCV), edf=5.89, nu=2885.11
INFO: fit_gender: gender=F, sigma2_reml=0.0245, tau2_bar=0.0018, sigma2_use=0.0227
INFO: fit_gender: gender=F, rmse_z=0.151, mae_z=0.109, elapsed=1.87s
INFO: AgeSplineFitter.fit: completed 2 genders in 4.21s
INFO: fit_age_model: rows 8234 → 8125, dropped_by_year=109, dropped_by_na=0, elapsed=4.21s
```

---

## **13. Сохранение и загрузка**

### **13.1. Функции сохранения**

```python
import pickle
from datetime import datetime
from pathlib import Path

def save_age_spline_models(
    models: dict[str, AgeSplineModel],
    output_path: str | Path,
    metadata: dict | None = None
) -> Path:
    """
    Сохранение словаря моделей в .pkl файл.
    
    Входы:
    - models: {"M": model_m, "F": model_f}
    - output_path: путь к .pkl файлу
    - metadata: дополнительные данные (дата обучения, версия кода)
    
    Сохраняет:
    - models (dict с AgeSplineModel)
    - metadata (дата, версия, validation_year, etc.)
    
    Выход: Path к созданному файлу
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "models": models,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            **(metadata or {})
        }
    }
    
    with open(output_file, "wb") as f:
        pickle.dump(artifact, f)
    
    logger.info("Age spline models saved: %s, genders=%s", 
                output_file, list(models.keys()))
    return output_file


def load_age_spline_models(
    input_path: str | Path
) -> dict[str, AgeSplineModel]:
    """
    Загрузка моделей из .pkl файла.
    
    Вход: путь к .pkl файлу
    Выход: {"M": model_m, "F": model_f}
    """
    with open(input_path, "rb") as f:
        artifact = pickle.load(f)
    
    models = artifact["models"]
    metadata = artifact.get("metadata", {})
    
    logger.info("Age spline models loaded: %s, created_at=%s, genders=%s",
                input_path, metadata.get("created_at"), list(models.keys()))
    
    return models
```

### **13.2. Версионирование metadata**

```python
metadata = {
    "created_at": "2025-01-15T14:32:00",
    "validation_year": 2025,
    "n_total": 8125,
    "genders": ["M", "F"],
    "age_center": 35,
    "age_scale": 10
}
```

---

## **14. Контракты и инварианты**

### **14.1. Перед fit()**

```python
# Обязательные колонки в df
assert {"gender", "age", "z"}.issubset(df.columns)

# Нет NA в обязательных колонках
assert not df[["gender", "age", "z"]].isna().any().any()

# Возраст в разумных пределах
# (проверка выполняется внутри fit_gender через global clamp)
```

### **14.2. После fit()**

```python
# models содержит те полы, что были в df
genders_in_df = set(df["gender"].unique())
assert set(models.keys()) == genders_in_df

# Для каждой модели
for gender, model in models.items():
    # sigma2_use > 0
    assert model.sigma2_use > 0, f"Gender {gender}: sigma2_use <= 0"
    
    # Центрирование проверено при обучении (A @ C ≈ 0)
    
    # Деградация зафиксирована
    assert model.fit_report["degrade"] in ["none", "linear_only", "constant_only"]
    
    # Если degrade == "none", то coef_beta не пустой
    if model.fit_report["degrade"] == "none":
        assert len(model.coef_beta) > 0
```

### **14.3. При predict()**

```python
# gender обучен
assert gender in models, f"Gender {gender} not trained"

# Нет ограничений на age — predict работает для любого возраста
# (экстраполяция естественная через B-сплайны)
```

### **14.4. Проверка центрирования в predict**

```python
# m(age_center) = μ + γ * x0 = μ + γ * 0 = μ
age_center = model.age_center
m_at_center = model.predict_mean(age_center)

expected = model.coef_mu  # так как x0 = 0, то γ*x0 = 0

assert np.abs(m_at_center - expected) < 1e-6, \
    f"m(age_center) = {m_at_center}, expected {expected}"

# Отдельно: h(0) = 0
x0 = 0.0
B_raw_x0 = build_raw_basis(np.array([x0]), model.knots_x, model.degree)
B_cent_x0 = B_raw_x0 @ model.basis_centering["C"]
h_x0 = (B_cent_x0 @ model.coef_beta.values)[0]

assert np.abs(h_x0) < 1e-8, f"h(x0) = {h_x0}, expected ≈ 0"
```

---

## **15. Ограничения и границы ответственности**

### **15.1. AgeSplineFitter**
- Получает **готовый df** с колонками ["gender", "age", "z"]
- Все параметры считаны в `__init__` из config
- В `fit` НЕ обращается к config
- Никаких LOY-логики внутри
- Возвращает `dict[gender, AgeSplineModel]`

### **15.2. MarathonModel.fit_age_model**
- Выполняет LOY фильтрацию (year != validation_year)
- Вычисляет z = Y - ln(R_use)
- Подготавливает df для фиттера
- Сохраняет результат в `self.age_spline_models`
- Опционально записывает в .pkl

---

## **16. Порядок реализации (пошаговый план)**

### **Этап 0: Подготовка структуры (1 час)**
- Создать `age_spline_model.py` и `age_spline_fit.py`
- Объявить dataclass `AgeSplineModel` с заглушками
- Класс `AgeSplineFitter` с `__init__` и заглушками
- Добавить секцию в `config.yaml`
- Обновить `MarathonModel.__init__`

**Результат**: структура готова, импорты работают

---

### **Этап 1: Нормировка возраста (1 час)**

**Задачи**:
1. Функция `normalize_age(age, age_center, age_scale) -> x`
2. Тесты на нормировку

**Код**:
```python
def normalize_age(age: float, age_center: float, age_scale: float) -> float:
    return (age - age_center) / age_scale
```

**Проверка**: 
- age=35 → x=0
- age=45 → x=1.0
- age=18 → x=-1.7

---

### **Этап 2: Узлы и "сырой" базис (2-3 часа)**

**Задачи**:
1. `build_knots_x(x_values, degree, max_inner_knots, min_knot_gap)`
2. `build_raw_basis(x_values, knots_x, degree)`
3. `build_raw_basis_derivative(x_values, knots_x, degree)`
4. Тесты на детерминированность узлов

**Проверка**: 
- Одинаковые x_values → идентичные knots
- B_raw правильной размерности
- Сумма базисных функций ≈ 1

---

### **Этап 3: Центрирование через C (3-4 часа)**

**Задачи**:
1. `build_centering_matrix(knots_x, degree, x0=0.0) -> (A, C)`
2. SVD для нахождения C
3. Детерминированная ориентация столбцов
4. Проверки: A @ C ≈ 0, численная проверка h(0)=0 и h'(0)=0

**Проверка**:
- |A @ C| < 1e-10
- Для случайного β: |h(0)| < 1e-8, |h'(0)| < 1e-8

---

### **Этап 4: Штрафуемая МНК (4-5 часов)**

**Задачи**:
1. `solve_penalized_lsq(W, z, lambda, penalty_idx)`
2. `select_lambda_gcv(W, z, lambda_grid, penalty_idx)`
3. Вычисление edf

**Проверка**:
- Решение штрафуемой системы корректно
- edf убывает с ростом λ

---

### **Этап 5: Дисперсии (2 часа)**

**Задачи**:
1. `compute_tau2_bar(gender_df, trace_references, gender)`
2. sigma2_use = max(floor, sigma2_reml - tau2_bar)
3. nu = max(floor, n - edf)

**Проверка**:
- tau2_bar >= 0
- sigma2_use >= sigma2_floor
- nu >= nu_floor

---

### **Этап 6: Winsorization (1-2 часа)**

**Задачи**:
1. `apply_winsor(z, z_hat, k) -> (z_winsor, params)`
2. Повторная подгонка

**Проверка**:
- fraction_clamped < 0.1 при разумных данных

---

### **Этап 7: Деградация (1-2 часа)**

**Задачи**:
1. `determine_degradation(n, age_range_years, config)`
2. Ветвления: constant_only, linear_only, spline

**Проверка**:
- Артефакт единообразен независимо от режима

---

### **Этап 8: fit_gender — сборка (2-3 часа)**

**Задачи**:
1. Реализация полного `fit_gender`
2. Интеграция всех предыдущих этапов
3. Заполнение `AgeSplineModel` и `fit_report`

**Проверка**:
- Нормальные данные: обучается без ошибок
- Малые данные: срабатывает деградация

---

### **Этап 9: fit — оркестратор (1 час)**

**Задачи**:
1. `AgeSplineFitter.fit(df) -> dict[str, AgeSplineModel]`
2. Цикл по полам
3. Логирование

---

### **Этап 10: predict_mean и design_row (1-2 часа)**

**Задачи**:
1. `AgeSplineModel.predict_mean(age)`
2. `AgeSplineModel.design_row(age)`

**Проверка**:
- predict_mean(age_center) ≈ coef_mu (так как x0=0)
- design_row возвращает правильные ключи

---

### **Этап 11: Интеграция в MarathonModel (1-2 часа)**

**Задачи**:
1. Обновление `MarathonModel.fit_age_model()`
2. Вычисление z = Y - ln(R_use)
3. Вызов фиттера
4. Сохранение результата

**Проверка**:
- Pipeline `run()` работает до fit_age_model

---

### **Этап 12: Сохранение/загрузка (1 час)**

**Задачи**:
1. `save_age_spline_models(models, path, metadata)`
2. `load_age_spline_models(path)`

**Проверка**:
- Сохранение → загрузка → идентичность

---

### **Этап 13: End-to-end тест (2 часа)**

**Задачи**:
1. Полный pipeline от загрузки до fit_age_model
2. Диагностика на реальных данных

**Проверка**:
- Pipeline завершается успешно
- fit_report содержит разумные значения

---

## **17. Структура модулей — ДВА ФАЙЛА**

```
age_spline_model.py (~250-300 строк)
├── normalize_age()
├── @dataclass AgeSplineModel
│   ├── predict_mean()
│   └── design_row()
├── save_age_spline_models()
└── load_age_spline_models()

age_spline_fit.py (~400-500 строк)
├── build_knots_x()
├── build_raw_basis()
├── build_raw_basis_derivative()
├── build_centering_matrix()
├── solve_penalized_lsq()
├── select_lambda_gcv()
├── compute_tau2_bar()
├── apply_winsor()
├── determine_degradation()
└── class AgeSplineFitter
    ├── __init__()
    ├── fit()
    └── fit_gender()
```

**Обоснование**:
- `age_spline_model.py` — модель и предсказание (production)
- `age_spline_fit.py` — обучение и вспомогательные функции (training only)

---

## **18. Ключевые отличия от первоначальной версии**

1. **ЕДИНАЯ шкала x** = (age - 35) / 10 для всего проекта
2. **x0 = 0.0 ВСЕГДА** (опорная точка = age_center)
3. **Фиксированная нормировка**, не зависящая от данных
4. **Центрирование в x=0**: h(0)=0, h'(0)=0
5. **Детерминированный SVD** с фиксированной ориентацией столбцов C
6. **Контракт входа**: фиттер получает ["gender", "age", "z"]
7. **Вычисление z** в MarathonModel.fit_age_model, не внутри фиттера
8. **age_center** из preprocessing — единственный источник
9. **Global clamp** применяется только при обучении
10. **Два файла** для лучшей читаемости

---

**Спецификация готова к реализации.**
