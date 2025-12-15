"""
Примеры использования age_references в MarathonModel.
"""

from logging_setup import easy_logging
from main import MarathonModel

# Отключаем логи для чистоты вывода
easy_logging(False)

# Создаем и запускаем модель
model = MarathonModel(
    data_path=r"C:\Users\andre\github\Marathons\Data",
    validation_year=2025,
    verbose=False
)
model.run()

print("=" * 70)
print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ age_references")
print("=" * 70)

# ==================== Пример 1: Список трасс ====================
print("\n1. Список доступных трасс:")
print("-" * 70)
races = list(model.age_references.keys())
print(f"Всего трасс: {len(races)}")
for i, race in enumerate(races, 1):
    df = model.age_references[race]
    print(f"  {i}. {race} - {len(df)} возрастных групп")

# ==================== Пример 2: Прямой доступ ====================
print("\n2. Прямой доступ к трассе:")
print("-" * 70)
if 'Белые ночи' in model.age_references:
    white_nights = model.age_references['Белые ночи']
    print(f"Белые ночи: {len(white_nights)} групп")
    print("\nПервые 5 строк:")
    print(white_nights.head().to_string(index=False))
else:
    print("Трасса 'Белые ночи' не найдена")

# ==================== Пример 3: Безопасный доступ через метод ====================
print("\n3. Безопасный доступ через get_age_references():")
print("-" * 70)
white_nights = model.get_age_references('Белые ночи')
if white_nights is not None:
    print(f"✓ Трасса найдена: {len(white_nights)} групп")
else:
    print("✗ Трасса не найдена")

# Попытка получить несуществующую трассу
fake_race = model.get_age_references('Несуществующая трасса')
if fake_race is None:
    print("✓ Корректно вернул None для несуществующей трассы")

# ==================== Пример 4: Фильтрация по полу ====================
print("\n4. Фильтрация по полу:")
print("-" * 70)
if 'Дорога жизни' in model.age_references:
    road_of_life = model.age_references['Дорога жизни']

    males = road_of_life[road_of_life['gender'] == 'M']
    females = road_of_life[road_of_life['gender'] == 'F']

    print(f"Дорога жизни:")
    print(f"  Мужчины: {len(males)} возрастных групп")
    print(f"  Женщины: {len(females)} возрастных групп")

    if not males.empty:
        print(f"\nМужчины, первые 3 строки:")
        print(males.head(3)[['age', 'age_median_time', 'n_total']].to_string(index=False))

# ==================== Пример 5: Поиск эталона для конкретного возраста ====================
print("\n5. Поиск эталона для конкретного возраста:")
print("-" * 70)
race_name = list(model.age_references.keys())[0]  # берем первую трассу
race_df = model.age_references[race_name]

target_age = 30
target_gender = 'M'

result = race_df[
    (race_df['age'] == target_age) &
    (race_df['gender'] == target_gender)
    ]

if not result.empty:
    row = result.iloc[0]
    time_minutes = row['age_median_time'] / 60
    print(f"Трасса: {race_name}")
    print(f"Эталон для {target_gender}, возраст {target_age}:")
    print(f"  Медиана времени: {time_minutes:.1f} минут")
    print(f"  Выборка: {row['n_total']} человек")
else:
    print(f"Нет данных для {target_gender}, возраст {target_age}")

# ==================== Пример 6: Сравнение трасс ====================
print("\n6. Сравнение эталонов между трассами:")
print("-" * 70)
if len(model.age_references) >= 2:
    races_list = list(model.age_references.keys())[:2]
    compare_age = 35
    compare_gender = 'M'

    print(f"Сравнение для {compare_gender}, возраст {compare_age}:\n")

    for race_name in races_list:
        race_df = model.age_references[race_name]
        result = race_df[
            (race_df['age'] == compare_age) &
            (race_df['gender'] == compare_gender)
            ]

        if not result.empty:
            time_minutes = result.iloc[0]['age_median_time'] / 60
            print(f"  {race_name:20s}: {time_minutes:.1f} мин")
        else:
            print(f"  {race_name:20s}: нет данных")

# ==================== Пример 7: Итерация по всем трассам ====================
print("\n7. Статистика по всем трассам:")
print("-" * 70)
for race_id, df in model.age_references.items():
    male_count = len(df[df['gender'] == 'M'])
    female_count = len(df[df['gender'] == 'F'])
    age_range = f"{df['age'].min()}-{df['age'].max()}"

    print(f"{race_id:25s}: {male_count:2d}M + {female_count:2d}F, возраст {age_range}")

print("\n" + "=" * 70)
print("ЗАВЕРШЕНО")
print("=" * 70)