"""
Утилиты для красивого отображения времён финиша.
"""

import pandas as pd


def seconds_to_hms(seconds: float) -> str:
    """
    Преобразовать секунды в формат HH:MM:SS.
    
    Args:
        seconds: Время в секундах
        
    Returns:
        Строка формата "HH:MM:SS"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_age_references(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить читаемые колонки с временами в формате HH:MM:SS.
    
    Args:
        df: DataFrame с age_references
        
    Returns:
        DataFrame с дополнительными колонками:
        - median_hms: медиана в формате HH:MM:SS
        - std_hms: стандартное отклонение в формате HH:MM:SS
        - range_hms: диапазон времён "от - до"
        - cv_percent: коэффициент вариации (%)
    """
    result = df.copy()
    
    # Медиана в HH:MM:SS
    result['median_hms'] = result['age_median_time'].apply(seconds_to_hms)
    
    # Стандартное отклонение в HH:MM:SS
    result['std_hms'] = result['age_median_std'].apply(seconds_to_hms)
    
    # Диапазон времён (±1σ)
    lower = result['age_median_time'] - result['age_median_std']
    upper = result['age_median_time'] + result['age_median_std']
    result['range_hms'] = (
        lower.apply(seconds_to_hms) + ' - ' + upper.apply(seconds_to_hms)
    )
    
    # Коэффициент вариации в %
    result['cv_percent'] = (
        result['age_median_std'] / result['age_median_time'] * 100
    ).round(1)
    
    return result


def print_age_references_pretty(df: pd.DataFrame) -> None:
    """
    Красиво вывести таблицу age_references.
    
    Args:
        df: DataFrame с age_references
    """
    formatted = format_age_references(df)
    
    # Колонки для вывода
    columns = [
        'gender', 'age', 
        'median_hms', 'std_hms', 
        'cv_percent', 'n_total'
    ]
    
    print(formatted[columns].to_string(index=False))


def analyze_age_group(df: pd.DataFrame, gender: str, age: int) -> None:
    """
    Детальный анализ одной возрастной группы.
    
    Args:
        df: DataFrame с age_references
        gender: Пол ('M' или 'F')
        age: Возраст
    """
    row = df[(df['gender'] == gender) & (df['age'] == age)]
    
    if row.empty:
        print(f"Нет данных для {gender}, возраст {age}")
        return
    
    row = row.iloc[0]
    
    median_sec = row['age_median_time']
    std_sec = row['age_median_std']
    
    median_hms = seconds_to_hms(median_sec)
    std_hms = seconds_to_hms(std_sec)
    
    lower = median_sec - std_sec
    upper = median_sec + std_sec
    
    cv = (std_sec / median_sec) * 100
    
    print(f"\n{'='*60}")
    print(f"Анализ группы: {gender}, возраст {age}")
    print(f"{'='*60}")
    print(f"Выборка: {row['n_total']} человек")
    print(f"\nМедиана времени финиша:")
    print(f"  {median_hms} ({median_sec:.0f} сек)")
    print(f"\nСтандартное отклонение:")
    print(f"  ±{std_hms} ({std_sec:.0f} сек)")
    print(f"\nДиапазон времён (±1σ):")
    print(f"  {seconds_to_hms(lower)} - {seconds_to_hms(upper)}")
    print(f"\nОтносительный разброс:")
    print(f"  {cv:.1f}% от медианы")
    
    if cv < 5:
        print(f"Очень однородная группа!")
    elif cv < 10:
        print(f"Однородная группа")
    elif cv < 15:
        print(f"! Умеренный разброс")
    else:
        print(f" !!! Большой разброс (неоднородная группа)")
    
    print(f"{'='*60}\n")


# ------
# Примеры использования:

if __name__ == "__main__":
    # Создаем тестовые данные
    test_df = pd.DataFrame({
        'gender': ['F', 'F', 'F', 'F', 'F'],
        'age': [23, 24, 25, 26, 27],
        'age_median_time': [8917.0, 9579.0, 9666.0, 12549.0, 12451.0],
        'age_median_var': [8.294982e+05, 1.455888e+06, 1.920848e+06, 1.824900e+06, 1.652569e+06],
        'age_median_std': [910.767913, 1206.601893, 1385.946712, 1350.888548, 1285.522798],
        'n_total': [53, 55, 63, 83, 89],
    })
    
    print("\n" + "="*60)
    print("КРАСИВЫЙ ВЫВОД ВОЗРАСТНЫХ ЭТАЛОНОВ")
    print("="*60 + "\n")
    
    print_age_references_pretty(test_df)
    
    print("\n" + "="*60)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ")
    print("="*60)
    
    analyze_age_group(test_df, 'F', 23)
    analyze_age_group(test_df, 'F', 26)
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ ВОЗРАСТОВ")
    print("="*60 + "\n")
    
    formatted = format_age_references(test_df)
    print("Как растёт медиана и разброс с возрастом:\n")
    print(formatted[['age', 'median_hms', 'std_hms', 'cv_percent']].to_string(index=False))
